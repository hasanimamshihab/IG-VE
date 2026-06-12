from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/N/project/SingleCell_Image/Ronak/Project_folder")

GE_FILE = PROJECT_ROOT / "data/gene_expression_data.csv"
META_FILE = PROJECT_ROOT / "data/Paired_data_metadata.csv"

SELECTED_PROBE_FILE = PROJECT_ROOT / "outputs/gene_branch/final_selected_probe_list.csv"
MEASURED_AGFV_FILE = PROJECT_ROOT / "outputs/aligned_gfv/aligned_gfv_measured_ge_302_subjects.csv"

AGFV_ABLATION_FILE = PROJECT_ROOT / "outputs/ablation/agfv_nn_feature_ablation.csv"
GE_PROBE_ABLATION_FILE = PROJECT_ROOT / "outputs/ablation/ge_branch_probe_ablation.csv"
GE_GENE_ABLATION_FILE = PROJECT_ROOT / "outputs/ablation/ge_branch_gene_ablation.csv"

OUT_DIR = PROJECT_ROOT / "outputs/ablation"
TABLE_DIR = PROJECT_ROOT / "outputs/tables/supplementary"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

ALL_OUT = OUT_DIR / "agfv_gene_association_all.csv"
TOP_OUT = OUT_DIR / "agfv_gene_association_top.csv"
GENE_OUT = OUT_DIR / "agfv_gene_association_gene_level.csv"
SUMMARY_OUT = OUT_DIR / "agfv_gene_association_summary.txt"
TABLE_OUT = TABLE_DIR / "Table_S20_AGFV_gene_association.xlsx"

TOP_AGFV_FEATURES = 25
TOP_PROBES_PER_AGFV = 100


def to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def get_agfv_cols(df: pd.DataFrame):
    pat = re.compile(r"^AGFV\d+$", re.IGNORECASE)
    cols = [c for c in df.columns if pat.match(c)]
    return sorted(cols, key=lambda x: int(re.findall(r"\d+", x)[0]))


def load_expression_matrix():
    ge = pd.read_csv(GE_FILE, low_memory=False)

    first_col = ge.iloc[:, 0].astype(str).str.strip()
    matches = ge.index[first_col == "ProbeSet"].tolist()

    if not matches:
        raise ValueError("Could not find row labeled ProbeSet in first column.")

    probeset_row = matches[0]
    expr = ge.iloc[probeset_row + 1:].copy()

    probe_col = expr.columns[0]
    locus_col = expr.columns[1]
    symbol_col = expr.columns[2]

    expr[probe_col] = expr[probe_col].astype(str).str.strip()

    return expr, probe_col, locus_col, symbol_col


def prepare_expression_for_measured_agfv():
    selected = pd.read_csv(SELECTED_PROBE_FILE)

    if "Probe ID / feature ID" not in selected.columns:
        raise ValueError("Selected probe file must contain 'Probe ID / feature ID'.")

    selected["Probe ID / feature ID"] = selected["Probe ID / feature ID"].astype(str)
    panel_probes = selected["Probe ID / feature ID"].tolist()

    expr, probe_col, locus_col, symbol_col = load_expression_matrix()

    meta = pd.read_csv(META_FILE)
    agfv = pd.read_csv(MEASURED_AGFV_FILE)

    required_meta = ["subject_id", "ge_sample_column", "diagnosis", "diagnosis_binary", "ge_column_found"]
    missing = [c for c in required_meta if c not in meta.columns]
    if missing:
        raise ValueError(f"Missing metadata columns: {missing}")

    meta = meta.copy()
    meta["subject_id"] = meta["subject_id"].astype(str)
    agfv["subject_id"] = agfv["subject_id"].astype(str)

    meta = meta[to_bool(meta["ge_column_found"])]
    meta = meta[meta["diagnosis"].astype(str).isin(["AD", "CN"])].copy()

    keep_cols = [
        "subject_id",
        "ge_sample_column",
        "diagnosis",
        "diagnosis_binary",
    ]
    if "split" in meta.columns:
        keep_cols.append("split")

    meta_small = meta[keep_cols].drop_duplicates("subject_id")

    merged = agfv.merge(meta_small, on="subject_id", how="left", suffixes=("", "_meta"))

    if merged["ge_sample_column"].isna().any():
        missing_ids = merged.loc[merged["ge_sample_column"].isna(), "subject_id"].head().tolist()
        raise ValueError(f"Missing ge_sample_column for AGFV subjects. Examples: {missing_ids}")

    if "split" not in merged.columns:
        if "split_meta" in merged.columns:
            merged["split"] = merged["split_meta"]
        else:
            raise ValueError("Measured AGFV file or metadata must contain split.")

    sample_cols = merged["ge_sample_column"].astype(str).tolist()
    missing_sample_cols = [c for c in sample_cols if c not in expr.columns]
    if missing_sample_cols:
        raise ValueError(f"Missing expression sample columns. Examples: {missing_sample_cols[:5]}")

    expr_panel = expr[expr[probe_col].astype(str).isin(panel_probes)].copy()
    expr_panel = expr_panel.drop_duplicates(subset=[probe_col], keep="first")
    expr_panel = expr_panel.set_index(probe_col)

    available_probes = [p for p in panel_probes if p in expr_panel.index]
    expr_panel = expr_panel.loc[available_probes]

    x_df = expr_panel[sample_cols].T
    x_df.index = merged.index
    x_df = x_df.apply(pd.to_numeric, errors="coerce")

    if x_df.isna().sum().sum() > 0:
        raise ValueError("Missing expression values after panel extraction.")

    selected_used = selected[selected["Probe ID / feature ID"].isin(available_probes)].copy()
    selected_used = selected_used.drop_duplicates(subset=["Probe ID / feature ID"], keep="first")
    selected_used = selected_used.set_index("Probe ID / feature ID").loc[available_probes].reset_index()

    agfv_cols = get_agfv_cols(merged)
    if len(agfv_cols) != 128:
        raise ValueError(f"Expected 128 AGFV columns, found {len(agfv_cols)}")

    a_df = merged[agfv_cols].astype(float)

    return x_df, a_df, merged, selected_used, available_probes


def pearson_corr_matrix(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x - np.nanmean(x, axis=0, keepdims=True)
    y = y - np.nanmean(y, axis=0, keepdims=True)

    x_ss = np.sqrt(np.nansum(x ** 2, axis=0))
    y_ss = np.sqrt(np.nansum(y ** 2, axis=0))

    denom = np.outer(x_ss, y_ss)
    corr = x.T @ y
    corr = corr / np.where(denom == 0, np.nan, denom)

    return corr


def rank_transform(df: pd.DataFrame):
    return df.rank(axis=0, method="average").values


def split_gene_symbols(value):
    if pd.isna(value):
        return ["Unknown"]

    genes = []
    for g in str(value).split("||"):
        g = g.strip()
        if g and g.lower() not in ["nan", "none", ""]:
            genes.append(g)

    return genes if genes else ["Unknown"]


def load_agfv_feature_importance():
    ab = pd.read_csv(AGFV_ABLATION_FILE)

    required = ["Split group", "AGFV feature", "Feature ablation score"]
    missing = [c for c in required if c not in ab.columns]
    if missing:
        raise ValueError(f"Missing AGFV ablation columns: {missing}")

    heldout = ab[ab["Split group"] == "Heldout"].copy()
    heldout = heldout.sort_values("Feature ablation score", ascending=False)

    top = heldout.head(TOP_AGFV_FEATURES).copy()

    return top


def load_ge_probe_scores():
    if not GE_PROBE_ABLATION_FILE.exists():
        return pd.DataFrame()

    probe = pd.read_csv(GE_PROBE_ABLATION_FILE)

    if "Split group" in probe.columns:
        probe = probe[probe["Split group"] == "Heldout"].copy()

    if "Probe ID / feature ID" not in probe.columns:
        return pd.DataFrame()

    if "Probe ablation score" not in probe.columns:
        probe["Probe ablation score"] = (
            probe.get("Delta AUC", 0).fillna(0)
            + probe.get("Delta balanced accuracy", 0).fillna(0)
            + 0.10 * probe.get("Mean absolute probability change", 0).fillna(0)
        )

    keep = [
        "Probe ID / feature ID",
        "Probe ablation score",
        "Delta AUC",
        "Delta balanced accuracy",
        "Mean absolute probability change",
    ]
    keep = [c for c in keep if c in probe.columns]

    return probe[keep].drop_duplicates("Probe ID / feature ID")


def main():
    print("========== MAP AGFV FEATURES TO GENES ==========")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Measured AGFV: {MEASURED_AGFV_FILE}")
    print(f"AGFV feature ablation: {AGFV_ABLATION_FILE}")
    print("===============================================\n")

    required_files = [
        GE_FILE,
        META_FILE,
        SELECTED_PROBE_FILE,
        MEASURED_AGFV_FILE,
        AGFV_ABLATION_FILE,
    ]

    for f in required_files:
        if not f.exists():
            raise FileNotFoundError(f)

    x_df, a_df, merged, selected_used, available_probes = prepare_expression_for_measured_agfv()

    agfv_importance = load_agfv_feature_importance()
    top_agfv_features = agfv_importance["AGFV feature"].tolist()

    missing_agfv = [c for c in top_agfv_features if c not in a_df.columns]
    if missing_agfv:
        raise ValueError(f"Top AGFV features missing from measured AGFV file: {missing_agfv}")

    print("Measured subjects:", merged.shape[0])
    print("Expression probes:", x_df.shape[1])
    print("AGFV features:", a_df.shape[1])
    print("Top AGFV features used:", len(top_agfv_features))
    print()

    # Use all measured-GE subjects for molecular association mapping.
    x = x_df.values
    a = a_df[top_agfv_features].values

    pearson = pearson_corr_matrix(x, a)

    x_rank = rank_transform(x_df)
    a_rank = rank_transform(a_df[top_agfv_features])
    spearman = pearson_corr_matrix(x_rank, a_rank)

    rows = []

    feature_score_map = dict(
        zip(
            agfv_importance["AGFV feature"],
            agfv_importance["Feature ablation score"],
        )
    )

    for i, probe_id in enumerate(available_probes):
        probe_meta = selected_used.iloc[i].to_dict()

        for j, agfv_feature in enumerate(top_agfv_features):
            rows.append(
                {
                    "AGFV feature": agfv_feature,
                    "AGFV feature ablation score": feature_score_map.get(agfv_feature, np.nan),
                    "Probe ID / feature ID": probe_id,
                    "Gene symbol": probe_meta.get("Gene symbol", "Unknown"),
                    "Genomic locus": probe_meta.get("Genomic locus", probe_meta.get("Locus", np.nan)),
                    "Pearson r": float(pearson[i, j]),
                    "Spearman r": float(spearman[i, j]),
                    "Abs Pearson r": float(abs(pearson[i, j])),
                    "Abs Spearman r": float(abs(spearman[i, j])),
                    "Association direction": "positive" if pearson[i, j] >= 0 else "negative",
                }
            )

    out = pd.DataFrame(rows)

    probe_scores = load_ge_probe_scores()
    if not probe_scores.empty:
        out = out.merge(probe_scores, on="Probe ID / feature ID", how="left")
    else:
        out["Probe ablation score"] = np.nan

    # Normalize weights for combined score.
    feature_weight = out["AGFV feature ablation score"].clip(lower=0).fillna(0)
    probe_weight = out["Probe ablation score"].clip(lower=0).fillna(0)

    if feature_weight.max() > 0:
        feature_weight = feature_weight / feature_weight.max()
    else:
        feature_weight = pd.Series(0.0, index=out.index)

    if probe_weight.max() > 0:
        probe_weight = probe_weight / probe_weight.max()
    else:
        probe_weight = pd.Series(0.0, index=out.index)

    out["Combined AGFV-gene score"] = out["Abs Pearson r"] * (1 + feature_weight) * (1 + probe_weight)

    top_rows = []
    for agfv_feature, sub in out.groupby("AGFV feature"):
        tmp = sub.sort_values(
            ["Combined AGFV-gene score", "Abs Pearson r", "Abs Spearman r"],
            ascending=False,
        ).head(TOP_PROBES_PER_AGFV)
        top_rows.append(tmp)

    top = pd.concat(top_rows, ignore_index=True)
    top = top.sort_values(
        ["Combined AGFV-gene score", "Abs Pearson r"],
        ascending=False,
    )

    # Gene-level aggregation.
    gene_rows = []
    for _, r in out.iterrows():
        for gene in split_gene_symbols(r.get("Gene symbol", "Unknown")):
            tmp = r.to_dict()
            tmp["Gene"] = gene
            gene_rows.append(tmp)

    gene_long = pd.DataFrame(gene_rows)

    gene_level = (
        gene_long.groupby(["AGFV feature", "Gene"])
        .agg(
            Probe_count=("Probe ID / feature ID", "nunique"),
            Max_abs_Pearson_r=("Abs Pearson r", "max"),
            Mean_abs_Pearson_r=("Abs Pearson r", "mean"),
            Max_abs_Spearman_r=("Abs Spearman r", "max"),
            Mean_abs_Spearman_r=("Abs Spearman r", "mean"),
            Max_combined_score=("Combined AGFV-gene score", "max"),
            Mean_combined_score=("Combined AGFV-gene score", "mean"),
            AGFV_feature_ablation_score=("AGFV feature ablation score", "first"),
        )
        .reset_index()
        .sort_values(
            ["Max_combined_score", "Max_abs_Pearson_r"],
            ascending=False,
        )
    )

    out.to_csv(ALL_OUT, index=False)
    top.to_csv(TOP_OUT, index=False)
    gene_level.to_csv(GENE_OUT, index=False)

    with pd.ExcelWriter(TABLE_OUT) as writer:
        top.head(1000).to_excel(writer, sheet_name="Top_probe_AGFV_links", index=False)
        gene_level.head(1000).to_excel(writer, sheet_name="Top_gene_AGFV_links", index=False)
        agfv_importance.to_excel(writer, sheet_name="Top_AGFV_features", index=False)

    summary = []
    summary.append("========== MAP AGFV FEATURES TO GENES ==========")
    summary.append(f"Measured GE subjects: {merged.shape[0]}")
    summary.append(f"Selected probes: {x_df.shape[1]}")
    summary.append(f"Top AGFV features analyzed: {len(top_agfv_features)}")
    summary.append("")
    summary.append("Top 30 AGFV feature-probe-gene links:")
    summary.append(
        top[
            [
                "AGFV feature",
                "Gene symbol",
                "Probe ID / feature ID",
                "Pearson r",
                "Spearman r",
                "AGFV feature ablation score",
                "Probe ablation score",
                "Combined AGFV-gene score",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )
    summary.append("")
    summary.append("Top 30 AGFV feature-gene links:")
    summary.append(
        gene_level[
            [
                "AGFV feature",
                "Gene",
                "Probe_count",
                "Max_abs_Pearson_r",
                "Max_abs_Spearman_r",
                "AGFV_feature_ablation_score",
                "Max_combined_score",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )
    summary.append("")
    summary.append("Saved:")
    summary.append(str(ALL_OUT))
    summary.append(str(TOP_OUT))
    summary.append(str(GENE_OUT))
    summary.append(str(TABLE_OUT))
    summary.append("STATUS: DONE")

    SUMMARY_OUT.write_text("\n".join(summary) + "\n")

    print("\n".join(summary))
    print("================================================")


if __name__ == "__main__":
    main()
