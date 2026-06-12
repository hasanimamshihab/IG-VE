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

DK_REGION_FILE = PROJECT_ROOT / "outputs/ablation/mri_dk_region_ablation_region_level.csv"
DK_REGION_FEATURE_FILE = PROJECT_ROOT / "outputs/ablation/mri_dk_region_ablation_region_agfv_feature_level.csv"

OUT_DIR = PROJECT_ROOT / "outputs/ablation"
TABLE_DIR = PROJECT_ROOT / "outputs/tables/supplementary"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

AGFV_GENE_ALL_OUT = OUT_DIR / "agfv_gene_association_all128_probe_level.csv"
AGFV_GENE_LEVEL_OUT = OUT_DIR / "agfv_gene_association_all128_gene_level.csv"

BRIDGE_ALL_OUT = OUT_DIR / "dk_region_agfv_gene_bridge_all.csv"
BRIDGE_TOP_OUT = OUT_DIR / "dk_region_agfv_gene_bridge_top.csv"
REGION_SUMMARY_OUT = OUT_DIR / "dk_region_gene_bridge_region_summary.csv"
SUMMARY_OUT = OUT_DIR / "dk_region_agfv_gene_bridge_summary.txt"
TABLE_OUT = TABLE_DIR / "Table_S22_DK_region_AGFV_gene_bridge.xlsx"

TOP_GENES_PER_AGFV = 50
TOP_BRIDGE_ROWS = 5000


def to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def get_agfv_cols(df: pd.DataFrame):
    pat = re.compile(r"^AGFV\d+$", re.IGNORECASE)
    cols = [c for c in df.columns if pat.match(c)]
    return sorted(cols, key=lambda x: int(re.findall(r"\d+", x)[0]))


def normalize_nonnegative(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce").fillna(0)
    s = s.clip(lower=0)
    maxv = s.max()
    if maxv <= 0:
        return pd.Series(0.0, index=s.index)
    return s / maxv


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


def prepare_expression_and_agfv():
    selected = pd.read_csv(SELECTED_PROBE_FILE)
    selected["Probe ID / feature ID"] = selected["Probe ID / feature ID"].astype(str)

    panel_probes = selected["Probe ID / feature ID"].tolist()

    expr, probe_col, locus_col, symbol_col = load_expression_matrix()

    meta = pd.read_csv(META_FILE)
    agfv = pd.read_csv(MEASURED_AGFV_FILE)

    meta["subject_id"] = meta["subject_id"].astype(str)
    agfv["subject_id"] = agfv["subject_id"].astype(str)

    meta = meta[to_bool(meta["ge_column_found"])]
    meta = meta[meta["diagnosis"].astype(str).isin(["AD", "CN"])].copy()

    meta_small = meta[
        ["subject_id", "ge_sample_column", "diagnosis", "diagnosis_binary"]
    ].drop_duplicates("subject_id")

    merged = agfv.merge(meta_small, on="subject_id", how="left", suffixes=("", "_meta"))

    if merged["ge_sample_column"].isna().any():
        bad = merged.loc[merged["ge_sample_column"].isna(), "subject_id"].head().tolist()
        raise ValueError(f"Missing ge_sample_column for measured AGFV subjects: {bad}")

    sample_cols = merged["ge_sample_column"].astype(str).tolist()
    missing_cols = [c for c in sample_cols if c not in expr.columns]

    if missing_cols:
        raise ValueError(f"Missing expression sample columns: {missing_cols[:10]}")

    expr_panel = expr[expr[probe_col].astype(str).isin(panel_probes)].copy()
    expr_panel = expr_panel.drop_duplicates(subset=[probe_col], keep="first")
    expr_panel = expr_panel.set_index(probe_col)

    available_probes = [p for p in panel_probes if p in expr_panel.index]
    expr_panel = expr_panel.loc[available_probes]

    x_df = expr_panel[sample_cols].T
    x_df.index = merged.index
    x_df = x_df.apply(pd.to_numeric, errors="coerce")

    if x_df.isna().sum().sum() > 0:
        raise ValueError("Missing expression values after selected-probe extraction.")

    selected_used = selected[
        selected["Probe ID / feature ID"].isin(available_probes)
    ].copy()
    selected_used = selected_used.drop_duplicates("Probe ID / feature ID", keep="first")
    selected_used = selected_used.set_index("Probe ID / feature ID").loc[available_probes].reset_index()

    agfv_cols = get_agfv_cols(merged)

    if len(agfv_cols) != 128:
        raise ValueError(f"Expected 128 AGFV columns, found {len(agfv_cols)}")

    a_df = merged[agfv_cols].astype(float)

    return x_df, a_df, selected_used, available_probes, agfv_cols


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


def load_agfv_feature_scores():
    ab = pd.read_csv(AGFV_ABLATION_FILE)
    ab = ab[ab["Split group"] == "Heldout"].copy()

    if "Feature ablation score" not in ab.columns:
        ab["Feature ablation score"] = (
            ab["Delta AUC"].fillna(0)
            + ab["Delta balanced accuracy"].fillna(0)
            + 0.10 * ab["Mean absolute probability change"].fillna(0)
        )

    keep = [
        "AGFV feature",
        "Feature ablation score",
        "Delta AUC",
        "Delta balanced accuracy",
        "Mean absolute probability change",
    ]

    keep = [c for c in keep if c in ab.columns]

    return ab[keep].drop_duplicates("AGFV feature")


def load_ge_probe_scores():
    if not GE_PROBE_ABLATION_FILE.exists():
        return pd.DataFrame()

    probe = pd.read_csv(GE_PROBE_ABLATION_FILE)

    if "Split group" in probe.columns:
        probe = probe[probe["Split group"] == "Heldout"].copy()

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


def build_all128_agfv_gene_association():
    x_df, a_df, selected_used, available_probes, agfv_cols = prepare_expression_and_agfv()

    print("Measured GE subjects:", x_df.shape[0])
    print("Selected probes:", x_df.shape[1])
    print("AGFV features:", len(agfv_cols))

    x = x_df.values
    a = a_df[agfv_cols].values

    pearson = pearson_corr_matrix(x, a)
    spearman = pearson_corr_matrix(rank_transform(x_df), rank_transform(a_df[agfv_cols]))

    rows = []

    for i, probe_id in enumerate(available_probes):
        probe_meta = selected_used.iloc[i].to_dict()

        for j, agfv_feature in enumerate(agfv_cols):
            rows.append(
                {
                    "AGFV feature": agfv_feature,
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

    probe_level = pd.DataFrame(rows)

    agfv_scores = load_agfv_feature_scores()
    probe_scores = load_ge_probe_scores()

    probe_level = probe_level.merge(agfv_scores, on="AGFV feature", how="left")

    if not probe_scores.empty:
        probe_level = probe_level.merge(probe_scores, on="Probe ID / feature ID", how="left")
    else:
        probe_level["Probe ablation score"] = np.nan

    agfv_norm = normalize_nonnegative(probe_level["Feature ablation score"])
    probe_norm = normalize_nonnegative(probe_level["Probe ablation score"])

    probe_level["Probe-AGFV molecular score"] = (
        probe_level["Abs Pearson r"].fillna(0)
        * (1.0 + agfv_norm)
        * (1.0 + probe_norm)
    )

    gene_rows = []

    for _, r in probe_level.iterrows():
        for gene in split_gene_symbols(r.get("Gene symbol", "Unknown")):
            tmp = r.to_dict()
            tmp["Gene"] = gene
            gene_rows.append(tmp)

    gene_long = pd.DataFrame(gene_rows)

    gene_long_sorted = gene_long.sort_values(
        ["AGFV feature", "Gene", "Probe-AGFV molecular score"],
        ascending=[True, True, False],
    )

    representative = gene_long_sorted.drop_duplicates(["AGFV feature", "Gene"])[
        [
            "AGFV feature",
            "Gene",
            "Probe ID / feature ID",
            "Gene symbol",
            "Pearson r",
            "Spearman r",
            "Association direction",
            "Probe ablation score",
        ]
    ].rename(
        columns={
            "Probe ID / feature ID": "Top probe ID / feature ID",
            "Gene symbol": "Top probe gene symbol",
            "Pearson r": "Top probe Pearson r",
            "Spearman r": "Top probe Spearman r",
            "Association direction": "Top probe association direction",
            "Probe ablation score": "Top probe ablation score",
        }
    )

    gene_level = (
        gene_long.groupby(["AGFV feature", "Gene"])
        .agg(
            Probe_count=("Probe ID / feature ID", "nunique"),
            Max_abs_Pearson_r=("Abs Pearson r", "max"),
            Mean_abs_Pearson_r=("Abs Pearson r", "mean"),
            Max_abs_Spearman_r=("Abs Spearman r", "max"),
            Mean_abs_Spearman_r=("Abs Spearman r", "mean"),
            Max_probe_AGFV_molecular_score=("Probe-AGFV molecular score", "max"),
            Mean_probe_AGFV_molecular_score=("Probe-AGFV molecular score", "mean"),
            AGFV_feature_ablation_score=("Feature ablation score", "first"),
        )
        .reset_index()
        .merge(representative, on=["AGFV feature", "Gene"], how="left")
        .sort_values(
            ["Max_probe_AGFV_molecular_score", "Max_abs_Pearson_r"],
            ascending=False,
        )
    )

    probe_level.to_csv(AGFV_GENE_ALL_OUT, index=False)
    gene_level.to_csv(AGFV_GENE_LEVEL_OUT, index=False)

    top_gene_per_agfv = (
        gene_level.sort_values(
            ["AGFV feature", "Max_probe_AGFV_molecular_score", "Max_abs_Pearson_r"],
            ascending=[True, False, False],
        )
        .groupby("AGFV feature", group_keys=False)
        .head(TOP_GENES_PER_AGFV)
        .reset_index(drop=True)
    )

    return probe_level, gene_level, top_gene_per_agfv


def build_bridge(top_gene_per_agfv):
    region = pd.read_csv(DK_REGION_FILE)
    region_feature = pd.read_csv(DK_REGION_FEATURE_FILE)

    region_heldout = region[region["Split group"] == "Heldout"].copy()

    if "Region ablation score" not in region_heldout.columns:
        region_heldout["Region ablation score"] = (
            region_heldout["Delta AUC"].fillna(0)
            + region_heldout["Delta balanced accuracy"].fillna(0)
            + region_heldout["Mean absolute probability change"].fillna(0)
            + 0.10 * region_heldout["Prediction changed, fraction"].fillna(0)
            + region_heldout["Mean AGFV absolute change"].fillna(0)
        )

    region_keep = [
        "Region label",
        "Region name",
        "hemisphere",
        "acronyms",
        "subcortex",
        "RSN",
        "BraakStage",
        "Delta AUC",
        "Delta balanced accuracy",
        "Mean absolute probability change",
        "Prediction changed, fraction",
        "Mean AGFV absolute change",
        "Region ablation score",
        "MNI.x",
        "MNI.y",
        "MNI.z",
    ]

    region_keep = [c for c in region_keep if c in region_heldout.columns]

    region_heldout = region_heldout[region_keep].drop_duplicates("Region label")

    rf = region_feature.merge(
        region_heldout,
        on=["Region label", "Region name", "hemisphere", "acronyms", "subcortex", "RSN", "BraakStage"],
        how="left",
        suffixes=("", "_region"),
    )

    bridge = rf.merge(top_gene_per_agfv, on="AGFV feature", how="inner")

    region_norm = normalize_nonnegative(bridge["Region ablation score"])
    agfv_change_norm = normalize_nonnegative(bridge["Mean absolute AGFV feature change"])
    classifier_norm = normalize_nonnegative(bridge["AGFV_feature_ablation_score"])

    bridge["Region-AGFV-gene score"] = (
        bridge["Max_abs_Pearson_r"].fillna(0)
        * (1.0 + region_norm)
        * (1.0 + agfv_change_norm)
        * (1.0 + classifier_norm)
    )

    bridge = bridge.sort_values(
        [
            "Region-AGFV-gene score",
            "Region ablation score",
            "Mean absolute AGFV feature change",
            "Max_abs_Pearson_r",
        ],
        ascending=False,
    )

    top_bridge = bridge.head(TOP_BRIDGE_ROWS).copy()

    region_summary_rows = []

    for region_label, sub in bridge.groupby("Region label"):
        sub_sorted = sub.sort_values("Region-AGFV-gene score", ascending=False)

        region_name = sub_sorted["Region name"].iloc[0]
        hemisphere = sub_sorted["hemisphere"].iloc[0] if "hemisphere" in sub_sorted.columns else np.nan
        rsn = sub_sorted["RSN"].iloc[0] if "RSN" in sub_sorted.columns else np.nan

        top_genes = []
        for g in sub_sorted["Gene"].astype(str).tolist():
            if g not in top_genes:
                top_genes.append(g)
            if len(top_genes) >= 15:
                break

        top_agfv = []
        for f in sub_sorted["AGFV feature"].astype(str).tolist():
            if f not in top_agfv:
                top_agfv.append(f)
            if len(top_agfv) >= 15:
                break

        region_summary_rows.append(
            {
                "Region label": region_label,
                "Region name": region_name,
                "hemisphere": hemisphere,
                "RSN": rsn,
                "Region ablation score": sub_sorted["Region ablation score"].iloc[0],
                "Delta AUC": sub_sorted["Delta AUC"].iloc[0],
                "Delta balanced accuracy": sub_sorted["Delta balanced accuracy"].iloc[0],
                "Mean absolute probability change": sub_sorted["Mean absolute probability change"].iloc[0],
                "Mean AGFV absolute change": sub_sorted["Mean AGFV absolute change"].iloc[0],
                "Max Region-AGFV-gene score": sub_sorted["Region-AGFV-gene score"].max(),
                "Mean top-50 Region-AGFV-gene score": sub_sorted["Region-AGFV-gene score"].head(50).mean(),
                "Top linked AGFV features": "; ".join(top_agfv),
                "Top linked genes": "; ".join(top_genes),
            }
        )

    region_summary = pd.DataFrame(region_summary_rows).sort_values(
        ["Region ablation score", "Max Region-AGFV-gene score"],
        ascending=False,
    )

    return bridge, top_bridge, region_summary, region_heldout, region_feature


def main():
    print("========== BUILD DK REGION → AGFV → GENE BRIDGE ==========")
    print(f"Project root: {PROJECT_ROOT}")
    print("==========================================================\n")

    required = [
        GE_FILE,
        META_FILE,
        SELECTED_PROBE_FILE,
        MEASURED_AGFV_FILE,
        AGFV_ABLATION_FILE,
        GE_PROBE_ABLATION_FILE,
        DK_REGION_FILE,
        DK_REGION_FEATURE_FILE,
    ]

    for f in required:
        if not f.exists():
            raise FileNotFoundError(f)

    probe_level, gene_level, top_gene_per_agfv = build_all128_agfv_gene_association()
    bridge, top_bridge, region_summary, region_heldout, region_feature = build_bridge(top_gene_per_agfv)

    bridge.to_csv(BRIDGE_ALL_OUT, index=False)
    top_bridge.to_csv(BRIDGE_TOP_OUT, index=False)
    region_summary.to_csv(REGION_SUMMARY_OUT, index=False)

    with pd.ExcelWriter(TABLE_OUT) as writer:
        top_bridge.to_excel(writer, sheet_name="Top_region_AGFV_gene_links", index=False)
        region_summary.to_excel(writer, sheet_name="Region_gene_summary", index=False)
        gene_level.head(10000).to_excel(writer, sheet_name="AGFV_gene_links_all128", index=False)
        region_heldout.to_excel(writer, sheet_name="Heldout_region_ablation", index=False)
        region_feature.head(5000).to_excel(writer, sheet_name="Region_AGFV_feature_change", index=False)

    summary = []
    summary.append("========== DK REGION → AGFV → GENE BRIDGE ==========")
    summary.append(f"AGFV-gene probe-level rows: {probe_level.shape[0]}")
    summary.append(f"AGFV-gene gene-level rows: {gene_level.shape[0]}")
    summary.append(f"Bridge rows: {bridge.shape[0]}")
    summary.append(f"Region summary rows: {region_summary.shape[0]}")
    summary.append("")
    summary.append("Top 30 DK region-AGFV-gene links:")
    summary.append(
        top_bridge[
            [
                "Region label",
                "Region name",
                "hemisphere",
                "RSN",
                "AGFV feature",
                "Gene",
                "Top probe ID / feature ID",
                "Max_abs_Pearson_r",
                "Mean absolute AGFV feature change",
                "Region ablation score",
                "AGFV_feature_ablation_score",
                "Region-AGFV-gene score",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )
    summary.append("")
    summary.append("Top 20 region-level summaries:")
    summary.append(
        region_summary[
            [
                "Region label",
                "Region name",
                "hemisphere",
                "RSN",
                "Region ablation score",
                "Delta AUC",
                "Mean absolute probability change",
                "Mean AGFV absolute change",
                "Top linked AGFV features",
                "Top linked genes",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )
    summary.append("")
    summary.append("Saved:")
    summary.append(str(AGFV_GENE_ALL_OUT))
    summary.append(str(AGFV_GENE_LEVEL_OUT))
    summary.append(str(BRIDGE_ALL_OUT))
    summary.append(str(BRIDGE_TOP_OUT))
    summary.append(str(REGION_SUMMARY_OUT))
    summary.append(str(TABLE_OUT))
    summary.append("STATUS: DONE")

    SUMMARY_OUT.write_text("\n".join(summary) + "\n")

    print("\n".join(summary))
    print("====================================================")


if __name__ == "__main__":
    main()
