"""
limma-based differential expression analysis for AD versus CN.

This script prepares ADNI gene-expression data for limma, runs the R limma
analysis, and creates a journal-style supplementary Excel table.

Model:
    expression ~ diagnosis + age + sex + APOE4

Diagnosis coefficient:
    AD versus CN

Outputs:
- outputs/DEG/limma_input_expression_matrix.csv
- outputs/DEG/limma_input_feature_info.csv
- outputs/DEG/limma_input_sample_metadata.csv
- outputs/DEG/deg_results_all_genes.csv
- outputs/DEG/nominal_significant_gene_panel_p_lt_0_05.csv
- outputs/DEG/deg_sample_metadata.csv
- outputs/DEG/deg_design_matrix.csv
- outputs/tables/supplementary/Table_S3_DEG_results.xlsx
"""

from pathlib import Path
import subprocess
import shutil
import pandas as pd

from src.xigcvae.utils.excel_style import style_supplementary_table


ROOT = Path("/N/project/SingleCell_Image/Ronak/Project_folder")
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs" / "DEG"
TABLE_SUPP_DIR = ROOT / "outputs" / "tables" / "supplementary"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_SUPP_DIR.mkdir(parents=True, exist_ok=True)

GE_FILE = DATA_DIR / "gene_expression_data.csv"
META_FILE = DATA_DIR / "Paired_data_metadata.csv"
R_SCRIPT = ROOT / "src" / "xigcvae" / "deg" / "run_limma_deg.R"

LIMMA_EXPR_INPUT = OUT_DIR / "limma_input_expression_matrix.csv"
LIMMA_FEATURE_INPUT = OUT_DIR / "limma_input_feature_info.csv"
LIMMA_META_INPUT = OUT_DIR / "limma_input_sample_metadata.csv"

DEG_ALL_CSV = OUT_DIR / "deg_results_all_genes.csv"
NOMINAL_PANEL_CSV = OUT_DIR / "nominal_significant_gene_panel_p_lt_0_05.csv"
SAMPLE_META_CSV = OUT_DIR / "deg_sample_metadata.csv"
DESIGN_MATRIX_CSV = OUT_DIR / "deg_design_matrix.csv"
TABLE_S3_XLSX = TABLE_SUPP_DIR / "Table_S3_DEG_results.xlsx"


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def find_row_index(df: pd.DataFrame, label: str) -> int:
    matches = df.index[df.iloc[:, 0].astype(str).str.strip() == label].tolist()
    if not matches:
        raise ValueError(f"Could not find row labeled: {label}")
    return matches[0]


def prepare_limma_inputs() -> pd.DataFrame:
    print("Reading paired metadata...")
    meta = pd.read_csv(META_FILE)

    if "ge_column_found" in meta.columns:
        meta["ge_column_found"] = bool_series(meta["ge_column_found"])
    else:
        meta["ge_column_found"] = True

    meta = meta[
        (meta["ge_column_found"]) &
        (meta["diagnosis"].isin(["AD", "CN"]))
    ].copy()

    meta["diagnosis_numeric"] = (meta["diagnosis"] == "AD").astype(int)

    required_covariates = [
        "ge_sample_column",
        "diagnosis",
        "diagnosis_numeric",
        "age",
        "sex",
        "apoe4_allele_count",
    ]

    missing_covariates = [c for c in required_covariates if c not in meta.columns]
    if missing_covariates:
        raise ValueError(f"Missing required metadata columns: {missing_covariates}")

    print("Reading gene-expression file...")
    ge = pd.read_csv(GE_FILE, low_memory=False)

    probeset_row = find_row_index(ge, "ProbeSet")

    sample_cols = meta["ge_sample_column"].astype(str).tolist()
    missing_cols = [c for c in sample_cols if c not in ge.columns]

    if missing_cols:
        raise ValueError(
            "Some GE sample columns from metadata are missing in gene_expression_data.csv: "
            f"{missing_cols[:10]}"
        )

    expr = ge.iloc[probeset_row + 1:].copy()

    feature_info = expr.iloc[:, :3].copy()
    feature_info.columns = ["probe_id", "locus_link", "gene_symbol"]
    feature_info["probe_id"] = feature_info["probe_id"].astype(str)

    expr_matrix = expr[sample_cols].apply(pd.to_numeric, errors="coerce")
    expr_matrix.insert(0, "probe_id", feature_info["probe_id"].values)

    # Remove probes with any missing expression across selected samples.
    before_features = len(expr_matrix)
    expr_matrix = expr_matrix.dropna(axis=0)
    after_features = len(expr_matrix)

    retained_probe_ids = set(expr_matrix["probe_id"].astype(str))
    feature_info = feature_info[
        feature_info["probe_id"].astype(str).isin(retained_probe_ids)
    ].copy()

    # Keep feature_info in the same order as expression matrix.
    feature_info = feature_info.set_index("probe_id").loc[
        expr_matrix["probe_id"].astype(str)
    ].reset_index()

    sample_meta = meta[required_covariates].copy()
    sample_meta["ge_sample_column"] = sample_meta["ge_sample_column"].astype(str)

    sample_meta.to_csv(LIMMA_META_INPUT, index=False)
    expr_matrix.to_csv(LIMMA_EXPR_INPUT, index=False)
    feature_info.to_csv(LIMMA_FEATURE_INPUT, index=False)

    print("Prepared limma inputs.")
    print(f"Measured GE subjects before limma covariate filtering: {len(sample_meta)}")
    print(sample_meta["diagnosis"].value_counts().to_string())
    print(f"Expression features before missing-expression filtering: {before_features}")
    print(f"Expression features after missing-expression filtering: {after_features}")

    return sample_meta


def run_limma() -> None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError(
            "Rscript was not found in the current environment. "
            "Load an R module first, then rerun this script."
        )

    cmd = [rscript, str(R_SCRIPT), str(ROOT)]

    print()
    print("Running limma through Rscript...")
    print(" ".join(cmd))

    completed = subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True,
    )

    print(completed.stdout)

    if completed.stderr.strip():
        print("R stderr:")
        print(completed.stderr)


def make_table_s3() -> pd.DataFrame:
    if not DEG_ALL_CSV.exists():
        raise FileNotFoundError(f"limma output not found: {DEG_ALL_CSV}")

    results = pd.read_csv(DEG_ALL_CSV)

    with pd.ExcelWriter(TABLE_S3_XLSX, engine="openpyxl") as writer:
        results.to_excel(writer, index=False, sheet_name="Table S3")

    style_supplementary_table(
        TABLE_S3_XLSX,
        "Table S3",
        freeze_panes="A2",
        max_width=34,
    )

    return results


def main() -> None:
    stale_top_464 = OUT_DIR / "top_464_gene_panel.csv"
    if stale_top_464.exists():
        stale_top_464.unlink()

    print("========== limma DEG ANALYSIS ==========")
    print("Model: expression ~ diagnosis + age + sex + APOE4")
    print("Diagnosis coefficient: AD versus CN")
    print()

    prepare_limma_inputs()
    run_limma()
    results = make_table_s3()

    sample_meta = pd.read_csv(SAMPLE_META_CSV)

    print("========== DEG SUMMARY ==========")
    print(f"Samples used in limma: {len(sample_meta)}")
    print(sample_meta["diagnosis"].value_counts().to_string())
    print(f"Expression features tested: {len(results)}")
    print(f"Nominal P < 0.05: {(results['Raw P value'] < 0.05).sum()}")
    print(f"FDR < 0.05: {(results['FDR-adjusted P value'] < 0.05).sum()}")
    print()
    print("Top 10 limma DEG results:")
    print(
        results.head(10)[
            [
                "Gene symbol",
                "Probe ID / feature ID",
                "limma logFC AD vs CN",
                "Raw P value",
                "FDR-adjusted P value",
                "Direction in AD",
            ]
        ].to_string(index=False)
    )
    print()
    print("Saved:")
    print(DEG_ALL_CSV)
    print(NOMINAL_PANEL_CSV)
    print(SAMPLE_META_CSV)
    print(DESIGN_MATRIX_CSV)
    print(TABLE_S3_XLSX)
    print("=================================")


if __name__ == "__main__":
    main()
