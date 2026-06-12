from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path("/N/project/SingleCell_Image/Ronak/Project_folder")

FILES = {
    "GE probe ablation": PROJECT_ROOT / "outputs/ablation/ge_branch_probe_ablation.csv",
    "GE gene ablation": PROJECT_ROOT / "outputs/ablation/ge_branch_gene_ablation.csv",
    "AGFV NN feature ablation": PROJECT_ROOT / "outputs/ablation/agfv_nn_feature_ablation.csv",
    "AGFV gene association top": PROJECT_ROOT / "outputs/ablation/agfv_gene_association_top.csv",
    "AGFV gene association gene level": PROJECT_ROOT / "outputs/ablation/agfv_gene_association_gene_level.csv",
    "DK subject-region ablation": PROJECT_ROOT / "outputs/ablation/mri_dk_region_ablation_subject_region.csv",
    "DK region-level ablation": PROJECT_ROOT / "outputs/ablation/mri_dk_region_ablation_region_level.csv",
    "DK region-AGFV feature ablation": PROJECT_ROOT / "outputs/ablation/mri_dk_region_ablation_region_agfv_feature_level.csv",
    "AGFV-gene all128 probe level": PROJECT_ROOT / "outputs/ablation/agfv_gene_association_all128_probe_level.csv",
    "AGFV-gene all128 gene level": PROJECT_ROOT / "outputs/ablation/agfv_gene_association_all128_gene_level.csv",
    "DK region-AGFV-gene bridge all": PROJECT_ROOT / "outputs/ablation/dk_region_agfv_gene_bridge_all.csv",
    "DK region-AGFV-gene bridge top": PROJECT_ROOT / "outputs/ablation/dk_region_agfv_gene_bridge_top.csv",
    "DK region-gene summary": PROJECT_ROOT / "outputs/ablation/dk_region_gene_bridge_region_summary.csv",
}

TABLES = {
    "Table S18": PROJECT_ROOT / "outputs/tables/supplementary/Table_S18_GE_branch_probe_gene_ablation.xlsx",
    "Table S19": PROJECT_ROOT / "outputs/tables/supplementary/Table_S19_AGFV_NN_feature_ablation.xlsx",
    "Table S20": PROJECT_ROOT / "outputs/tables/supplementary/Table_S20_AGFV_gene_association.xlsx",
    "Table S21": PROJECT_ROOT / "outputs/tables/supplementary/Table_S21_DK_region_ablation_to_AGFV.xlsx",
    "Table S22": PROJECT_ROOT / "outputs/tables/supplementary/Table_S22_DK_region_AGFV_gene_bridge.xlsx",
}

SUMMARY_OUT = PROJECT_ROOT / "outputs/ablation/check_ablation_outputs_summary.txt"


def check(cond, msg, passes, errors):
    if cond:
        passes.append(msg)
    else:
        errors.append(msg)


def main():
    print("========== CHECK ABLATION OUTPUTS ==========")
    print(f"Project root: {PROJECT_ROOT}")
    print()

    passes = []
    errors = []
    rows = []

    for name, path in FILES.items():
        exists = path.exists()
        check(exists, f"{name} exists: {path}", passes, errors)

        if exists:
            df = pd.read_csv(path)
            rows.append(
                {
                    "Output": name,
                    "Path": str(path),
                    "Rows": int(df.shape[0]),
                    "Columns": int(df.shape[1]),
                }
            )
            check(df.shape[0] > 0, f"{name} is non-empty", passes, errors)

    for name, path in TABLES.items():
        exists = path.exists()
        check(exists, f"{name} exists: {path}", passes, errors)

    # Specific expected dimensions/checks
    if FILES["AGFV NN feature ablation"].exists():
        agfv = pd.read_csv(FILES["AGFV NN feature ablation"])
        check(agfv["AGFV feature"].nunique() == 128, "AGFV ablation has 128 unique AGFV features", passes, errors)
        check({"Train", "Validation", "Test", "Heldout", "All"}.issubset(set(agfv["Split group"])), "AGFV ablation has expected split groups", passes, errors)

    if FILES["DK region-level ablation"].exists():
        dk = pd.read_csv(FILES["DK region-level ablation"])
        heldout = dk[dk["Split group"] == "Heldout"]
        check(heldout["Region label"].nunique() >= 100, "DK heldout region ablation has at least 100 regions", passes, errors)
        check("Region ablation score" in dk.columns, "DK region ablation score column exists", passes, errors)

    if FILES["DK region-gene summary"].exists():
        rs = pd.read_csv(FILES["DK region-gene summary"])
        check(rs["Region label"].nunique() >= 100, "DK region-gene summary has at least 100 regions", passes, errors)
        check("Top linked genes" in rs.columns, "Region summary has Top linked genes column", passes, errors)

    if FILES["DK region-AGFV-gene bridge top"].exists():
        bridge = pd.read_csv(FILES["DK region-AGFV-gene bridge top"])
        required_cols = [
            "Region label",
            "Region name",
            "AGFV feature",
            "Gene",
            "Region-AGFV-gene score",
        ]
        for col in required_cols:
            check(col in bridge.columns, f"Bridge top has column: {col}", passes, errors)

    summary = []
    summary.append("========== CHECK ABLATION OUTPUTS ==========")

    if rows:
        shape_df = pd.DataFrame(rows)
        summary.append("")
        summary.append("Output shapes:")
        summary.append(shape_df.to_string(index=False))

    summary.append("")
    summary.append(f"Passed checks: {len(passes)}")
    summary.append(f"Failed checks: {len(errors)}")

    if errors:
        summary.append("")
        summary.append("FAILED CHECKS:")
        for e in errors:
            summary.append(f"- {e}")
        status = "FAIL"
    else:
        summary.append("")
        summary.append("STATUS: PASS")
        status = "PASS"

    SUMMARY_OUT.write_text("\n".join(summary) + "\n")

    print("\n".join(summary))
    print("============================================")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
