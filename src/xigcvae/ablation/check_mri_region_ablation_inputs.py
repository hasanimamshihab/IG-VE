from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path("/N/project/SingleCell_Image/Ronak/Project_folder")

CANDIDATE_FILES = [
    PROJECT_ROOT / "outputs/mri_branch/mri_branch_input_manifest_qc.csv",
    PROJECT_ROOT / "outputs/mri_branch/mfv_features_all_1495_mri_subjects.csv",
    PROJECT_ROOT / "outputs/aligned_gfv/aligned_gfv_combined_1508_subjects.csv",
    PROJECT_ROOT / "outputs/final_agfv_nn_classifier/final_agfv_nn_classifier_metrics.csv",
    PROJECT_ROOT / "outputs/ablation/agfv_gene_association_gene_level.csv",
    PROJECT_ROOT / "outputs/ablation/agfv_nn_feature_ablation.csv",
]

OUT_DIR = PROJECT_ROOT / "outputs/ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_OUT = OUT_DIR / "check_mri_region_ablation_inputs_summary.txt"


def find_columns(df: pd.DataFrame, patterns: list[str]):
    cols = []
    lower_cols = {c.lower(): c for c in df.columns}

    for c in df.columns:
        cl = c.lower()
        if any(p.lower() in cl for p in patterns):
            cols.append(c)

    return cols


def main():
    print("========== CHECK MRI REGION ABLATION INPUTS ==========")
    print(f"Project root: {PROJECT_ROOT}")
    print()

    summary = []
    summary.append("========== CHECK MRI REGION ABLATION INPUTS ==========")
    summary.append(f"Project root: {PROJECT_ROOT}")
    summary.append("")

    for f in CANDIDATE_FILES:
        exists = f.exists()
        print(f"{'FOUND' if exists else 'MISSING'}: {f}")
        summary.append(f"{'FOUND' if exists else 'MISSING'}: {f}")

        if exists and f.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(f, nrows=5)
                summary.append(f"  Columns ({len(df.columns)}): {list(df.columns)}")
                print(f"  Columns ({len(df.columns)}): {list(df.columns)}")

                path_cols = find_columns(
                    df,
                    [
                        "path",
                        "nii",
                        "image",
                        "mri",
                        "t1",
                        "file",
                    ],
                )
                if path_cols:
                    summary.append(f"  Possible image/path columns: {path_cols}")
                    print(f"  Possible image/path columns: {path_cols}")

            except Exception as e:
                summary.append(f"  Could not read CSV preview: {e}")
                print(f"  Could not read CSV preview: {e}")

        summary.append("")

    # Search for possible atlas/region files in project folder, excluding archives and .git.
    search_roots = [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "outputs",
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "manifests",
    ]

    atlas_keywords = [
        "atlas",
        "dk",
        "desikan",
        "killiany",
        "aseg",
        "roi",
        "mask",
        "region",
        "label",
    ]

    candidates = []

    for root in search_roots:
        if not root.exists():
            continue

        for p in root.rglob("*"):
            if not p.is_file():
                continue

            sp = str(p).lower()

            if ".git" in sp or "archive" in sp:
                continue

            name = p.name.lower()

            if any(k in name for k in atlas_keywords):
                candidates.append(p)

    summary.append("Possible atlas/ROI/mask/region files:")
    print()
    print("Possible atlas/ROI/mask/region files:")

    if candidates:
        for p in candidates[:200]:
            summary.append(str(p))
            print(p)

        if len(candidates) > 200:
            summary.append(f"... plus {len(candidates) - 200} more")
            print(f"... plus {len(candidates) - 200} more")
    else:
        summary.append("None found.")
        print("None found.")

    summary.append("")
    summary.append("Interpretation:")
    summary.append(
        "If MRI image paths are available but no anatomical atlas masks are available, "
        "we can first perform coarse occlusion/patch ablation. "
        "For anatomical region-gene interpretation, we need an atlas/ROI mask in the same space as the MRI images."
    )
    summary.append("")
    summary.append("STATUS: DONE")

    SUMMARY_OUT.write_text("\n".join(summary) + "\n")

    print()
    print(f"Summary saved to: {SUMMARY_OUT}")
    print("STATUS: DONE")
    print("=====================================================")


if __name__ == "__main__":
    main()
