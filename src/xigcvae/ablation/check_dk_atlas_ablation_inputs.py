from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/N/project/SingleCell_Image/Ronak/Project_folder")

DK_DIR = Path("/N/project/ADRD/imagingprocess/ROI/DesikanKilliany")
DK_ATLAS = DK_DIR / "desikanKillianyMNI152_2mm.nii.gz"
DK_ROI_CSV = DK_DIR / "DesikanKilliany_ROI.csv"
DK_INDEX_TXT = DK_DIR / "DesikanKilliany_Index.txt"
DK_NODE_NAMES = DK_DIR / "desikanKillianyNodeNames.txt"
DK_NODE_INDEX = DK_DIR / "desikanKillianyNodeIndex.1D"
DK_MASK_DIR = DK_DIR / "mask"

MRI_FEATURE_FILE = PROJECT_ROOT / "outputs/mri_branch/mfv_features_all_1495_mri_subjects.csv"
AGFV_COMBINED_FILE = PROJECT_ROOT / "outputs/aligned_gfv/aligned_gfv_combined_1508_subjects.csv"

OUT_DIR = PROJECT_ROOT / "outputs/ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_OUT = OUT_DIR / "check_dk_atlas_ablation_inputs_summary.txt"


def read_text_preview(path: Path, n=20):
    if not path.exists():
        return [f"MISSING: {path}"]

    lines = path.read_text(errors="ignore").splitlines()
    return lines[:n]


def main():
    print("========== CHECK DK ATLAS ABLATION INPUTS ==========")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"DK atlas dir: {DK_DIR}")
    print()

    summary = []
    summary.append("========== CHECK DK ATLAS ABLATION INPUTS ==========")
    summary.append(f"Project root: {PROJECT_ROOT}")
    summary.append(f"DK atlas dir: {DK_DIR}")
    summary.append("")

    required = [
        DK_DIR,
        DK_ATLAS,
        DK_ROI_CSV,
        DK_INDEX_TXT,
        DK_NODE_NAMES,
        DK_NODE_INDEX,
        MRI_FEATURE_FILE,
        AGFV_COMBINED_FILE,
    ]

    for p in required:
        status = "FOUND" if p.exists() else "MISSING"
        print(f"{status}: {p}")
        summary.append(f"{status}: {p}")

    print()
    summary.append("")

    if not DK_ATLAS.exists():
        raise FileNotFoundError(DK_ATLAS)

    atlas_img = nib.load(str(DK_ATLAS))
    atlas_data = np.asanyarray(atlas_img.dataobj)

    labels = np.unique(atlas_data)
    labels = labels[np.isfinite(labels)]
    labels = labels.astype(int)

    nonzero_labels = labels[labels != 0]

    print("Atlas shape:", atlas_img.shape)
    print("Atlas affine:")
    print(atlas_img.affine)
    print("Atlas voxel sizes:", atlas_img.header.get_zooms()[:3])
    print("Number of nonzero labels:", len(nonzero_labels))
    print("First 30 labels:", nonzero_labels[:30].tolist())
    print("Last 30 labels:", nonzero_labels[-30:].tolist())
    print()

    summary.append(f"Atlas shape: {atlas_img.shape}")
    summary.append("Atlas affine:")
    summary.append(str(atlas_img.affine))
    summary.append(f"Atlas voxel sizes: {atlas_img.header.get_zooms()[:3]}")
    summary.append(f"Number of nonzero labels: {len(nonzero_labels)}")
    summary.append(f"First 30 labels: {nonzero_labels[:30].tolist()}")
    summary.append(f"Last 30 labels: {nonzero_labels[-30:].tolist()}")
    summary.append("")

    if DK_ROI_CSV.exists():
        roi = pd.read_csv(DK_ROI_CSV)
        print("ROI CSV shape:", roi.shape)
        print("ROI CSV columns:", list(roi.columns))
        print("ROI CSV preview:")
        print(roi.head(20).to_string(index=False))
        print()

        summary.append(f"ROI CSV shape: {roi.shape}")
        summary.append(f"ROI CSV columns: {list(roi.columns)}")
        summary.append("ROI CSV preview:")
        summary.append(roi.head(20).to_string(index=False))
        summary.append("")
    else:
        roi = None

    print("Index TXT preview:")
    index_preview = read_text_preview(DK_INDEX_TXT, n=20)
    for line in index_preview:
        print(line)
    print()

    summary.append("Index TXT preview:")
    summary.extend(index_preview)
    summary.append("")

    print("Node names preview:")
    node_preview = read_text_preview(DK_NODE_NAMES, n=20)
    for line in node_preview:
        print(line)
    print()

    summary.append("Node names preview:")
    summary.extend(node_preview)
    summary.append("")

    print("Mask directory:")
    if DK_MASK_DIR.exists():
        masks = sorted(DK_MASK_DIR.glob("*"))
        print(f"Mask files found: {len(masks)}")
        for p in masks[:20]:
            print(p.name)
        if len(masks) > 20:
            print(f"... plus {len(masks)-20} more")
        summary.append(f"Mask files found: {len(masks)}")
        summary.extend([p.name for p in masks[:50]])
    else:
        print("Mask directory missing.")
        summary.append("Mask directory missing.")
    print()

    # Check MRI image compatibility using several heldout MRI-derived subjects.
    agfv = pd.read_csv(AGFV_COMBINED_FILE)

    mri_df = agfv[
        (agfv["agfv_source"].astype(str) == "mri_mfv_encoder")
        & (agfv["split"].astype(str).isin(["Validation", "Test"]))
    ].copy()

    if "mri_abs_path" not in mri_df.columns:
        raise ValueError("AGFV combined file does not contain mri_abs_path.")

    mri_df = mri_df.dropna(subset=["mri_abs_path"]).head(10).copy()

    print("Sample MRI image checks:")
    summary.append("Sample MRI image checks:")

    for _, row in mri_df.iterrows():
        p = Path(str(row["mri_abs_path"]))
        exists = p.exists()

        print()
        print("Subject:", row["subject_id"])
        print("Path:", p)
        print("Exists:", exists)

        summary.append("")
        summary.append(f"Subject: {row['subject_id']}")
        summary.append(f"Path: {p}")
        summary.append(f"Exists: {exists}")

        if exists:
            img = nib.load(str(p))
            print("MRI shape:", img.shape)
            print("MRI voxel sizes:", img.header.get_zooms()[:3])
            print("MRI affine:")
            print(img.affine)

            same_shape = tuple(img.shape[:3]) == tuple(atlas_img.shape[:3])
            print("Same 3D shape as DK atlas:", same_shape)

            affine_close = np.allclose(img.affine, atlas_img.affine, atol=1e-3)
            print("Affine close to DK atlas:", affine_close)

            summary.append(f"MRI shape: {img.shape}")
            summary.append(f"MRI voxel sizes: {img.header.get_zooms()[:3]}")
            summary.append("MRI affine:")
            summary.append(str(img.affine))
            summary.append(f"Same 3D shape as DK atlas: {same_shape}")
            summary.append(f"Affine close to DK atlas: {affine_close}")

    summary.append("")
    summary.append("Interpretation:")
    summary.append(
        "If MRI images and DK atlas are in the same MNI 2mm space, atlas-based region ablation can be performed directly. "
        "If shapes/affines differ, the DK atlas should be resampled or resized with nearest-neighbor interpolation to the MRI/model input space."
    )
    summary.append("")
    summary.append("STATUS: DONE")

    SUMMARY_OUT.write_text("\n".join(summary) + "\n")

    print()
    print(f"Summary saved to: {SUMMARY_OUT}")
    print("STATUS: DONE")
    print("================================================")


if __name__ == "__main__":
    main()
