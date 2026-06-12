from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.xigcvae.models.mri_branch import MRI3DBranch
from src.xigcvae.models.aligned_gfv_model import AlignedGFVModel
from src.xigcvae.models.agfv_nn_classifier import AGFVNNClassifier


PROJECT_ROOT = Path("/N/project/SingleCell_Image/Ronak/Project_folder")

AGFV_COMBINED_FILE = PROJECT_ROOT / "outputs/aligned_gfv/aligned_gfv_combined_1508_subjects.csv"

MRI_MODEL_FILE = PROJECT_ROOT / "outputs/mri_branch/final_mri_branch_model.pt"
MRI_CONFIG_FILE = PROJECT_ROOT / "outputs/mri_branch/final_mri_branch_config.json"

AGFV_MODEL_FILE = PROJECT_ROOT / "outputs/aligned_gfv/final_aligned_gfv_model.pt"
AGFV_SCALER_FILE = PROJECT_ROOT / "outputs/aligned_gfv/final_aligned_gfv_scalers.pkl"
AGFV_CONFIG_FILE = PROJECT_ROOT / "outputs/aligned_gfv/final_aligned_gfv_config.json"

NN_MODEL_FILE = PROJECT_ROOT / "outputs/final_agfv_nn_classifier/final_agfv_nn_classifier_model.pt"
NN_SCALER_FILE = PROJECT_ROOT / "outputs/final_agfv_nn_classifier/final_agfv_nn_classifier_scaler.pkl"
NN_CONFIG_FILE = PROJECT_ROOT / "outputs/final_agfv_nn_classifier/final_agfv_nn_classifier_config.json"

OUT_DIR = PROJECT_ROOT / "outputs/ablation"
TABLE_DIR = PROJECT_ROOT / "outputs/tables/supplementary"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT_PATCH_OUT = OUT_DIR / "mri_patch_ablation_subject_patch.csv"
PATCH_OUT = OUT_DIR / "mri_patch_ablation_patch_level.csv"
PATCH_FEATURE_OUT = OUT_DIR / "mri_patch_ablation_patch_agfv_feature_level.csv"
SUMMARY_OUT = OUT_DIR / "mri_patch_ablation_summary.txt"
TABLE_OUT = TABLE_DIR / "Table_S21_MRI_patch_ablation_to_AGFV.xlsx"

TARGET_SHAPE = (64, 64, 64)
GRID_SHAPE = (4, 4, 4)
BATCH_SIZE = 8

# Use all validation/test MRI-derived subjects.
# Set to an integer only for a pilot; keep None for manuscript-level output.
MAX_SUBJECTS = None


def get_agfv_cols(df: pd.DataFrame):
    pat = re.compile(r"^AGFV\d+$", re.IGNORECASE)
    cols = [c for c in df.columns if pat.match(c)]
    return sorted(cols, key=lambda x: int(re.findall(r"\d+", x)[0]))


def resolve_path(path_value):
    p = Path(str(path_value))
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def robust_normalize_mri(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    mask = np.isfinite(data) & (data != 0)

    if mask.sum() < 100:
        return np.zeros_like(data, dtype=np.float32)

    vals = data[mask]
    lo, hi = np.percentile(vals, [1, 99])

    clipped = np.clip(data, lo, hi)
    vals_clip = clipped[mask]

    mean = vals_clip.mean()
    std = vals_clip.std()

    if std < 1e-6:
        std = 1.0

    normed = (clipped - mean) / std
    normed[~mask] = 0.0

    return normed.astype(np.float32)


def resize_mri(data: np.ndarray, target_shape=TARGET_SHAPE) -> np.ndarray:
    if tuple(data.shape) == tuple(target_shape):
        return data.astype(np.float32)

    factors = [t / s for t, s in zip(target_shape, data.shape)]
    resized = zoom(data, zoom=factors, order=1)

    return resized.astype(np.float32)


def load_mri_volume(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj).astype(np.float32)
    data = robust_normalize_mri(data)
    data = resize_mri(data, TARGET_SHAPE)
    return data[None, ...].astype(np.float32)


def make_patches():
    patches = []

    nx, ny, nz = GRID_SHAPE
    sx, sy, sz = TARGET_SHAPE

    x_edges = np.linspace(0, sx, nx + 1, dtype=int)
    y_edges = np.linspace(0, sy, ny + 1, dtype=int)
    z_edges = np.linspace(0, sz, nz + 1, dtype=int)

    patch_index = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                x0, x1 = int(x_edges[ix]), int(x_edges[ix + 1])
                y0, y1 = int(y_edges[iy]), int(y_edges[iy + 1])
                z0, z1 = int(z_edges[iz]), int(z_edges[iz + 1])

                patches.append(
                    {
                        "Patch ID": f"patch_x{ix}_y{iy}_z{iz}",
                        "Patch index": patch_index,
                        "Grid x": ix,
                        "Grid y": iy,
                        "Grid z": iz,
                        "x0": x0,
                        "x1": x1,
                        "y0": y0,
                        "y1": y1,
                        "z0": z0,
                        "z1": z1,
                        "Patch label": f"x[{x0}:{x1}]_y[{y0}:{y1}]_z[{z0}:{z1}]",
                    }
                )
                patch_index += 1

    return patches


def compute_metrics(y_true, prob):
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    pred = (prob >= 0.5).astype(int)

    auc = float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) == 2 else np.nan
    acc = float(accuracy_score(y_true, pred))
    bacc = float(balanced_accuracy_score(y_true, pred))
    precision = float(precision_score(y_true, pred, zero_division=0))
    recall = float(recall_score(y_true, pred, zero_division=0))
    f1 = float(f1_score(y_true, pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "AUC": auc,
        "Accuracy": acc,
        "Balanced accuracy": bacc,
        "Precision": precision,
        "AD sensitivity": recall,
        "CN specificity": float(spec),
        "F1": f1,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def load_mri_model(device):
    cfg = {}
    if MRI_CONFIG_FILE.exists():
        with open(MRI_CONFIG_FILE, "r") as f:
            cfg = json.load(f)

    ckpt = torch.load(MRI_MODEL_FILE, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        mfv_dim = int(ckpt.get("mfv_dim", cfg.get("mfv_dim", 128)))
        dropout = float(ckpt.get("dropout", cfg.get("dropout", 0.30)))
    else:
        state = ckpt
        mfv_dim = int(cfg.get("mfv_dim", 128))
        dropout = float(cfg.get("dropout", 0.30))

    model = MRI3DBranch(mfv_dim=mfv_dim, dropout=dropout, num_classes=2).to(device)
    model.load_state_dict(state)
    model.eval()

    return model


def load_agfv_model(device):
    with open(AGFV_CONFIG_FILE, "r") as f:
        cfg = json.load(f)

    ckpt = torch.load(AGFV_MODEL_FILE, map_location=device)

    best_cfg = cfg.get("best_config", {})
    hidden_dim = int(best_cfg.get("hidden_dim", 128))
    dropout = float(best_cfg.get("dropout", 0.20))

    model = AlignedGFVModel(
        gfv_dim=128,
        mfv_dim=128,
        agfv_dim=128,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    model.load_state_dict(state)
    model.eval()

    with open(AGFV_SCALER_FILE, "rb") as f:
        scalers = pickle.load(f)

    mfv_scaler = scalers["mfv_scaler"]
    agfv_cols = scalers["agfv_cols"]

    return model, mfv_scaler, agfv_cols


def load_nn_classifier(device):
    with open(NN_CONFIG_FILE, "r") as f:
        cfg_json = json.load(f)

    ckpt = torch.load(NN_MODEL_FILE, map_location=device)

    best_cfg = ckpt["config"]
    hidden_dims = tuple(best_cfg["hidden_dims"])
    dropout = float(best_cfg["dropout"])

    model = AGFVNNClassifier(
        input_dim=128,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with open(NN_SCALER_FILE, "rb") as f:
        scaler_obj = pickle.load(f)

    nn_scaler = scaler_obj["scaler"]

    return model, nn_scaler, ckpt


@torch.no_grad()
def mri_to_mfv(mri_model, x_batch_np, device):
    xb = torch.tensor(x_batch_np, dtype=torch.float32, device=device)
    _, mfv = mri_model(xb)
    return mfv.cpu().numpy()


@torch.no_grad()
def mfv_to_agfv(agfv_model, mfv_scaler, mfv_np, device):
    mfv_scaled = mfv_scaler.transform(mfv_np).astype(np.float32)
    xb = torch.tensor(mfv_scaled, dtype=torch.float32, device=device)
    agfv = agfv_model.encode_mfv(xb)
    return agfv.cpu().numpy()


@torch.no_grad()
def agfv_to_prob(nn_model, nn_scaler, agfv_np, device):
    agfv_scaled = nn_scaler.transform(agfv_np).astype(np.float32)
    xb = torch.tensor(agfv_scaled, dtype=torch.float32, device=device)
    logits = nn_model(xb)
    prob = torch.softmax(logits, dim=1)[:, 1]
    return prob.cpu().numpy()


def make_ablated_batch(x_np, patches):
    # x_np shape: 1 x 64 x 64 x 64
    batch = np.repeat(x_np[None, ...], repeats=len(patches), axis=0)

    for i, p in enumerate(patches):
        batch[
            i,
            :,
            p["x0"]:p["x1"],
            p["y0"]:p["y1"],
            p["z0"]:p["z1"],
        ] = 0.0

    return batch.astype(np.float32)


def main():
    print("========== MRI PATCH ABLATION TO AGFV ==========")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"AGFV combined file: {AGFV_COMBINED_FILE}")
    print(f"Grid shape: {GRID_SHAPE}")
    print("================================================\n")

    for f in [
        AGFV_COMBINED_FILE,
        MRI_MODEL_FILE,
        AGFV_MODEL_FILE,
        AGFV_SCALER_FILE,
        AGFV_CONFIG_FILE,
        NN_MODEL_FILE,
        NN_SCALER_FILE,
        NN_CONFIG_FILE,
    ]:
        if not f.exists():
            raise FileNotFoundError(f)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print()

    df = pd.read_csv(AGFV_COMBINED_FILE)

    required_cols = ["subject_id", "diagnosis", "diagnosis_binary", "split", "agfv_source", "mri_abs_path"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in AGFV combined file: {missing}")

    use_df = df[
        (df["agfv_source"].astype(str) == "mri_mfv_encoder")
        & (df["split"].astype(str).isin(["Validation", "Test"]))
    ].copy()

    use_df["mri_path_resolved"] = use_df["mri_abs_path"].apply(resolve_path)
    use_df["mri_file_exists"] = use_df["mri_path_resolved"].apply(lambda p: Path(p).exists())

    missing_files = use_df[~use_df["mri_file_exists"]]
    if len(missing_files) > 0:
        raise FileNotFoundError(
            f"{len(missing_files)} MRI files missing. Examples:\n"
            + "\n".join(missing_files["mri_path_resolved"].astype(str).head(10).tolist())
        )

    use_df = use_df.sort_values(["split", "subject_id"]).reset_index(drop=True)

    if MAX_SUBJECTS is not None:
        use_df = use_df.groupby("split", group_keys=False).head(MAX_SUBJECTS).reset_index(drop=True)

    print("Subjects for MRI patch ablation:", use_df.shape[0])
    print(pd.crosstab(use_df["split"], use_df["diagnosis"]))
    print()

    patches = make_patches()
    n_patches = len(patches)
    n_features = 128

    mri_model = load_mri_model(device)
    agfv_model, mfv_scaler, agfv_cols = load_agfv_model(device)
    nn_model, nn_scaler, nn_ckpt = load_nn_classifier(device)

    subject_patch_rows = []

    sum_abs_agfv_delta = np.zeros((n_patches, n_features), dtype=np.float64)
    sum_signed_agfv_delta = np.zeros((n_patches, n_features), dtype=np.float64)
    patch_subject_count = np.zeros(n_patches, dtype=np.int64)

    baseline_probs = []
    baseline_labels = []
    baseline_splits = []

    for idx, row in use_df.iterrows():
        if idx == 0 or (idx + 1) % 25 == 0:
            print(f"Processing subject {idx + 1}/{use_df.shape[0]}: {row['subject_id']}")

        y_true = int(row["diagnosis_binary"])
        split = str(row["split"])
        mri_path = Path(row["mri_path_resolved"])

        x = load_mri_volume(mri_path)

        base_mfv = mri_to_mfv(mri_model, x[None, ...], device)
        base_agfv = mfv_to_agfv(agfv_model, mfv_scaler, base_mfv, device)
        base_prob = float(agfv_to_prob(nn_model, nn_scaler, base_agfv, device)[0])
        base_pred = int(base_prob >= 0.5)

        baseline_probs.append(base_prob)
        baseline_labels.append(y_true)
        baseline_splits.append(split)

        ab_batch = make_ablated_batch(x, patches)

        ab_mfv_all = []
        for start in range(0, n_patches, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_patches)
            ab_mfv = mri_to_mfv(mri_model, ab_batch[start:end], device)
            ab_mfv_all.append(ab_mfv)

        ab_mfv_all = np.vstack(ab_mfv_all)
        ab_agfv_all = mfv_to_agfv(agfv_model, mfv_scaler, ab_mfv_all, device)
        ab_prob_all = agfv_to_prob(nn_model, nn_scaler, ab_agfv_all, device)

        agfv_delta = ab_agfv_all - base_agfv[0][None, :]
        abs_agfv_delta = np.abs(agfv_delta)

        for p in patches:
            pi = p["Patch index"]

            prob_ab = float(ab_prob_all[pi])
            pred_ab = int(prob_ab >= 0.5)

            sum_abs_agfv_delta[pi] += abs_agfv_delta[pi]
            sum_signed_agfv_delta[pi] += agfv_delta[pi]
            patch_subject_count[pi] += 1

            subject_patch_rows.append(
                {
                    "subject_id": row["subject_id"],
                    "diagnosis": row["diagnosis"],
                    "diagnosis_binary": y_true,
                    "split": split,
                    "Patch ID": p["Patch ID"],
                    "Patch index": pi,
                    "Grid x": p["Grid x"],
                    "Grid y": p["Grid y"],
                    "Grid z": p["Grid z"],
                    "Patch label": p["Patch label"],
                    "Baseline AD probability": base_prob,
                    "Ablated AD probability": prob_ab,
                    "Signed probability change": base_prob - prob_ab,
                    "Absolute probability change": abs(base_prob - prob_ab),
                    "Baseline prediction": base_pred,
                    "Ablated prediction": pred_ab,
                    "Prediction changed": int(base_pred != pred_ab),
                    "AGFV L2 change": float(np.linalg.norm(agfv_delta[pi])),
                    "AGFV mean absolute change": float(abs_agfv_delta[pi].mean()),
                    "Top changed AGFV feature": agfv_cols[int(np.argmax(abs_agfv_delta[pi]))],
                    "Top changed AGFV absolute delta": float(abs_agfv_delta[pi].max()),
                }
            )

    subject_patch_df = pd.DataFrame(subject_patch_rows)

    baseline_metrics_rows = []
    baseline_all = compute_metrics(np.asarray(baseline_labels), np.asarray(baseline_probs))
    baseline_metrics_rows.append({"Split group": "Heldout", **baseline_all})

    for split_name in ["Validation", "Test"]:
        mask = np.asarray(baseline_splits) == split_name
        m = compute_metrics(np.asarray(baseline_labels)[mask], np.asarray(baseline_probs)[mask])
        baseline_metrics_rows.append({"Split group": split_name, **m})

    patch_rows = []
    for p in patches:
        pi = p["Patch index"]
        sub = subject_patch_df[subject_patch_df["Patch index"] == pi]

        for group_name, group_sub in [
            ("Heldout", sub),
            ("Validation", sub[sub["split"] == "Validation"]),
            ("Test", sub[sub["split"] == "Test"]),
        ]:
            if len(group_sub) == 0:
                continue

            patch_rows.append(
                {
                    "Split group": group_name,
                    "Patch ID": p["Patch ID"],
                    "Patch index": pi,
                    "Grid x": p["Grid x"],
                    "Grid y": p["Grid y"],
                    "Grid z": p["Grid z"],
                    "Patch label": p["Patch label"],
                    "Subjects, n": int(len(group_sub)),
                    "Mean absolute probability change": float(group_sub["Absolute probability change"].mean()),
                    "Median absolute probability change": float(group_sub["Absolute probability change"].median()),
                    "Mean signed probability change": float(group_sub["Signed probability change"].mean()),
                    "Prediction changed, n": int(group_sub["Prediction changed"].sum()),
                    "Prediction changed, fraction": float(group_sub["Prediction changed"].mean()),
                    "Mean AGFV L2 change": float(group_sub["AGFV L2 change"].mean()),
                    "Mean AGFV absolute change": float(group_sub["AGFV mean absolute change"].mean()),
                    "Mean top AGFV absolute delta": float(group_sub["Top changed AGFV absolute delta"].mean()),
                }
            )

    patch_df = pd.DataFrame(patch_rows)

    patch_df["Patch ablation score"] = (
        patch_df["Mean absolute probability change"].fillna(0)
        + 0.10 * patch_df["Prediction changed, fraction"].fillna(0)
        + patch_df["Mean AGFV absolute change"].fillna(0)
    )

    feature_rows = []
    for p in patches:
        pi = p["Patch index"]
        count = max(int(patch_subject_count[pi]), 1)

        mean_abs = sum_abs_agfv_delta[pi] / count
        mean_signed = sum_signed_agfv_delta[pi] / count

        for j, feature_name in enumerate(agfv_cols):
            feature_rows.append(
                {
                    "Patch ID": p["Patch ID"],
                    "Patch index": pi,
                    "Grid x": p["Grid x"],
                    "Grid y": p["Grid y"],
                    "Grid z": p["Grid z"],
                    "Patch label": p["Patch label"],
                    "AGFV feature": feature_name,
                    "Mean absolute AGFV feature change": float(mean_abs[j]),
                    "Mean signed AGFV feature change": float(mean_signed[j]),
                }
            )

    feature_df = pd.DataFrame(feature_rows)

    subject_patch_df.to_csv(SUBJECT_PATCH_OUT, index=False)
    patch_df.to_csv(PATCH_OUT, index=False)
    feature_df.to_csv(PATCH_FEATURE_OUT, index=False)

    heldout_patch = patch_df[patch_df["Split group"] == "Heldout"].sort_values(
        ["Patch ablation score", "Mean absolute probability change", "Mean AGFV absolute change"],
        ascending=False,
    )

    heldout_feature = feature_df.copy()
    heldout_feature = heldout_feature.sort_values(
        ["Mean absolute AGFV feature change"],
        ascending=False,
    )

    with pd.ExcelWriter(TABLE_OUT) as writer:
        heldout_patch.to_excel(writer, sheet_name="Heldout_patch_ablation", index=False)
        patch_df.to_excel(writer, sheet_name="All_patch_ablation", index=False)
        heldout_feature.head(2000).to_excel(writer, sheet_name="Top_patch_AGFV_features", index=False)
        pd.DataFrame(baseline_metrics_rows).to_excel(writer, sheet_name="Baseline_metrics", index=False)

    summary = []
    summary.append("========== MRI PATCH ABLATION TO AGFV ==========")
    summary.append(f"Subjects analyzed: {use_df.shape[0]}")
    summary.append(f"Grid shape: {GRID_SHAPE}")
    summary.append(f"Patches: {n_patches}")
    summary.append(f"Best NN config: {nn_ckpt['config']['name']}")
    summary.append("")
    summary.append("Baseline metrics from MRI-image -> MFV -> AGFV -> NN:")
    summary.append(pd.DataFrame(baseline_metrics_rows).to_string(index=False))
    summary.append("")
    summary.append("Top 20 heldout MRI patches by ablation score:")
    summary.append(
        heldout_patch[
            [
                "Patch ID",
                "Patch label",
                "Mean absolute probability change",
                "Prediction changed, fraction",
                "Mean AGFV absolute change",
                "Patch ablation score",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )
    summary.append("")
    summary.append("Top 20 patch-AGFV feature changes:")
    summary.append(
        heldout_feature[
            [
                "Patch ID",
                "Patch label",
                "AGFV feature",
                "Mean absolute AGFV feature change",
                "Mean signed AGFV feature change",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )
    summary.append("")
    summary.append("Saved:")
    summary.append(str(SUBJECT_PATCH_OUT))
    summary.append(str(PATCH_OUT))
    summary.append(str(PATCH_FEATURE_OUT))
    summary.append(str(TABLE_OUT))
    summary.append("STATUS: DONE")

    SUMMARY_OUT.write_text("\n".join(summary) + "\n")

    print()
    print("\n".join(summary))
    print("================================================")


if __name__ == "__main__":
    main()
