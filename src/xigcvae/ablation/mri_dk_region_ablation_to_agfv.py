from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nibabel.processing import resample_from_to
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

DK_DIR = Path("/N/project/ADRD/imagingprocess/ROI/DesikanKilliany")
DK_ATLAS = DK_DIR / "desikanKillianyMNI152_2mm.nii.gz"
DK_ROI_CSV = DK_DIR / "DesikanKilliany_ROI.csv"

AGFV_COMBINED_FILE = PROJECT_ROOT / "outputs/aligned_gfv/aligned_gfv_combined_1508_subjects.csv"

MRI_MODEL_FILE = PROJECT_ROOT / "outputs/mri_branch/final_mri_branch_model.pt"
MRI_CONFIG_FILE = PROJECT_ROOT / "outputs/mri_branch/final_mri_branch_config.json"

AGFV_MODEL_FILE = PROJECT_ROOT / "outputs/aligned_gfv/final_aligned_gfv_model.pt"
AGFV_SCALER_FILE = PROJECT_ROOT / "outputs/aligned_gfv/final_aligned_gfv_scalers.pkl"
AGFV_CONFIG_FILE = PROJECT_ROOT / "outputs/aligned_gfv/final_aligned_gfv_config.json"

NN_MODEL_FILE = PROJECT_ROOT / "outputs/final_agfv_nn_classifier/final_agfv_nn_classifier_model.pt"
NN_SCALER_FILE = PROJECT_ROOT / "outputs/final_agfv_nn_classifier/final_agfv_nn_classifier_scaler.pkl"

OUT_DIR = PROJECT_ROOT / "outputs/ablation"
TABLE_DIR = PROJECT_ROOT / "outputs/tables/supplementary"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT_REGION_OUT = OUT_DIR / "mri_dk_region_ablation_subject_region.csv"
REGION_OUT = OUT_DIR / "mri_dk_region_ablation_region_level.csv"
REGION_FEATURE_OUT = OUT_DIR / "mri_dk_region_ablation_region_agfv_feature_level.csv"
SUMMARY_OUT = OUT_DIR / "mri_dk_region_ablation_summary.txt"
TABLE_OUT = TABLE_DIR / "Table_S21_DK_region_ablation_to_AGFV.xlsx"

TARGET_SHAPE = (64, 64, 64)
BATCH_SIZE = 8
ABLATE_VALUE = 0.0

# For a quick pilot, set to 20. For final manuscript-level output, keep None.
MAX_SUBJECTS_PER_SPLIT = None


def torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_agfv_cols(df: pd.DataFrame):
    pat = re.compile(r"^AGFV\d+$", re.IGNORECASE)
    cols = [c for c in df.columns if pat.match(c)]
    return sorted(cols, key=lambda x: int(re.findall(r"\d+", x)[0]))


def robust_normalize_mri(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    if data.ndim > 3:
        data = np.squeeze(data)
        if data.ndim > 3:
            data = data[..., 0]

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


def resize_float(data: np.ndarray, target_shape=TARGET_SHAPE) -> np.ndarray:
    if tuple(data.shape) == tuple(target_shape):
        return data.astype(np.float32)

    factors = [t / s for t, s in zip(target_shape, data.shape)]
    return zoom(data, zoom=factors, order=1).astype(np.float32)


def resize_label(data: np.ndarray, target_shape=TARGET_SHAPE) -> np.ndarray:
    if tuple(data.shape) == tuple(target_shape):
        return data.astype(np.int32)

    factors = [t / s for t, s in zip(target_shape, data.shape)]
    return zoom(data, zoom=factors, order=0).astype(np.int32)


def cosine(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return np.nan
    return float(np.dot(a, b) / denom)


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


def load_roi_table():
    roi = pd.read_csv(DK_ROI_CSV)
    roi["Region label"] = pd.to_numeric(roi["Index"], errors="coerce").astype("Int64")
    roi = roi.dropna(subset=["Region label"]).copy()
    roi["Region label"] = roi["Region label"].astype(int)

    roi["Region name"] = roi["ROI"].astype(str)

    keep_cols = [
        "Region label",
        "Region name",
        "hemisphere",
        "acronyms",
        "subcortex",
        "RSN",
        "BraakStage",
        "MNI.x",
        "MNI.y",
        "MNI.z",
    ]

    keep_cols = [c for c in keep_cols if c in roi.columns]
    return roi[keep_cols].drop_duplicates("Region label")


def load_mri_model(device):
    cfg = {}
    if MRI_CONFIG_FILE.exists():
        with open(MRI_CONFIG_FILE, "r") as f:
            cfg = json.load(f)

    ckpt = torch_load(MRI_MODEL_FILE, map_location=device)

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

    best_cfg = cfg.get("best_config", {})
    hidden_dim = int(best_cfg.get("hidden_dim", cfg.get("hidden_dim", 128)))
    dropout = float(best_cfg.get("dropout", cfg.get("dropout", 0.20)))

    model = AlignedGFVModel(
        gfv_dim=128,
        mfv_dim=128,
        agfv_dim=128,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    ckpt = torch_load(AGFV_MODEL_FILE, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    with open(AGFV_SCALER_FILE, "rb") as f:
        scalers = pickle.load(f)

    mfv_scaler = scalers.get("mfv_scaler", None)
    if mfv_scaler is None:
        mfv_scaler = scalers.get("mri_scaler", None)

    if mfv_scaler is None:
        raise ValueError("Could not find mfv_scaler in final_aligned_gfv_scalers.pkl")

    return model, mfv_scaler


def load_nn_classifier(device):
    ckpt = torch_load(NN_MODEL_FILE, map_location=device)

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
    agfv_cols = scaler_obj["agfv_cols"]

    return model, nn_scaler, agfv_cols, ckpt


@torch.no_grad()
def mri_to_mfv(mri_model, x_batch_np, device):
    xb = torch.tensor(x_batch_np, dtype=torch.float32, device=device)
    out = mri_model(xb)

    if isinstance(out, tuple) and len(out) >= 2:
        mfv = out[1]
    elif isinstance(out, dict):
        mfv = out.get("mfv", out.get("features", None))
        if mfv is None:
            raise ValueError("MRI model output dict does not contain mfv/features.")
    else:
        raise ValueError("Unexpected MRI model output format.")

    return mfv.detach().cpu().numpy()


@torch.no_grad()
def mfv_to_agfv(agfv_model, mfv_scaler, mfv_np, device):
    mfv_scaled = mfv_scaler.transform(mfv_np).astype(np.float32)
    xb = torch.tensor(mfv_scaled, dtype=torch.float32, device=device)

    if hasattr(agfv_model, "encode_mfv"):
        out = agfv_model.encode_mfv(xb)
    else:
        raise ValueError("AlignedGFVModel does not have encode_mfv method.")

    if isinstance(out, tuple):
        out = out[0]

    return out.detach().cpu().numpy()


@torch.no_grad()
def agfv_to_prob(nn_model, nn_scaler, agfv_np, device):
    agfv_scaled = nn_scaler.transform(agfv_np).astype(np.float32)
    xb = torch.tensor(agfv_scaled, dtype=torch.float32, device=device)
    logits = nn_model(xb)
    prob = torch.softmax(logits, dim=1)[:, 1]
    return prob.detach().cpu().numpy()


def atlas_cache_key(img):
    return (
        tuple(img.shape[:3]),
        tuple(np.round(img.affine.reshape(-1), 4)),
    )


def get_resized_atlas_labels(img, atlas_img, cache):
    key = atlas_cache_key(img)

    if key in cache:
        return cache[key]

    resampled = resample_from_to(
        atlas_img,
        (img.shape[:3], img.affine),
        order=0,
    )

    labels = np.rint(np.asanyarray(resampled.dataobj)).astype(np.int32)
    labels_64 = resize_label(labels, TARGET_SHAPE)

    cache[key] = labels_64

    return labels_64


def load_subject_mri_and_labels(path, atlas_img, atlas_cache):
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj).astype(np.float32)

    if data.ndim > 3:
        data = np.squeeze(data)
        if data.ndim > 3:
            data = data[..., 0]

    labels_64 = get_resized_atlas_labels(img, atlas_img, atlas_cache)

    data_norm = robust_normalize_mri(data)
    data_64 = resize_float(data_norm, TARGET_SHAPE)

    if labels_64.shape != data_64.shape:
        raise ValueError(f"Shape mismatch after resize: MRI={data_64.shape}, atlas={labels_64.shape}")

    return data_64[None, ...].astype(np.float32), labels_64


def make_ablated_region_batch(x, labels_64, regions):
    batch = np.repeat(x[None, ...], repeats=len(regions), axis=0)

    for i, region in enumerate(regions):
        label = int(region["Region label"])
        mask = labels_64 == label
        batch[i, 0, mask] = ABLATE_VALUE

    return batch.astype(np.float32)


def main():
    print("========== DK/ASEG MRI REGION ABLATION TO AGFV ==========")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"DK atlas: {DK_ATLAS}")
    print(f"AGFV combined file: {AGFV_COMBINED_FILE}")
    print("=========================================================\n")

    for f in [
        DK_ATLAS,
        DK_ROI_CSV,
        AGFV_COMBINED_FILE,
        MRI_MODEL_FILE,
        AGFV_MODEL_FILE,
        AGFV_SCALER_FILE,
        AGFV_CONFIG_FILE,
        NN_MODEL_FILE,
        NN_SCALER_FILE,
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

    atlas_img = nib.load(str(DK_ATLAS))
    roi_table = load_roi_table()

    df = pd.read_csv(AGFV_COMBINED_FILE)

    required_cols = [
        "subject_id",
        "diagnosis",
        "diagnosis_binary",
        "split",
        "agfv_source",
        "mri_abs_path",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing AGFV combined columns: {missing}")

    nn_model, nn_scaler, agfv_cols, nn_ckpt = load_nn_classifier(device)

    use_df = df[
        (df["agfv_source"].astype(str) == "mri_mfv_encoder")
        & (df["split"].astype(str).isin(["Validation", "Test"]))
    ].copy()

    use_df["mri_abs_path"] = use_df["mri_abs_path"].astype(str)
    use_df["mri_file_exists"] = use_df["mri_abs_path"].apply(lambda p: Path(p).exists())

    missing_files = use_df[~use_df["mri_file_exists"]]
    if len(missing_files) > 0:
        raise FileNotFoundError(
            f"{len(missing_files)} MRI files missing. Examples:\n"
            + "\n".join(missing_files["mri_abs_path"].head(10).tolist())
        )

    use_df = use_df.sort_values(["split", "subject_id"]).reset_index(drop=True)

    if MAX_SUBJECTS_PER_SPLIT is not None:
        use_df = (
            use_df.groupby("split", group_keys=False)
            .head(MAX_SUBJECTS_PER_SPLIT)
            .reset_index(drop=True)
        )

    print("MRI-derived heldout subjects for region ablation:", use_df.shape[0])
    print(pd.crosstab(use_df["split"], use_df["diagnosis"]))
    print()

    atlas_cache = {}

    first_path = Path(use_df.iloc[0]["mri_abs_path"])
    _, first_labels = load_subject_mri_and_labels(first_path, atlas_img, atlas_cache)

    available_labels = set(np.unique(first_labels).astype(int).tolist())
    available_labels.discard(0)

    roi_table = roi_table[roi_table["Region label"].isin(available_labels)].copy()
    roi_table = roi_table.sort_values("Region label").reset_index(drop=True)

    regions = roi_table.to_dict("records")

    print("DK/aseg labels available after resampling/resizing:", len(regions))
    print(roi_table[["Region label", "Region name"]].head(20).to_string(index=False))
    print()

    mri_model = load_mri_model(device)
    agfv_model, mfv_scaler = load_agfv_model(device)

    n_regions = len(regions)
    n_features = len(agfv_cols)

    subject_region_rows = []

    sum_abs_agfv_delta = np.zeros((n_regions, n_features), dtype=np.float64)
    sum_signed_agfv_delta = np.zeros((n_regions, n_features), dtype=np.float64)
    region_subject_count = np.zeros(n_regions, dtype=np.int64)

    baseline_probs = []
    baseline_labels = []
    baseline_splits = []
    baseline_saved_agfv_cosines = []
    baseline_saved_agfv_l2 = []
    baseline_saved_prob_diffs = []

    for idx, row in use_df.iterrows():
        if idx == 0 or (idx + 1) % 25 == 0:
            print(f"Processing subject {idx + 1}/{use_df.shape[0]}: {row['subject_id']}")

        subject_id = row["subject_id"]
        y_true = int(row["diagnosis_binary"])
        split = str(row["split"])
        diagnosis = row["diagnosis"]
        mri_path = Path(row["mri_abs_path"])

        x, labels_64 = load_subject_mri_and_labels(mri_path, atlas_img, atlas_cache)

        base_mfv = mri_to_mfv(mri_model, x[None, ...], device)
        base_agfv = mfv_to_agfv(agfv_model, mfv_scaler, base_mfv, device)
        base_prob = float(agfv_to_prob(nn_model, nn_scaler, base_agfv, device)[0])
        base_pred = int(base_prob >= 0.5)

        saved_agfv = row[agfv_cols].astype(float).values.reshape(1, -1)
        saved_prob = float(agfv_to_prob(nn_model, nn_scaler, saved_agfv, device)[0])

        baseline_saved_agfv_cosines.append(cosine(base_agfv[0], saved_agfv[0]))
        baseline_saved_agfv_l2.append(float(np.linalg.norm(base_agfv[0] - saved_agfv[0])))
        baseline_saved_prob_diffs.append(float(abs(base_prob - saved_prob)))

        baseline_probs.append(base_prob)
        baseline_labels.append(y_true)
        baseline_splits.append(split)

        ab_batch = make_ablated_region_batch(x, labels_64, regions)

        ab_mfv_all = []
        for start in range(0, n_regions, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_regions)
            ab_mfv = mri_to_mfv(mri_model, ab_batch[start:end], device)
            ab_mfv_all.append(ab_mfv)

        ab_mfv_all = np.vstack(ab_mfv_all)
        ab_agfv_all = mfv_to_agfv(agfv_model, mfv_scaler, ab_mfv_all, device)
        ab_prob_all = agfv_to_prob(nn_model, nn_scaler, ab_agfv_all, device)

        agfv_delta = ab_agfv_all - base_agfv[0][None, :]
        abs_agfv_delta = np.abs(agfv_delta)

        for ri, region in enumerate(regions):
            label = int(region["Region label"])
            mask_voxels = int(np.sum(labels_64 == label))

            prob_ab = float(ab_prob_all[ri])
            pred_ab = int(prob_ab >= 0.5)

            sum_abs_agfv_delta[ri] += abs_agfv_delta[ri]
            sum_signed_agfv_delta[ri] += agfv_delta[ri]
            region_subject_count[ri] += 1

            top_feature_idx = int(np.argmax(abs_agfv_delta[ri]))

            out_row = {
                "subject_id": subject_id,
                "diagnosis": diagnosis,
                "diagnosis_binary": y_true,
                "split": split,
                "Region label": label,
                "Region name": region.get("Region name"),
                "hemisphere": region.get("hemisphere"),
                "acronyms": region.get("acronyms"),
                "subcortex": region.get("subcortex"),
                "RSN": region.get("RSN"),
                "BraakStage": region.get("BraakStage"),
                "Mask voxels in 64^3": mask_voxels,
                "Baseline AD probability": base_prob,
                "Ablated AD probability": prob_ab,
                "Signed probability change": base_prob - prob_ab,
                "Absolute probability change": abs(base_prob - prob_ab),
                "Baseline prediction": base_pred,
                "Ablated prediction": pred_ab,
                "Prediction changed": int(base_pred != pred_ab),
                "AGFV L2 change": float(np.linalg.norm(agfv_delta[ri])),
                "AGFV mean absolute change": float(abs_agfv_delta[ri].mean()),
                "Top changed AGFV feature": agfv_cols[top_feature_idx],
                "Top changed AGFV absolute delta": float(abs_agfv_delta[ri, top_feature_idx]),
            }

            for coord_col in ["MNI.x", "MNI.y", "MNI.z"]:
                if coord_col in region:
                    out_row[coord_col] = region.get(coord_col)

            subject_region_rows.append(out_row)

    subject_region_df = pd.DataFrame(subject_region_rows)

    baseline_metrics_rows = []
    baseline_probs_np = np.asarray(baseline_probs)
    baseline_labels_np = np.asarray(baseline_labels)
    baseline_splits_np = np.asarray(baseline_splits)

    for group_name, mask in [
        ("Heldout", np.ones_like(baseline_labels_np, dtype=bool)),
        ("Validation", baseline_splits_np == "Validation"),
        ("Test", baseline_splits_np == "Test"),
    ]:
        m = compute_metrics(baseline_labels_np[mask], baseline_probs_np[mask])
        baseline_metrics_rows.append({"Split group": group_name, **m})

    baseline_metrics_df = pd.DataFrame(baseline_metrics_rows)

    region_rows = []

    for ri, region in enumerate(regions):
        label = int(region["Region label"])
        sub = subject_region_df[subject_region_df["Region label"] == label]

        for group_name, group_sub in [
            ("Heldout", sub),
            ("Validation", sub[sub["split"] == "Validation"]),
            ("Test", sub[sub["split"] == "Test"]),
        ]:
            if len(group_sub) == 0:
                continue

            base_group = baseline_metrics_df[baseline_metrics_df["Split group"] == group_name].iloc[0].to_dict()
            ab_metrics = compute_metrics(
                group_sub["diagnosis_binary"].values,
                group_sub["Ablated AD probability"].values,
            )

            row_out = {
                "Split group": group_name,
                "Region label": label,
                "Region name": region.get("Region name"),
                "hemisphere": region.get("hemisphere"),
                "acronyms": region.get("acronyms"),
                "subcortex": region.get("subcortex"),
                "RSN": region.get("RSN"),
                "BraakStage": region.get("BraakStage"),
                "Subjects, n": int(len(group_sub)),
                "Mean mask voxels in 64^3": float(group_sub["Mask voxels in 64^3"].mean()),
                "Baseline AUC": base_group["AUC"],
                "Ablated AUC": ab_metrics["AUC"],
                "Delta AUC": base_group["AUC"] - ab_metrics["AUC"],
                "Baseline balanced accuracy": base_group["Balanced accuracy"],
                "Ablated balanced accuracy": ab_metrics["Balanced accuracy"],
                "Delta balanced accuracy": base_group["Balanced accuracy"] - ab_metrics["Balanced accuracy"],
                "Baseline accuracy": base_group["Accuracy"],
                "Ablated accuracy": ab_metrics["Accuracy"],
                "Delta accuracy": base_group["Accuracy"] - ab_metrics["Accuracy"],
                "Baseline F1": base_group["F1"],
                "Ablated F1": ab_metrics["F1"],
                "Delta F1": base_group["F1"] - ab_metrics["F1"],
                "Mean absolute probability change": float(group_sub["Absolute probability change"].mean()),
                "Median absolute probability change": float(group_sub["Absolute probability change"].median()),
                "Mean signed probability change": float(group_sub["Signed probability change"].mean()),
                "Prediction changed, n": int(group_sub["Prediction changed"].sum()),
                "Prediction changed, fraction": float(group_sub["Prediction changed"].mean()),
                "Mean AGFV L2 change": float(group_sub["AGFV L2 change"].mean()),
                "Mean AGFV absolute change": float(group_sub["AGFV mean absolute change"].mean()),
                "Mean top AGFV absolute delta": float(group_sub["Top changed AGFV absolute delta"].mean()),
            }

            for coord_col in ["MNI.x", "MNI.y", "MNI.z"]:
                if coord_col in region:
                    row_out[coord_col] = region.get(coord_col)

            region_rows.append(row_out)

    region_df = pd.DataFrame(region_rows)

    region_df["Region ablation score"] = (
        region_df["Delta AUC"].fillna(0)
        + region_df["Delta balanced accuracy"].fillna(0)
        + region_df["Mean absolute probability change"].fillna(0)
        + 0.10 * region_df["Prediction changed, fraction"].fillna(0)
        + region_df["Mean AGFV absolute change"].fillna(0)
    )

    feature_rows = []

    for ri, region in enumerate(regions):
        count = max(int(region_subject_count[ri]), 1)
        mean_abs = sum_abs_agfv_delta[ri] / count
        mean_signed = sum_signed_agfv_delta[ri] / count

        for fj, feature_name in enumerate(agfv_cols):
            row_out = {
                "Region label": int(region["Region label"]),
                "Region name": region.get("Region name"),
                "hemisphere": region.get("hemisphere"),
                "acronyms": region.get("acronyms"),
                "subcortex": region.get("subcortex"),
                "RSN": region.get("RSN"),
                "BraakStage": region.get("BraakStage"),
                "AGFV feature": feature_name,
                "Mean absolute AGFV feature change": float(mean_abs[fj]),
                "Mean signed AGFV feature change": float(mean_signed[fj]),
            }

            for coord_col in ["MNI.x", "MNI.y", "MNI.z"]:
                if coord_col in region:
                    row_out[coord_col] = region.get(coord_col)

            feature_rows.append(row_out)

    feature_df = pd.DataFrame(feature_rows)

    subject_region_df.to_csv(SUBJECT_REGION_OUT, index=False)
    region_df.to_csv(REGION_OUT, index=False)
    feature_df.to_csv(REGION_FEATURE_OUT, index=False)

    heldout_region = region_df[region_df["Split group"] == "Heldout"].copy()
    heldout_region = heldout_region.sort_values(
        ["Region ablation score", "Delta AUC", "Mean absolute probability change", "Mean AGFV absolute change"],
        ascending=False,
    )

    top_region_feature = feature_df.sort_values(
        "Mean absolute AGFV feature change",
        ascending=False,
    )

    with pd.ExcelWriter(TABLE_OUT) as writer:
        heldout_region.to_excel(writer, sheet_name="Heldout_region_ablation", index=False)
        region_df.to_excel(writer, sheet_name="All_region_ablation", index=False)
        top_region_feature.head(3000).to_excel(writer, sheet_name="Top_region_AGFV_features", index=False)
        baseline_metrics_df.to_excel(writer, sheet_name="Baseline_metrics", index=False)

    summary = []
    summary.append("========== DK/ASEG MRI REGION ABLATION TO AGFV ==========")
    summary.append(f"Subjects analyzed: {use_df.shape[0]}")
    summary.append(f"Regions analyzed: {n_regions}")
    summary.append(f"Target MRI shape used by model: {TARGET_SHAPE}")
    summary.append(f"Best NN config: {nn_ckpt['config']['name']}")
    summary.append("")
    summary.append("Baseline metrics from MRI image -> MFV -> AGFV -> NN:")
    summary.append(baseline_metrics_df.to_string(index=False))
    summary.append("")
    summary.append("Baseline consistency check against saved MRI-derived AGFV:")
    summary.append(f"Mean AGFV cosine, rerun image AGFV vs saved AGFV: {np.nanmean(baseline_saved_agfv_cosines):.6f}")
    summary.append(f"Mean AGFV L2 distance, rerun image AGFV vs saved AGFV: {np.nanmean(baseline_saved_agfv_l2):.6f}")
    summary.append(f"Mean absolute NN probability difference, rerun AGFV vs saved AGFV: {np.nanmean(baseline_saved_prob_diffs):.6f}")
    summary.append("")
    summary.append("Top 25 heldout DK/aseg regions by ablation score:")
    summary.append(
        heldout_region[
            [
                "Region label",
                "Region name",
                "hemisphere",
                "RSN",
                "Delta AUC",
                "Delta balanced accuracy",
                "Mean absolute probability change",
                "Prediction changed, fraction",
                "Mean AGFV absolute change",
                "Region ablation score",
            ]
        ]
        .head(25)
        .to_string(index=False)
    )
    summary.append("")
    summary.append("Top 25 region-AGFV feature changes:")
    summary.append(
        top_region_feature[
            [
                "Region label",
                "Region name",
                "hemisphere",
                "RSN",
                "AGFV feature",
                "Mean absolute AGFV feature change",
                "Mean signed AGFV feature change",
            ]
        ]
        .head(25)
        .to_string(index=False)
    )
    summary.append("")
    summary.append("Saved:")
    summary.append(str(SUBJECT_REGION_OUT))
    summary.append(str(REGION_OUT))
    summary.append(str(REGION_FEATURE_OUT))
    summary.append(str(TABLE_OUT))
    summary.append("STATUS: DONE")

    SUMMARY_OUT.write_text("\n".join(summary) + "\n")

    print()
    print("\n".join(summary))
    print("==========================================================")


if __name__ == "__main__":
    main()
