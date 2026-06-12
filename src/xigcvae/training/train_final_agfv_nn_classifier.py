from __future__ import annotations

import copy
import json
import pickle
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.xigcvae.models.agfv_nn_classifier import AGFVNNClassifier


PROJECT_ROOT = Path("/N/project/SingleCell_Image/Ronak/Project_folder")

AGFV_FILE = PROJECT_ROOT / "outputs/aligned_gfv/aligned_gfv_combined_1508_subjects.csv"

OUT_DIR = PROJECT_ROOT / "outputs/final_agfv_nn_classifier"
TABLE_SUPP_DIR = PROJECT_ROOT / "outputs/tables/supplementary"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_SUPP_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 64
MAX_EPOCHS = 500
PATIENCE = 70

CONFIGS = [
    {
        "name": "mlp_64_do020_wd001_lr0003",
        "hidden_dims": (64,),
        "dropout": 0.20,
        "weight_decay": 1e-3,
        "lr": 3e-4,
    },
    {
        "name": "mlp_64_do035_wd002_lr0003",
        "hidden_dims": (64,),
        "dropout": 0.35,
        "weight_decay": 2e-3,
        "lr": 3e-4,
    },
    {
        "name": "mlp_128_do035_wd002_lr0002",
        "hidden_dims": (128,),
        "dropout": 0.35,
        "weight_decay": 2e-3,
        "lr": 2e-4,
    },
    {
        "name": "mlp_64_32_do035_wd002_lr0003",
        "hidden_dims": (64, 32),
        "dropout": 0.35,
        "weight_decay": 2e-3,
        "lr": 3e-4,
    },
    {
        "name": "mlp_128_64_do040_wd003_lr0002",
        "hidden_dims": (128, 64),
        "dropout": 0.40,
        "weight_decay": 3e-3,
        "lr": 2e-4,
    },
]


class AGFVDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_agfv_cols(df: pd.DataFrame):
    pat = re.compile(r"^AGFV\d+$", re.IGNORECASE)
    cols = [c for c in df.columns if pat.match(c)]
    return sorted(cols, key=lambda x: int(re.findall(r"\d+", x)[0]))


def get_label(df: pd.DataFrame):
    if "diagnosis_binary" in df.columns:
        return df["diagnosis_binary"].astype(int).values

    if "diagnosis" in df.columns:
        return df["diagnosis"].map({"CN": 0, "AD": 1}).astype(int).values

    raise ValueError("Input file must contain diagnosis_binary or diagnosis.")


def compute_metrics(y_true, prob):
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).astype(float)
    pred = (prob >= 0.5).astype(int)

    auc = roc_auc_score(y_true, prob) if len(np.unique(y_true)) == 2 else np.nan
    acc = accuracy_score(y_true, pred)
    bacc = balanced_accuracy_score(y_true, pred)
    precision = precision_score(y_true, pred, zero_division=0)
    recall = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return {
        "AUC": float(auc),
        "Accuracy": float(acc),
        "Balanced accuracy": float(bacc),
        "Precision": float(precision),
        "AD sensitivity": float(recall),
        "CN specificity": float(specificity),
        "F1": float(f1),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


@torch.no_grad()
def predict_prob(model, X, device):
    model.eval()
    probs = []

    for start in range(0, X.shape[0], 512):
        end = min(start + 512, X.shape[0])
        xb = torch.tensor(X[start:end], dtype=torch.float32, device=device)
        logits = model(xb)
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        probs.append(prob)

    return np.concatenate(probs)


def metric_row(model_name, split_name, subgroup_name, y, prob):
    m = compute_metrics(y, prob)
    row = {
        "Model": model_name,
        "Split": split_name,
        "Subgroup": subgroup_name,
        "Subjects, n": int(len(y)),
        "AD, n": int(np.sum(np.asarray(y) == 1)),
        "CN, n": int(np.sum(np.asarray(y) == 0)),
    }
    row.update(m)
    return row


def train_one_config(cfg, X, y, train_idx, val_idx, device):
    set_seed(SEED)

    train_ds = AGFVDataset(X[train_idx], y[train_idx])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    model = AGFVNNClassifier(
        input_dim=X.shape[1],
        hidden_dims=tuple(cfg["hidden_dims"]),
        dropout=cfg["dropout"],
    ).to(device)

    counts = np.bincount(y[train_idx], minlength=2).astype(float)
    weights = counts.sum() / (2.0 * np.maximum(counts, 1.0))
    class_weight = torch.tensor(weights, dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    best_state = None
    best_epoch = 0
    best_score = -np.inf
    patience_counter = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=class_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            losses.append(loss.item())

        train_prob = predict_prob(model, X[train_idx], device)
        val_prob = predict_prob(model, X[val_idx], device)

        train_m = compute_metrics(y[train_idx], train_prob)
        val_m = compute_metrics(y[val_idx], val_prob)

        gap = max(0.0, train_m["AUC"] - val_m["AUC"]) + max(
            0.0,
            train_m["Balanced accuracy"] - val_m["Balanced accuracy"],
        )

        # Validation-first, with small penalty for overfitting.
        score = val_m["AUC"] + val_m["Balanced accuracy"] - 0.15 * gap

        history.append(
            {
                "config": cfg["name"],
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "train_auc": train_m["AUC"],
                "train_balanced_accuracy": train_m["Balanced accuracy"],
                "validation_auc": val_m["AUC"],
                "validation_balanced_accuracy": val_m["Balanced accuracy"],
                "generalization_gap": float(gap),
                "selection_score": float(score),
            }
        )

        if score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"{cfg['name']} | Epoch {epoch:03d} | "
                f"Train AUC={train_m['AUC']:.3f}, Train bacc={train_m['Balanced accuracy']:.3f} | "
                f"Val AUC={val_m['AUC']:.3f}, Val bacc={val_m['Balanced accuracy']:.3f} | "
                f"Gap={gap:.3f} | Score={score:.3f}"
            )

        if patience_counter >= PATIENCE:
            print(f"{cfg['name']} early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_state is None:
        raise RuntimeError(f"No best state saved for {cfg['name']}")

    model.load_state_dict(best_state)

    return {
        "config": cfg,
        "model": model,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "history": pd.DataFrame(history),
    }


def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("========== TRAIN FINAL AGFV NN CLASSIFIER ==========")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input AGFV file: {AGFV_FILE}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("====================================================\n")

    if not AGFV_FILE.exists():
        raise FileNotFoundError(AGFV_FILE)

    df = pd.read_csv(AGFV_FILE)
    agfv_cols = get_agfv_cols(df)

    if len(agfv_cols) != 128:
        raise ValueError(f"Expected 128 AGFV columns, found {len(agfv_cols)}")

    if "split" not in df.columns:
        raise ValueError("AGFV combined file must contain split column.")
    if "subject_id" not in df.columns:
        raise ValueError("AGFV combined file must contain subject_id.")
    if df["subject_id"].duplicated().sum() > 0:
        raise ValueError("Duplicate subject_id found.")

    y = get_label(df)
    X_raw = df[agfv_cols].astype(float).values

    train_idx = df.index[df["split"] == "Train"].to_numpy()
    val_idx = df.index[df["split"] == "Validation"].to_numpy()
    test_idx = df.index[df["split"] == "Test"].to_numpy()

    scaler = StandardScaler()
    X = np.zeros_like(X_raw, dtype=np.float32)
    X[train_idx] = scaler.fit_transform(X_raw[train_idx])
    X[val_idx] = scaler.transform(X_raw[val_idx])
    X[test_idx] = scaler.transform(X_raw[test_idx])

    print("Input shape:", df.shape)
    print("AGFV columns:", len(agfv_cols))
    print("\nSplit counts:")
    print(df["split"].value_counts())
    print("\nSplit x diagnosis:")
    print(pd.crosstab(df["split"], df["diagnosis"]))
    print("\nAGFV source x split:")
    if "agfv_source" in df.columns:
        print(pd.crosstab(df["agfv_source"], df["split"]))
    else:
        print("No agfv_source column found.")
    print()

    best_result = None
    best_score = -np.inf
    all_histories = []
    all_config_rows = []

    for cfg in CONFIGS:
        print("\n==================================================")
        print("Training NN config:", cfg)
        print("==================================================")

        result = train_one_config(cfg, X, y, train_idx, val_idx, device)
        model = result["model"]

        train_prob = predict_prob(model, X[train_idx], device)
        val_prob = predict_prob(model, X[val_idx], device)
        test_prob = predict_prob(model, X[test_idx], device)

        train_m = compute_metrics(y[train_idx], train_prob)
        val_m = compute_metrics(y[val_idx], val_prob)
        test_m = compute_metrics(y[test_idx], test_prob)

        all_config_rows.append(
            {
                "Config": cfg["name"],
                "Best epoch": result["best_epoch"],
                "Selection score": result["best_score"],
                "Train AUC": train_m["AUC"],
                "Train balanced accuracy": train_m["Balanced accuracy"],
                "Validation AUC": val_m["AUC"],
                "Validation balanced accuracy": val_m["Balanced accuracy"],
                "Test AUC": test_m["AUC"],
                "Test balanced accuracy": test_m["Balanced accuracy"],
            }
        )

        all_histories.append(result["history"])

        if result["best_score"] > best_score:
            best_score = result["best_score"]
            best_result = {
                "config": cfg,
                "model": model,
                "best_epoch": result["best_epoch"],
                "best_score": result["best_score"],
            }

    if best_result is None:
        raise RuntimeError("No NN classifier selected.")

    best_cfg = best_result["config"]
    best_model = best_result["model"]

    print("\n========== BEST FINAL AGFV NN CLASSIFIER ==========")
    print("Best config:", best_cfg)
    print("Best epoch:", best_result["best_epoch"])
    print("Best validation selection score:", best_result["best_score"])
    print("===================================================\n")

    metrics_rows = []
    prediction_parts = []

    for split_name, idx in [
        ("Train", train_idx),
        ("Validation", val_idx),
        ("Test", test_idx),
    ]:
        prob = predict_prob(best_model, X[idx], device)
        metrics_rows.append(metric_row("Final AGFV NN classifier", split_name, "Overall", y[idx], prob))

        tmp = df.loc[idx, ["subject_id", "diagnosis", "split"]].copy()
        tmp["diagnosis_binary_checked"] = y[idx]

        if "agfv_source" in df.columns:
            tmp["agfv_source"] = df.loc[idx, "agfv_source"].values
        else:
            tmp["agfv_source"] = "unknown"

        tmp["final_agfv_nn_classifier_probability_AD"] = prob
        tmp["final_agfv_nn_classifier_prediction"] = (prob >= 0.5).astype(int)
        prediction_parts.append(tmp)

        if "agfv_source" in df.columns:
            for source_name, sub_df in df.loc[idx].groupby("agfv_source"):
                sub_idx = sub_df.index.to_numpy()
                if len(sub_idx) >= 2:
                    sub_prob = predict_prob(best_model, X[sub_idx], device)
                    metrics_rows.append(
                        metric_row("Final AGFV NN classifier", split_name, source_name, y[sub_idx], sub_prob)
                    )

    metrics_df = pd.DataFrame(metrics_rows)
    all_config_df = pd.DataFrame(all_config_rows)
    history_df = pd.concat(all_histories, ignore_index=True)
    pred_df = pd.concat(prediction_parts, ignore_index=True)

    model_path = OUT_DIR / "final_agfv_nn_classifier_model.pt"
    scaler_path = OUT_DIR / "final_agfv_nn_classifier_scaler.pkl"
    config_path = OUT_DIR / "final_agfv_nn_classifier_config.json"
    metrics_path = OUT_DIR / "final_agfv_nn_classifier_metrics.csv"
    all_config_path = OUT_DIR / "final_agfv_nn_classifier_all_config_metrics.csv"
    history_path = OUT_DIR / "final_agfv_nn_classifier_training_history.csv"
    pred_path = OUT_DIR / "final_agfv_nn_classifier_predictions.csv"
    supp_table_path = TABLE_SUPP_DIR / "Table_S17_Final_AGFV_NN_classifier_metrics.xlsx"

    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "config": best_cfg,
            "best_epoch": int(best_result["best_epoch"]),
            "selection_metric": "Validation AUC + Validation balanced accuracy - 0.15*generalization_gap",
            "selection_score": float(best_result["best_score"]),
        },
        model_path,
    )

    with open(scaler_path, "wb") as f:
        pickle.dump(
            {
                "scaler": scaler,
                "agfv_cols": agfv_cols,
                "seed": SEED,
            },
            f,
        )

    final_config = {
        "model": "Final AGFV NN classifier",
        "input_file": str(AGFV_FILE),
        "n_subjects": int(df.shape[0]),
        "n_agfv_features": int(len(agfv_cols)),
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "best_config": best_cfg,
        "best_epoch": int(best_result["best_epoch"]),
        "selection_metric": "Validation AUC + Validation balanced accuracy - 0.15*generalization_gap",
        "selection_score": float(best_result["best_score"]),
        "feature_note": (
            "Neural-network comparison classifier trained on AGFV features. "
            "Measured-GE subjects use GE-derived AGFV; MRI-only subjects use MRI-derived AGFV. "
            "This is a secondary comparison against the logistic final AGFV classifier."
        ),
    }

    with open(config_path, "w") as f:
        json.dump(final_config, f, indent=2)

    metrics_df.to_csv(metrics_path, index=False)
    all_config_df.to_csv(all_config_path, index=False)
    history_df.to_csv(history_path, index=False)
    pred_df.to_csv(pred_path, index=False)

    with pd.ExcelWriter(supp_table_path) as writer:
        metrics_df.to_excel(writer, sheet_name="Best_NN_metrics", index=False)
        all_config_df.to_excel(writer, sheet_name="NN_config_comparison", index=False)

    print("========== FINAL AGFV NN CLASSIFIER METRICS ==========")
    print(metrics_df.to_string(index=False))
    print("======================================================\n")

    print("Saved:")
    print(model_path)
    print(scaler_path)
    print(config_path)
    print(metrics_path)
    print(all_config_path)
    print(history_path)
    print(pred_path)
    print(supp_table_path)
    print("\nSTATUS: DONE")


if __name__ == "__main__":
    main()
