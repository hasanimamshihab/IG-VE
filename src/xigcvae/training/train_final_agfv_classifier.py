from __future__ import annotations

import json
import pickle
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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


PROJECT_ROOT = Path("/N/project/SingleCell_Image/Ronak/Project_folder")

AGFV_FILE = PROJECT_ROOT / "outputs/aligned_gfv/aligned_gfv_combined_1508_subjects.csv"

OUT_DIR = PROJECT_ROOT / "outputs/final_agfv_classifier"
TABLE_MAIN_DIR = PROJECT_ROOT / "outputs/tables/main"
TABLE_SUPP_DIR = PROJECT_ROOT / "outputs/tables/supplementary"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_MAIN_DIR.mkdir(parents=True, exist_ok=True)
TABLE_SUPP_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42

C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


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

    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, prob)
    else:
        auc = np.nan

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


def eval_model(model, scaler, X, y, split_name, subgroup_name, n_total=None):
    Xs = scaler.transform(X)
    prob = model.predict_proba(Xs)[:, 1]
    m = compute_metrics(y, prob)

    row = {
        "Model": "Final AGFV logistic classifier",
        "Split": split_name,
        "Subgroup": subgroup_name,
        "Subjects, n": int(len(y)),
        "AD, n": int(np.sum(np.asarray(y) == 1)),
        "CN, n": int(np.sum(np.asarray(y) == 0)),
    }
    row.update(m)
    return row, prob


def main():
    set_seed(SEED)

    print("========== TRAIN FINAL AGFV CLASSIFIER ==========")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input AGFV file: {AGFV_FILE}")
    print(f"Output dir: {OUT_DIR}")
    print("=================================================\n")

    if not AGFV_FILE.exists():
        raise FileNotFoundError(AGFV_FILE)

    df = pd.read_csv(AGFV_FILE)

    agfv_cols = get_agfv_cols(df)
    if len(agfv_cols) != 128:
        raise ValueError(f"Expected 128 AGFV columns, found {len(agfv_cols)}")

    if "split" not in df.columns:
        raise ValueError("AGFV combined file must contain a split column.")

    if "subject_id" not in df.columns:
        raise ValueError("AGFV combined file must contain subject_id.")

    if df["subject_id"].duplicated().sum() > 0:
        dup = df.loc[df["subject_id"].duplicated(), "subject_id"].head().tolist()
        raise ValueError(f"Duplicate subject_id values found. Example: {dup}")

    y = get_label(df)
    X = df[agfv_cols].astype(float).values

    if np.isnan(X).sum() > 0:
        raise ValueError("Missing values found in AGFV features.")

    df["diagnosis_binary_checked"] = y

    valid_splits = {"Train", "Validation", "Test"}
    observed_splits = set(df["split"].astype(str).unique())
    bad_splits = observed_splits - valid_splits
    if bad_splits:
        raise ValueError(f"Unexpected split values: {bad_splits}")

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

    train_idx = df.index[df["split"] == "Train"].to_numpy()
    val_idx = df.index[df["split"] == "Validation"].to_numpy()
    test_idx = df.index[df["split"] == "Test"].to_numpy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X[train_idx])

    all_model_rows = []
    best_model = None
    best_c = None
    best_score = -np.inf

    for C in C_GRID:
        model = LogisticRegression(
            C=C,
            penalty="l2",
            solver="lbfgs",
            class_weight="balanced",
            max_iter=5000,
            random_state=SEED,
        )
        model.fit(X_train_scaled, y[train_idx])

        train_row, _ = eval_model(model, scaler, X[train_idx], y[train_idx], "Train", "Overall")
        val_row, _ = eval_model(model, scaler, X[val_idx], y[val_idx], "Validation", "Overall")

        # validation-first selection
        # AUC + balanced accuracy, with tiny preference for stronger regularization if tied.
        score = val_row["AUC"] + val_row["Balanced accuracy"] - 1e-5 * C

        all_model_rows.append(
            {
                "C": C,
                "Train AUC": train_row["AUC"],
                "Train balanced accuracy": train_row["Balanced accuracy"],
                "Validation AUC": val_row["AUC"],
                "Validation balanced accuracy": val_row["Balanced accuracy"],
                "Validation selection score": float(score),
            }
        )

        print(
            f"C={C:<7} | "
            f"Train AUC={train_row['AUC']:.3f}, Train bacc={train_row['Balanced accuracy']:.3f} | "
            f"Val AUC={val_row['AUC']:.3f}, Val bacc={val_row['Balanced accuracy']:.3f} | "
            f"Score={score:.3f}"
        )

        if score > best_score:
            best_score = score
            best_model = model
            best_c = C

    if best_model is None:
        raise RuntimeError("No classifier selected.")

    print("\n========== BEST FINAL AGFV CLASSIFIER ==========")
    print(f"Best C: {best_c}")
    print(f"Best validation selection score: {best_score}")
    print("================================================\n")

    metrics_rows = []
    prediction_parts = []

    for split_name, idx in [
        ("Train", train_idx),
        ("Validation", val_idx),
        ("Test", test_idx),
    ]:
        row, prob = eval_model(best_model, scaler, X[idx], y[idx], split_name, "Overall")
        metrics_rows.append(row)

        tmp = df.loc[idx, ["subject_id", "diagnosis", "diagnosis_binary_checked", "split"]].copy()
        if "agfv_source" in df.columns:
            tmp["agfv_source"] = df.loc[idx, "agfv_source"].values
        else:
            tmp["agfv_source"] = "unknown"

        tmp["final_agfv_classifier_probability_AD"] = prob
        tmp["final_agfv_classifier_prediction"] = (prob >= 0.5).astype(int)
        prediction_parts.append(tmp)

        if "agfv_source" in df.columns:
            for source_name, sub_df in df.loc[idx].groupby("agfv_source"):
                sub_idx = sub_df.index.to_numpy()
                if len(sub_idx) >= 2:
                    sub_row, _ = eval_model(
                        best_model,
                        scaler,
                        X[sub_idx],
                        y[sub_idx],
                        split_name,
                        source_name,
                    )
                    metrics_rows.append(sub_row)

    metrics_df = pd.DataFrame(metrics_rows)
    all_model_df = pd.DataFrame(all_model_rows)
    pred_df = pd.concat(prediction_parts, ignore_index=True)

    model_path = OUT_DIR / "final_agfv_classifier_model.pkl"
    scaler_path = OUT_DIR / "final_agfv_classifier_scaler.pkl"
    config_path = OUT_DIR / "final_agfv_classifier_config.json"
    metrics_path = OUT_DIR / "final_agfv_classifier_metrics.csv"
    all_model_metrics_path = OUT_DIR / "final_agfv_classifier_all_model_metrics.csv"
    pred_path = OUT_DIR / "final_agfv_classifier_predictions.csv"

    main_table_path = TABLE_MAIN_DIR / "Table_2_Final_AGFV_classifier_performance.xlsx"
    supp_table_path = TABLE_SUPP_DIR / "Table_S16_Final_AGFV_classifier_metrics.xlsx"

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    with open(scaler_path, "wb") as f:
        pickle.dump(
            {
                "scaler": scaler,
                "agfv_cols": agfv_cols,
                "seed": SEED,
            },
            f,
        )

    config = {
        "model": "Final AGFV logistic classifier",
        "input_file": str(AGFV_FILE),
        "n_subjects": int(df.shape[0]),
        "n_agfv_features": int(len(agfv_cols)),
        "seed": SEED,
        "splits": {
            "Train": int(len(train_idx)),
            "Validation": int(len(val_idx)),
            "Test": int(len(test_idx)),
        },
        "best_C": float(best_c),
        "selection_metric": "Validation AUC + Validation balanced accuracy - 1e-5*C",
        "selection_score": float(best_score),
        "feature_note": (
            "Classifier trained on AGFV features for all subjects. "
            "Measured-GE subjects use GE-derived AGFV; MRI-only subjects use MRI-derived AGFV. "
            "Original GFV and MFV are not mixed directly in the classifier."
        ),
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    metrics_df.to_csv(metrics_path, index=False)
    all_model_df.to_csv(all_model_metrics_path, index=False)
    pred_df.to_csv(pred_path, index=False)

    with pd.ExcelWriter(main_table_path) as writer:
        metrics_df.to_excel(writer, sheet_name="Final_AGFV_classifier", index=False)

    with pd.ExcelWriter(supp_table_path) as writer:
        metrics_df.to_excel(writer, sheet_name="Final_metrics", index=False)
        all_model_df.to_excel(writer, sheet_name="C_grid_validation", index=False)

    print("========== FINAL AGFV CLASSIFIER METRICS ==========")
    print(metrics_df.to_string(index=False))
    print("===================================================\n")

    print("Saved:")
    print(model_path)
    print(scaler_path)
    print(config_path)
    print(metrics_path)
    print(all_model_metrics_path)
    print(pred_path)
    print(main_table_path)
    print(supp_table_path)
    print("\nSTATUS: DONE")


if __name__ == "__main__":
    main()
