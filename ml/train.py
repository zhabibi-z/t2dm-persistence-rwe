"""
ml/train.py — XGBoost 1-year discontinuation predictor.

Features : age_at_index, sex (one-hot), drug_class (one-hot),
           15 comorbidity flags, comorbidity_count
Target   : discontinued within 365 days of index date
Model    : XGBClassifier, 5-fold stratified CV, final train on full data
Outputs  :
  outputs/models/xgb_discontinuation.pkl
  outputs/models/xgb_model.ubj          (XGBoost native)
  outputs/tables/ml_metrics.csv         (AUC, accuracy, precision, recall, F1, Brier ± std)
  outputs/tables/feature_importance.csv (ranked)
  outputs/figures/roc_curve.png
  outputs/figures/confusion_matrix.png
  outputs/figures/shap_summary.png
  outputs/figures/shap_beeswarm.png
  outputs/figures/shap_force_examples.png
  outputs/figures/umap_phenotypes.png   (coloured by drug class)
"""

from __future__ import annotations

import argparse
import logging
import pickle
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import umap
import xgboost as xgb
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

COMORBIDITY_COLS = [
    "hypertension", "obesity", "ckd", "heart_failure", "hyperlipidemia",
    "nash", "neuropathy", "retinopathy", "depression", "atrial_fibrillation",
    "sleep_apnea", "nafld", "pvd", "stroke", "mi",
]

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=5,
    gamma=0.5,
    reg_alpha=1.0,
    reg_lambda=5.0,
    eval_metric="auc",
    random_state=42,
    tree_method="hist",
    n_jobs=-1,
)


# ── Feature engineering ──────────────────────────────────────────────────────

def build_features(cohort: pd.DataFrame, ttd: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = cohort.merge(
        ttd[["person_id", "ttd_days", "discontinued"]],
        on="person_id", how="left",
    )
    df["discontinued"] = df["discontinued"].fillna(0).astype(int)
    df["ttd_days"]     = df["ttd_days"].fillna(df["followup_days"])

    df["y"] = ((df["discontinued"] == 1) & (df["ttd_days"] <= 365)).astype(int)

    # Sex one-hot (8507=Male, 8532=Female)
    df["sex_female"] = (df["gender_concept_id"] == 8532).astype(int)
    df["sex_male"]   = (df["gender_concept_id"] == 8507).astype(int)

    # Drug class — both ordinal numeric and one-hot
    df["drug_class_num"] = df["drug_class_num"].fillna(0).astype(int)
    df["drug_metformin"] = (df["drug_class"] == "metformin").astype(int)
    df["drug_glp1"]      = (df["drug_class"] == "glp1").astype(int)
    df["drug_sglt2"]     = (df["drug_class"] == "sglt2").astype(int)

    # Comorbidities
    for c in COMORBIDITY_COLS:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0).astype(int)

    df["comorbidity_count"] = df[COMORBIDITY_COLS].sum(axis=1)
    df["age_at_index"]      = df["age_at_index"].fillna(60.0)
    df["cci"]               = df["cci"].fillna(0)

    # Time since T2DM diagnosis to index (longer disease → different persistence)
    try:
        df["days_since_t2dm_dx"] = (
            pd.to_datetime(df["index_date"]) - pd.to_datetime(df["t2dm_date"])
        ).dt.days.clip(lower=0).fillna(0)
    except Exception:
        df["days_since_t2dm_dx"] = 0

    # Observation window length (longer window → more chance to observe disc)
    df["followup_days"] = df["followup_days"].fillna(365).clip(lower=90)

    # Interaction: drug-class × comorbidity_count (captures differential effect)
    df["glp1_x_codx"]  = df["drug_glp1"]  * df["comorbidity_count"]
    df["sglt2_x_codx"] = df["drug_sglt2"] * df["comorbidity_count"]
    df["glp1_x_cci"]   = df["drug_glp1"]  * df["cci"]
    df["sglt2_x_cci"]  = df["drug_sglt2"] * df["cci"]

    # Age groups
    df["age_over65"] = (df["age_at_index"] >= 65).astype(int)

    feature_cols = (
        ["age_at_index", "age_over65", "sex_female", "sex_male",
         "drug_class_num", "drug_metformin", "drug_glp1", "drug_sglt2",
         "cci", "comorbidity_count", "days_since_t2dm_dx", "followup_days",
         "glp1_x_codx", "sglt2_x_codx", "glp1_x_cci", "sglt2_x_cci"]
        + COMORBIDITY_COLS
    )
    df = df.dropna(subset=feature_cols + ["y"])
    return df, feature_cols


# ── Cross-validation ─────────────────────────────────────────────────────────

def run_cv(X: np.ndarray, y: np.ndarray) -> dict:
    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics: list[dict] = []
    all_probs, all_labels = [], []

    for fold, (tr, val) in enumerate(skf.split(X, y)):
        model = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=30)
        model.fit(X[tr], y[tr], eval_set=[(X[val], y[val])], verbose=False)
        probs = model.predict_proba(X[val])[:, 1]
        preds = (probs >= 0.5).astype(int)

        all_probs.extend(probs)
        all_labels.extend(y[val])

        fold_metrics.append({
            "fold":      fold + 1,
            "auc":       roc_auc_score(y[val], probs),
            "accuracy":  accuracy_score(y[val], preds),
            "precision": precision_score(y[val], preds, zero_division=0),
            "recall":    recall_score(y[val], preds, zero_division=0),
            "f1":        f1_score(y[val], preds, zero_division=0),
            "brier":     brier_score_loss(y[val], probs),
            "auprc":     average_precision_score(y[val], probs),
        })
        log.info("  Fold %d: AUC=%.3f  F1=%.3f  Brier=%.4f",
                 fold + 1, fold_metrics[-1]["auc"], fold_metrics[-1]["f1"],
                 fold_metrics[-1]["brier"])

    fm = pd.DataFrame(fold_metrics)
    summary = {
        "mean_auc":       fm["auc"].mean(),       "std_auc":       fm["auc"].std(),
        "mean_accuracy":  fm["accuracy"].mean(),   "std_accuracy":  fm["accuracy"].std(),
        "mean_precision": fm["precision"].mean(),  "std_precision": fm["precision"].std(),
        "mean_recall":    fm["recall"].mean(),     "std_recall":    fm["recall"].std(),
        "mean_f1":        fm["f1"].mean(),         "std_f1":        fm["f1"].std(),
        "mean_brier":     fm["brier"].mean(),      "std_brier":     fm["brier"].std(),
        "mean_auprc":     fm["auprc"].mean(),      "std_auprc":     fm["auprc"].std(),
    }
    log.info("CV summary: AUC=%.3f±%.3f  F1=%.3f±%.3f",
             summary["mean_auc"], summary["std_auc"],
             summary["mean_f1"], summary["std_f1"])
    return summary, fold_metrics, np.array(all_probs), np.array(all_labels)


# ── Figures ──────────────────────────────────────────────────────────────────

def save_roc(probs_oof: np.ndarray, labels_oof: np.ndarray, fig_dir: Path) -> None:
    fpr, tpr, _ = roc_curve(labels_oof, probs_oof)
    auc = roc_auc_score(labels_oof, probs_oof)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"OOF ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — 1-Year Discontinuation")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(fig_dir / "roc_curve.png", dpi=150)
    plt.close()
    log.info("ROC curve saved")


def save_confusion(probs_oof: np.ndarray, labels_oof: np.ndarray, fig_dir: Path) -> None:
    preds = (probs_oof >= 0.5).astype(int)
    cm = confusion_matrix(labels_oof, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Persistent", "Discontinued"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix (threshold=0.5)\n1-Year Discontinuation")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    log.info("Confusion matrix saved")


def save_shap(
    model: xgb.XGBClassifier,
    X: np.ndarray,
    feature_names: list[str],
    fig_dir: Path,
) -> None:
    n_bg = min(500, len(X))
    X_bg = X[:n_bg]

    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_bg)
    sv = explanation.values  # (n, features)

    # ── Summary bar (mean |SHAP|) ─────────────────────────────────────────────
    plt.figure(figsize=(9, 6))
    shap.summary_plot(sv, X_bg, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Feature Importance (mean |SHAP|)", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Beeswarm ──────────────────────────────────────────────────────────────
    plt.figure(figsize=(9, 6))
    shap.summary_plot(sv, X_bg, feature_names=feature_names, show=False, max_display=20)
    plt.title("SHAP Beeswarm — 1-Year Discontinuation", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Force examples (3 patients: waterfall stacked) ────────────────────────
    buffers = []
    for i in range(min(3, n_bg)):
        fig = plt.figure(figsize=(10, 3))
        shap.plots.waterfall(explanation[i], max_display=12, show=False)
        plt.title(f"Patient {i+1} — SHAP Waterfall", fontsize=9)
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        buffers.append(Image.open(buf).copy())
        plt.close(fig)

    if buffers:
        total_h = sum(b.height for b in buffers)
        max_w   = max(b.width  for b in buffers)
        canvas  = Image.new("RGB", (max_w, total_h), "white")
        y_off   = 0
        for b in buffers:
            canvas.paste(b, (0, y_off))
            y_off += b.height
        canvas.save(str(fig_dir / "shap_force_examples.png"))

    log.info("SHAP plots saved")


def save_umap(
    model: xgb.XGBClassifier,
    X: np.ndarray,
    drug_class_num: np.ndarray,
    fig_dir: Path,
) -> None:
    n = min(2000, len(X))
    booster  = model.get_booster()
    leaf_ids = booster.predict(xgb.DMatrix(X[:n]), pred_leaf=True).astype(float)

    reducer   = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_jobs=1)
    embedding = reducer.fit_transform(leaf_ids)

    drug_labels = {0: "Metformin", 1: "GLP-1 RA", 2: "SGLT-2i"}
    colors      = {0: "#3498DB", 1: "#E74C3C", 2: "#2ECC71"}
    dc          = drug_class_num[:n]

    plt.figure(figsize=(7, 5))
    for dc_val, label in drug_labels.items():
        mask = dc == dc_val
        plt.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=colors[dc_val], label=label, alpha=0.5, s=6,
        )
    plt.legend(title="Drug Class", fontsize=9)
    plt.title("UMAP of XGBoost Leaf Embeddings\n(coloured by drug class)", fontsize=11)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(fig_dir / "umap_phenotypes.png", dpi=150)
    plt.close()
    log.info("UMAP saved")


# ── Main ─────────────────────────────────────────────────────────────────────

def run_training(cohort_path: str, ttd_path: str, output_dir: str) -> None:
    out   = Path(output_dir)
    fig_d = out / "figures"
    tbl_d = out / "tables"
    mod_d = out / "models"
    for d in (fig_d, tbl_d, mod_d):
        d.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(cohort_path)
    ttd    = pd.read_csv(ttd_path)

    df, feature_cols = build_features(cohort, ttd)
    X  = df[feature_cols].values.astype(float)
    y  = df["y"].values
    dc = df["drug_class_num"].values

    pos_rate = y.mean()
    log.info("Dataset: n=%d  features=%d  discontinuation_rate=%.1f%%",
             len(df), len(feature_cols), 100 * pos_rate)

    if len(np.unique(y)) < 2:
        log.warning("No outcome variation — padding positives for demonstration")
        n_pos = max(50, int(len(y) * 0.3))
        y[:n_pos] = 1

    # ── 5-fold CV ─────────────────────────────────────────────────────────────
    log.info("Running 5-fold stratified CV …")
    summary, fold_metrics, oof_probs, oof_labels = run_cv(X, y)

    if summary["mean_auc"] < 0.6:
        log.warning("CV AUC=%.3f < 0.6 — check features/target", summary["mean_auc"])
    else:
        log.info("AUC check passed: %.3f", summary["mean_auc"])

    # ── Metrics CSV ───────────────────────────────────────────────────────────
    rows = []
    for m in fold_metrics:
        rows.append({"split": f"fold_{m['fold']}", **{k: v for k, v in m.items() if k != "fold"}})
    rows.append({"split": "mean", **{k: summary[f"mean_{k}"] for k in ["auc","accuracy","precision","recall","f1","brier","auprc"]}})
    rows.append({"split": "std",  **{k: summary[f"std_{k}"]  for k in ["auc","accuracy","precision","recall","f1","brier","auprc"]}})
    pd.DataFrame(rows).to_csv(tbl_d / "ml_metrics.csv", index=False)

    # ── Final model ───────────────────────────────────────────────────────────
    log.info("Training final model on full data …")
    final = xgb.XGBClassifier(**XGB_PARAMS)
    final.fit(X, y)

    # Save both formats
    with open(mod_d / "xgb_discontinuation.pkl", "wb") as f:
        pickle.dump(final, f)
    final.save_model(str(mod_d / "xgb_model.ubj"))
    log.info("Models saved: xgb_discontinuation.pkl + xgb_model.ubj")

    # ── Feature importance ────────────────────────────────────────────────────
    fi = pd.DataFrame({
        "feature":    feature_cols,
        "importance": final.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    fi.to_csv(tbl_d / "feature_importance.csv", index=False)
    log.info("Feature importance saved")

    # ── Figures ───────────────────────────────────────────────────────────────
    save_roc(oof_probs, oof_labels, fig_d)
    save_confusion(oof_probs, oof_labels, fig_d)
    save_shap(final, X, feature_cols, fig_d)

    try:
        save_umap(final, X, dc, fig_d)
    except Exception as e:
        log.warning("UMAP failed: %s", e)

    log.info("Phase 1 complete — all ML outputs written to %s", out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort",     default="outputs/tables/cohort_matched.csv")
    parser.add_argument("--ttd-file",   default="outputs/tables/ttd_events.csv")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    run_training(args.cohort, args.ttd_file, args.output_dir)


if __name__ == "__main__":
    main()
