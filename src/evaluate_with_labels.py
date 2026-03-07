import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, confusion_matrix
)
import tensorflow as tf
import joblib

plt.switch_backend("Agg")

CONFIG = {
    "MODEL_PATH": "occupancy_fnn_model.h5",
    "SCALER_PATH": "scaler.pkl",
    "CSV_FILE": "Sensor_Data_Engineered.csv",
    "LABEL_COL": None,   
    "FEATURE_COLS": [
        "Temperature","Light","PIR",
        "Light_mean_3","Temp_mean_3","Light_diff_3","Temp_diff_3",
        "hour_sin"  
    ],
    "REPORT_DIR": "reports_labeled",
    "THRESH": 0.719,
    "BATCH_SIZE": 1024
}

os.makedirs(CONFIG["REPORT_DIR"], exist_ok=True)

def load_model_scaler():
    if not os.path.exists(CONFIG["MODEL_PATH"]):
        raise FileNotFoundError(CONFIG["MODEL_PATH"])
    model = tf.keras.models.load_model(CONFIG["MODEL_PATH"])
    scaler = joblib.load(CONFIG["SCALER_PATH"]) if os.path.exists(CONFIG["SCALER_PATH"]) else None
    print(f"Model input shape: {model.input_shape}, output shape: {model.output_shape}")
    return model, scaler

def load_data():
    if not os.path.exists(CONFIG["CSV_FILE"]):
        raise FileNotFoundError(CONFIG["CSV_FILE"])
    df = pd.read_csv(CONFIG["CSV_FILE"])
    print("CSV columns:", list(df.columns))
    if ("hour_sin" in CONFIG["FEATURE_COLS"] or "hour_cos" in CONFIG["FEATURE_COLS"]) and "date" in df.columns:
        if "hour_sin" not in df.columns or "hour_cos" not in df.columns:
            t = pd.to_datetime(df["date"], errors="coerce")
            h = t.dt.hour + t.dt.minute/60.0
            df["hour_sin"] = np.sin(2*np.pi*h/24.0)
            df["hour_cos"] = np.cos(2*np.pi*h/24.0)
            print("Computed hour_sin/hour_cos from date.")
    return df

def get_label_series(df: pd.DataFrame) -> pd.Series:
    if CONFIG["LABEL_COL"] and CONFIG["LABEL_COL"] in df.columns:
        y = df[CONFIG["LABEL_COL"]]
        print(f"Using label column: {CONFIG['LABEL_COL']}")
    else:
        candidates = [c for c in df.columns if c.lower() in ["label","target","occupancy","is_occupied","y"]]
        if len(candidates) >= 1:
            chosen = candidates[0]
            print(f"Auto-detected label column: {chosen}")
            y = df[chosen]
        else:
            if "PIR" not in df.columns:
                raise ValueError("No label column found and PIR not present to derive labels.")
            print("No label column found; deriving label from PIR > 0.")
            y = (pd.to_numeric(df["PIR"], errors="coerce").fillna(0) > 0).astype(int)
    y = y.map({True:1, False:0}).fillna(y).astype(float)
    uniq = pd.unique(y.dropna())
    if not set(np.unique(uniq)).issubset({0,1}):
        y = (y >= 0.5).astype(int)
    return y.astype(int)

def prepare_X(df, scaler, model):
    missing = [c for c in CONFIG["FEATURE_COLS"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    X = df[CONFIG["FEATURE_COLS"]].apply(pd.to_numeric, errors="coerce")\
        .fillna(method="ffill").fillna(method="bfill").fillna(0.0)\
        .astype(np.float32).values
    exp = model.input_shape[-1]
    if X.shape[1] != exp:
        raise ValueError(f"Width mismatch: model expects {exp}, got {X.shape[1]}")
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception as e:
            print(f"Scaling failed, using raw features: {e}")
    return X

def predict_probs(model, X):
    y = model.predict(X, batch_size=CONFIG["BATCH_SIZE"], verbose=0)
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 2:
        return y[:,1].astype(np.float32)
    return y.squeeze().astype(np.float32)

def metrics_and_plots(y_true, probs):
    thresh = CONFIG["THRESH"]
    y_pred = (probs >= thresh).astype(int)
    mets = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "avg_precision": float(average_precision_score(y_true, probs)),
        "threshold": float(thresh)
    }
    print(json.dumps(mets, indent=2))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    im = ax.imshow(cm, cmap="Blues")
    for (i,j),v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=12)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred 0","Pred 1"]); ax.set_yticklabels(["True 0","True 1"])
    ax.set_title(f"Confusion Matrix (thr={thresh:.3f})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(CONFIG["REPORT_DIR"], "cm.png"), dpi=160); plt.close(fig)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, probs)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(fpr, tpr, label=f"AUC={mets['roc_auc']:.3f}")
    ax.plot([0,1],[0,1],"--", color="gray")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(CONFIG["REPORT_DIR"], "roc.png"), dpi=160); plt.close(fig)

    # PR
    prec, rec, thr = precision_recall_curve(y_true, probs)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(rec, prec, label=f"AP={mets['avg_precision']:.3f}")
    if len(thr)>0:
        k = np.argmin(np.abs(thr - thresh))
        ax.scatter(rec[k], prec[k], color="red", zorder=3, label=f"thr≈{thr[k]:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision–Recall"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(CONFIG["REPORT_DIR"], "pr.png"), dpi=160); plt.close(fig)

    # Threshold sweep
    ts = np.linspace(0.05, 0.95, 19)
    P,R,F1 = [],[],[]
    for t in ts:
        yp = (probs >= t).astype(int)
        P.append(precision_score(y_true, yp, zero_division=0))
        R.append(recall_score(y_true, yp, zero_division=0))
        F1.append(f1_score(y_true, yp, zero_division=0))
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(ts, P, label="Precision"); ax.plot(ts, R, label="Recall"); ax.plot(ts, F1, label="F1")
    ax.axvline(thresh, color="red", ls="--", label=f"thr={thresh:.3f}")
    ax.set_ylim(0,1); ax.set_xlabel("Threshold"); ax.set_ylabel("Score"); ax.set_title("Threshold Sweep"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(CONFIG["REPORT_DIR"], "threshold_sweep.png"), dpi=160); plt.close(fig)

    # Reliability (calibration)
    def reliability(y, p, bins=12):
        bin_ids = np.clip((p*bins).astype(int), 0, bins-1)
        conf, acc = [], []
        for b in range(bins):
            idx = bin_ids == b
            if idx.sum() == 0: continue
            conf.append(p[idx].mean()); acc.append(y[idx].mean())
        return np.array(conf), np.array(acc)

    conf, acc = reliability(y_true, probs)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot([0,1],[0,1], "--", color="gray", label="Perfect")
    ax.scatter(conf, acc, color="#1f77b4")
    ax.set_xlabel("Predicted prob (bin mean)"); ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability Diagram"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(CONFIG["REPORT_DIR"], "reliability.png"), dpi=160); plt.close(fig)

    with open(os.path.join(CONFIG["REPORT_DIR"], "metrics.json"), "w") as f:
        json.dump(mets, f, indent=2)

    return mets

def main():
    model, scaler = load_model_scaler()
    df = load_data()

    y_true = get_label_series(df).values

    X = prepare_X(df, scaler, model)

    probs = predict_probs(model, X)

    metrics_and_plots(y_true, probs)

    print(f"\nReports written to: {os.path.abspath(CONFIG['REPORT_DIR'])}")

if __name__ == "__main__":
    main()
