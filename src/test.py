import os, json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from typing import List

plt.switch_backend("Agg")  

CONFIG = {
    "MODEL_PATH": "occupancy_fnn_model.h5",
    "SCALER_PATH": "scaler.pkl",
    "CSV_FILE": "Sensor_Data_Engineered.csv",           
    "REPORT_DIR": "reports",
    "BATCH_SIZE": 1024,
    "THRESH": 0.719,                       
    "FEATURE_COLS": [
        "Temperature",
        "Light",
        "PIR",
        "Light_mean_3",
        "Temp_mean_3",
        "Light_diff_3",
        "Temp_diff_3",
        "hour_sin"                          
    ],
    "TIME_COL": "date"
}

os.makedirs(CONFIG["REPORT_DIR"], exist_ok=True)

def log_section(title: str):
    bar = "=" * 74
    print(f"\n{bar}\n{title}\n{bar}")

def load_model_and_scaler():
    log_section("LOAD MODEL AND SCALER")
    if not os.path.exists(CONFIG["MODEL_PATH"]):
        raise FileNotFoundError(f"Model missing: {CONFIG['MODEL_PATH']}")
    model = tf.keras.models.load_model(CONFIG["MODEL_PATH"])
    print(f"✓ Model: {CONFIG['MODEL_PATH']} | input={model.input_shape} output={model.output_shape}")

    scaler = None
    if os.path.exists(CONFIG["SCALER_PATH"]):
        scaler = joblib.load(CONFIG["SCALER_PATH"])
        print(f"✓ Scaler: {CONFIG['SCALER_PATH']}")
    else:
        print("⚠ scaler.pkl not found. Proceeding without scaling.")
    return model, scaler

def load_sensor_data():
    log_section("LOAD SENSOR DATA")
    if not os.path.exists(CONFIG["CSV_FILE"]):
        raise FileNotFoundError(f"CSV missing: {CONFIG['CSV_FILE']}")
    df = pd.read_csv(CONFIG["CSV_FILE"])
    print(f"✓ CSV: {CONFIG['CSV_FILE']} | rows={len(df):,} cols={len(df.columns)}")
    if CONFIG["TIME_COL"] in df.columns:
        try:
            df[CONFIG["TIME_COL"]] = pd.to_datetime(df[CONFIG["TIME_COL"]], errors="coerce")
        except Exception:
            pass
    return df

def check_feature_columns(df: pd.DataFrame) -> List[str]:
    log_section("VALIDATE FEATURES")
    missing = [c for c in CONFIG["FEATURE_COLS"] if c not in df.columns]
    if missing:
        if "hour_sin" in missing and "hour_cos" in df.columns:
            print("ℹ 'hour_sin' missing but 'hour_cos' present; switching to hour_cos.")
            cols = [("hour_cos" if c == "hour_sin" else c) for c in CONFIG["FEATURE_COLS"]]
        else:
            raise ValueError(f"Missing required feature columns: {missing}")
    else:
        cols = CONFIG["FEATURE_COLS"]

    print(f"✓ Using features: {cols}")
    return cols

def extract_and_scale(df: pd.DataFrame, cols: List[str], scaler, expected_width: int):
    X_df = df[cols].copy()
    X = X_df.apply(pd.to_numeric, errors="coerce").fillna(method="ffill").fillna(method="bfill").fillna(0.0).astype(np.float32).values
    if X.shape[1] != expected_width:
        raise ValueError(f"Feature width mismatch: model expects {expected_width}, found {X.shape[1]} from columns {cols}")
    if scaler is not None:
        try:
            X = scaler.transform(X)
            print("✓ Applied StandardScaler to features")
        except Exception as e:
            print(f"⚠ Scaling failed: {e}. Continuing with raw features.")
    print(f"Feature matrix shape: {X.shape}")
    return X

def predict_probs(model, X: np.ndarray) -> np.ndarray:
    log_section("RUN INFERENCE")
    y = model.predict(X, batch_size=CONFIG["BATCH_SIZE"], verbose=0)
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 2:
        probs = y[:, 1]
    else:
        probs = y.squeeze()
    probs = probs.astype(np.float32)
    print(f"✓ Inference complete | N={len(probs):,} | min={probs.min():.4f} max={probs.max():.4f} mean={probs.mean():.4f}")
    return probs

def summarize(probs: np.ndarray):
    y_pred = (probs >= CONFIG["THRESH"]).astype(int)
    unocc = int((y_pred == 0).sum())
    occ = int((y_pred == 1).sum())
    rate = round(occ * 100.0 / len(y_pred), 2)
    stats = {
        "timestamp": datetime.now().isoformat(),
        "total_samples": int(len(probs)),
        "predicted_unoccupied": unocc,
        "predicted_occupied": occ,
        "occupancy_rate_percent": rate,
        "avg_probability": round(float(probs.mean()), 4),
        "probability_min": round(float(probs.min()), 4),
        "probability_max": round(float(probs.max()), 4),
        "probability_std": round(float(probs.std()), 4),
        "threshold": CONFIG["THRESH"],
        "model_path": CONFIG["MODEL_PATH"],
        "csv_file": CONFIG["CSV_FILE"],
    }
    log_section("SUMMARY")
    print(json.dumps(stats, indent=2))
    return y_pred, stats

def plot_figures(probs: np.ndarray, y_pred: np.ndarray):
    log_section("PLOTS")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(probs, bins=80, color="#3b82f6", edgecolor="black", alpha=0.8)
    ax.axvline(CONFIG["THRESH"], color="#ef4444", ls="--", lw=2, label=f"Threshold {CONFIG['THRESH']:.3f}")
    ax.set_title("Prediction Probability Distribution")
    ax.set_xlabel("Predicted probability (Occupied)")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(CONFIG["REPORT_DIR"], "01_probability_hist.png"), dpi=160)
    plt.close(fig)

    unocc = int((y_pred == 0).sum()); occ = int((y_pred == 1).sum())
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie([unocc, occ],
           labels=[f"Unoccupied\n{unocc:,} ({unocc*100/len(y_pred):.1f}%)",
                   f"Occupied\n{occ:,} ({occ*100/len(y_pred):.1f}%)"],
           explode=(0.05, 0.05),
           colors=["#60a5fa", "#ef4444"],
           autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10, "fontweight": "bold"})
    ax.set_title("Prediction Distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(CONFIG["REPORT_DIR"], "02_prediction_pie.png"), dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(probs, lw=0.9, color="#1f77b4")
    ax.axhline(CONFIG["THRESH"], color="#ef4444", ls="--", lw=2)
    ax.fill_between(range(len(probs)), CONFIG["THRESH"], 1.0, color="#ef4444", alpha=0.08, label="Occupied region")
    ax.fill_between(range(len(probs)), 0.0, CONFIG["THRESH"], color="#60a5fa", alpha=0.08, label="Unoccupied region")
    ax.set_ylim(0, 1)
    ax.set_title("Probability Over Samples")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("P(Occupied)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(CONFIG["REPORT_DIR"], "03_probability_timeline.png"), dpi=160)
    plt.close(fig)

def save_outputs(df: pd.DataFrame, probs: np.ndarray, y_pred: np.ndarray, feature_cols: List[str], stats: dict):
    log_section("SAVE OUTPUTS")
    out = pd.DataFrame()
    if CONFIG["TIME_COL"] in df.columns:
        out[CONFIG["TIME_COL"]] = df[CONFIG["TIME_COL"]]
    for c in feature_cols:
        out[c] = df[c]
    out["occupancy_probability"] = probs
    out["occupancy_prediction"] = y_pred
    out["occupancy_label"] = out["occupancy_prediction"].map({0: "Unoccupied", 1: "Occupied"})

    out_path = os.path.join(CONFIG["REPORT_DIR"], "predictions.csv")
    out.to_csv(out_path, index=False)
    print(f"✓ predictions.csv -> {out_path}")

    with open(os.path.join(CONFIG["REPORT_DIR"], "statistics.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("✓ statistics.json saved")

    print("\nSample rows:")
    print(out.head(10).to_string(index=False))

def main():
    try:
        log_section("AUTOMATED OCCUPANCY PREDICTION")
        model, scaler = load_model_and_scaler()
        df = load_sensor_data()

        feature_cols = check_feature_columns(df)

        expected_width = model.input_shape[-1] if isinstance(model.input_shape, tuple) else int(model.input_shape[1])
        X = extract_and_scale(df, feature_cols, scaler, expected_width)

        probs = predict_probs(model, X)
        y_pred, stats = summarize(probs)

        plot_figures(probs, y_pred)
        save_outputs(df, probs, y_pred, feature_cols, stats)

        log_section("DONE")
        print(f"Reports directory: {os.path.abspath(CONFIG['REPORT_DIR'])}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
