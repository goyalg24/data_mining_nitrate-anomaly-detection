"""
Sliding-window percentile thresholding for Nitrate (NO3N) anomaly detection.

Method
------
* Fixed window size W, overlapping windows, step size = 1.
* Window 1 covers x[0 : W]           (indices 0 .. W-1)
  Window i covers x[i-1 : i-1+W]     (indices i-1 .. i+W-2)
* Threshold  T_i = np.percentile(window_i, q, method="linear")
* Window 1: every point in it is labeled with T_1.
  Window i >= 2: ONLY the newly added point (index i+W-2) is labeled with T_i.
* Rule: one-sided (upper tail) -> anomaly if x >= T_i.
* T_i is computed only from values inside window i (no future data -> no leakage).

Usage
-----
    python anomaly_detection.py                   # W=500, q=88 (chosen settings)
    python anomaly_detection.py --W 750 --q 85    # try other settings
    python anomaly_detection.py --no-sweep        # skip the W/q sensitivity sweep
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

DATA_FILE = "AG_NO3_fill_cells_remove_NAN.csv"
VALUE_COL = "NO3N"          # Nitrate column
TRUTH_COL = "Student_Flag"  # ground truth: 1 = anomaly, 0 = normal
OUT_DIR = "outputs"


# ----------------------------------------------------------------------------
# Core algorithm
# ----------------------------------------------------------------------------
def sliding_window_detect(x, W, q):
    """Label each point of x as anomaly (True) / normal (False).

    Returns
    -------
    pred       : bool array, len(x)   -> predicted anomaly flags
    thresholds : float array, len(x)  -> threshold used to label each point
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if W > n:
        raise ValueError("Window size W must not exceed series length")

    pred = np.zeros(n, dtype=bool)
    thresholds = np.full(n, np.nan)

    # --- Window 1: indices [0, W-1]; label every point with T_1 -------------
    T1 = np.percentile(x[0:W], q, method="linear")
    pred[0:W] = x[0:W] >= T1
    thresholds[0:W] = T1

    # --- Windows 2 .. end: slide by 1, label only the newly added point -----
    # Window i (1-based) covers [i-1, i+W-2]; with start = i-1 the new point
    # is at index start + W - 1.
    for start in range(1, n - W + 1):
        window = x[start:start + W]
        T_i = np.percentile(window, q, method="linear")
        new_idx = start + W - 1
        thresholds[new_idx] = T_i
        pred[new_idx] = x[new_idx] >= T_i

    return pred, thresholds


def sliding_window_detect_fast(x, W, q):
    """Vectorised version of sliding_window_detect (identical output).
    Used only for the W/q sensitivity sweep so it runs in seconds."""
    x = np.asarray(x, dtype=float)
    T = np.percentile(sliding_window_view(x, W), q, axis=1, method="linear")
    pred = np.zeros(len(x), dtype=bool)
    pred[:W] = x[:W] >= T[0]
    pred[W - 1:] = x[W - 1:] >= T
    return pred


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def evaluate(pred, truth):
    truth = np.asarray(truth).astype(bool)
    pred = np.asarray(pred).astype(bool)
    TP = int(np.sum(pred & truth))
    FP = int(np.sum(pred & ~truth))
    FN = int(np.sum(~pred & truth))
    TN = int(np.sum(~pred & ~truth))
    P, N = TP + FN, TN + FP
    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "P (total anomalies)": P, "N (total normals)": N,
        "normal_accuracy (TN/N)": TN / N,
        "anomaly_accuracy (TP/P)": TP / P,
    }


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def plot_detections(df, pred, thresholds, truth, W, q, path):
    t = pd.to_datetime(df["Date"], format="%m/%d/%Y %H:%M")
    x = df[VALUE_COL].values
    truth = truth.astype(bool)
    tp = pred & truth
    fn = ~pred & truth
    fp = pred & ~truth

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2]})

    ax = axes[0]
    ax.plot(t, x, color="#9aa4b1", lw=0.6, label="NO3N", zorder=1)
    ax.plot(t, thresholds, color="#e08a00", lw=0.9,
            label=f"Sliding threshold T (q={q}, W={W})", zorder=2)
    ax.scatter(t[fp], x[fp], s=4, color="#f2b8b5", alpha=0.6,
               label=f"False positive ({fp.sum()})", zorder=3)
    ax.scatter(t[tp], x[tp], s=28, color="#1b7f3b", edgecolor="white", lw=0.4,
               label=f"True positive ({tp.sum()})", zorder=5)
    ax.scatter(t[fn], x[fn], s=34, marker="x", color="#c0392b",
               label=f"Missed anomaly / FN ({fn.sum()})", zorder=5)
    ax.set_yscale("log")
    ax.set_ylabel("NO3N (log scale)")
    ax.set_title("Nitrate anomaly detection - sliding-window percentile threshold")
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(t, x, color="#9aa4b1", lw=0.6, label="NO3N")
    ax.scatter(t[pred], x[pred], s=4, color="#d1495b",
               label=f"Predicted anomaly ({pred.sum()})")
    ax.scatter(t[truth], x[truth], s=22, facecolor="none", edgecolor="#1f4e9c",
               lw=0.9, label="Ground-truth anomaly (141)")
    ax.set_ylabel("NO3N (linear)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sensitivity_sweep(x, truth, path_csv, path_png, W_chosen, q_chosen):
    Ws = [100, 250, 500, 750, 1000, 1500, 2000]
    qs = np.round(np.arange(80, 97.5, 0.5), 1)
    rows = []
    for W in Ws:
        for q in qs:
            m = evaluate(sliding_window_detect_fast(x, W, q), truth)
            rows.append({"W": W, "q": q,
                         "normal_acc": m["normal_accuracy (TN/N)"],
                         "anomaly_acc": m["anomaly_accuracy (TP/P)"]})
    res = pd.DataFrame(rows)
    res["meets_targets"] = (res.normal_acc >= 0.80) & (res.anomaly_acc >= 0.75)
    res.to_csv(path_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # (a) accuracy vs q for the chosen W
    sub = res[res.W == W_chosen]
    ax = axes[0]
    ax.plot(sub.q, sub.normal_acc, "-o", ms=3, color="#1f4e9c", label="Normal accuracy (TN/N)")
    ax.plot(sub.q, sub.anomaly_acc, "-o", ms=3, color="#d1495b", label="Anomaly accuracy (TP/P)")
    ax.axhline(0.80, ls="--", color="#1f4e9c", alpha=0.5)
    ax.axhline(0.75, ls="--", color="#d1495b", alpha=0.5)
    ok = sub[sub.meets_targets]
    if len(ok):
        ax.axvspan(ok.q.min(), ok.q.max(), color="#1b7f3b", alpha=0.12,
                   label="Both targets met")
    ax.axvline(q_chosen, color="black", lw=1, label=f"Chosen q = {q_chosen}")
    ax.set_xlabel("Percentile q")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"(a) Accuracy vs q  (W = {W_chosen})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # (b) trade-off curves for each W
    ax = axes[1]
    cmap = plt.get_cmap("viridis")
    for k, W in enumerate(Ws):
        s = res[res.W == W]
        ax.plot(s.normal_acc, s.anomaly_acc, "-", lw=1.4,
                color=cmap(k / (len(Ws) - 1)), label=f"W={W}")
    ax.axvline(0.80, ls="--", color="grey")
    ax.axhline(0.75, ls="--", color="grey")
    ax.fill_between([0.80, 1.0], 0.75, 1.0, color="#1b7f3b", alpha=0.08)
    m = res[(res.W == W_chosen) & (res.q == q_chosen)].iloc[0]
    ax.scatter([m.normal_acc], [m.anomaly_acc], s=90, marker="*", color="black",
               zorder=5, label=f"Chosen (W={W_chosen}, q={q_chosen})")
    ax.set_xlim(0.65, 0.95)
    ax.set_ylim(0.55, 0.95)
    ax.set_xlabel("Normal accuracy (TN/N)")
    ax.set_ylabel("Anomaly accuracy (TP/P)")
    ax.set_title("(b) Trade-off for each W (q swept 80-97)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path_png, dpi=150)
    plt.close(fig)
    return res


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA_FILE)
    ap.add_argument("--W", type=int, default=500, help="window size")
    ap.add_argument("--q", type=float, default=88.0, help="percentile (0-100)")
    ap.add_argument("--no-sweep", action="store_true", help="skip sensitivity sweep")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(args.data)
    df = df.dropna(subset=[VALUE_COL]).reset_index(drop=True)  # file is already NaN-free
    x = df[VALUE_COL].to_numpy(dtype=float)
    truth = df[TRUTH_COL].to_numpy().astype(bool)

    # Run detection (explicit loop implementation)
    pred, thresholds = sliding_window_detect(x, args.W, args.q)
    # Sanity check: vectorised version must agree exactly
    assert np.array_equal(pred, sliding_window_detect_fast(x, args.W, args.q))

    metrics = evaluate(pred, truth)
    metrics.update({"W": args.W, "q": args.q, "rule": "one-sided upper tail: x >= T_i",
                    "n_points": len(x)})

    # Save predictions
    out = df.copy()
    out["threshold"] = thresholds
    out["Predicted_Anomaly"] = pred.astype(int)
    out.to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    plot_detections(df, pred, thresholds, truth, args.W, args.q,
                    os.path.join(OUT_DIR, "anomalies_plot.png"))

    if not args.no_sweep:
        sensitivity_sweep(x, truth,
                          os.path.join(OUT_DIR, "sensitivity_sweep.csv"),
                          os.path.join(OUT_DIR, "sensitivity_W_q.png"),
                          args.W, args.q)

    # Print summary
    print(f"W = {args.W}, q = {args.q}  (rule: anomaly if x >= T_i)")
    print(f"TP = {metrics['TP']}   FP = {metrics['FP']}")
    print(f"FN = {metrics['FN']}   TN = {metrics['TN']}")
    print(f"P = {metrics['P (total anomalies)']}   N = {metrics['N (total normals)']}")
    print(f"Normal  accuracy (TN/N) = {metrics['normal_accuracy (TN/N)']:.2%}")
    print(f"Anomaly accuracy (TP/P) = {metrics['anomaly_accuracy (TP/P)']:.2%}")
    print(f"Outputs written to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
