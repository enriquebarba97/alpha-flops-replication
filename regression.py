#!/usr/bin/env python3
"""
Fit the alpha-FLOPs regression model from a Conv2d timing sweep.

Collects a grid of Conv2d forward-pass timings, then fits the
alpha-FLOPs model separately for K=1 and K>1 groups using
scipy.optimize.curve_fit.

Usage:
    python regression.py collect    # run the measurement sweep
    python regression.py fit        # fit model from saved CSV
    python regression.py run        # collect + fit
"""

import argparse
import csv
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import curve_fit

# --- Configuration ---

NUM_WARMUP = 10
NUM_ITERATIONS = 50

GPU_NAME = "rtx4090"
DATA_DIR = os.path.join("data", GPU_NAME)
DATA_FILE = os.path.join(DATA_DIR, "regression-data.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)


# --- Utility functions ---

def create_conv_layer(Cin, Cout, K):
    """Create a Conv2d layer (no bias, K//2 padding, stride 1)."""
    return nn.Conv2d(Cin, Cout, K, stride=1, padding=K // 2, bias=False).to(device)


def compute_flops(W, H, K, Cin, Cout):
    """Compute FLOPs for a conv layer (in millions)."""
    return W * H * K * K * Cin * Cout / 1e6


def benchmark(W, H, Cin, Cout, K):
    """Warm up and time a Conv2d forward pass.

    Returns avg_time (seconds).
    """
    x = torch.randn(1, Cin, H, W, device=device)
    layer = create_conv_layer(Cin, Cout, K)

    # Warm-up
    for _ in range(NUM_WARMUP):
        _ = layer(x)
    torch.cuda.synchronize()

    # Measure
    start = time.time()
    for _ in range(NUM_ITERATIONS):
        _ = layer(x)
    torch.cuda.synchronize()
    end = time.time()

    return (end - start) / NUM_ITERATIONS


# --- Data collection ---

SIZES = [1, 2, 4, 8, 16, 32, 64, 96, 128, 256]
CHANNELS = [10*i for i in range(1,10)] + [100 + 50*i for i in range(0,20)] + [2**i*100 for i in range(4,6)]
KERNELS = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]


def collect():
    """Run the Conv2d timing sweep and save to CSV."""
    _ensure_dirs()
    with open(DATA_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["W", "H", "Cin", "Cout", "K", "avg_time", "FLOPs"])

        for K in KERNELS:
            for Cin in CHANNELS:
                for Cout in CHANNELS:
                    for W in SIZES:
                        H = W
                        avg_time = benchmark(W, H, Cin, Cout, K)
                        flops = compute_flops(W, H, K, Cin, Cout)
                        writer.writerow([W, H, Cin, Cout, K, avg_time, flops])

    print(f"Saved: {DATA_FILE}")


# --- Regression ---

def alpha_model(x, beta, gamma, final):
    S, K, F = x 
    Sk = np.log(K) + 1
    res = (Sk + beta * (S - Sk)) / S
    res = np.log(res)
    res = gamma * res
    res = final * np.exp(res)
    return res * F


def fit_group(df):
    """Fit alpha-FLOPs model to a DataFrame group (K=1 or K>1).

    Returns (beta, gamma, c) parameters.
    """
    S = (df["W"] * df["H"]).values.astype(float)
    K = df["K"].values.astype(float)
    avg_time = df["avg_time"].values * 1e3
    flops = df["FLOPs"].values
    y = avg_time

    params, _ = curve_fit(
        alpha_model, (S, K, flops), y,
        p0=[0.01, 0.8, 0.3],
        bounds=([0, 0, 0], [2, 2, 2]),
    )
    return params


def fit():
    """Load sweep CSV and fit alpha-FLOPs model for K=1 and K>1."""
    df = pd.read_csv(DATA_FILE)

    df_k1 = df[df["K"] == 1]
    df_kn = df[df["K"] > 1]

    params_k1 = fit_group(df_k1)
    params_kn = fit_group(df_kn)

    print("Fitting for K = 1:")
    print(f"  beta  = {params_k1[0]:.8f}")
    print(f"  gamma = {params_k1[1]:.7f}")
    print(f"  final     = {params_k1[2]:.8e}")

    print("\nFitting for K > 1:")
    print(f"  beta  = {params_kn[0]:.8f}")
    print(f"  gamma = {params_kn[1]:.7f}")
    print(f"  final     = {params_kn[2]:.8e}")

    return params_k1, params_kn


# --- CLI ---

def main():
    global GPU_NAME, DATA_DIR, DATA_FILE

    parser = argparse.ArgumentParser(description="Alpha-FLOPs regression")
    parser.add_argument(
        "action", choices=["collect", "fit", "run"],
        help="collect: run sweep, fit: fit model, run: both",
    )
    parser.add_argument(
        "--gpu-name", type=str, default=GPU_NAME,
        help=f"GPU name used for data subdirectory (default: {GPU_NAME})",
    )
    args = parser.parse_args()

    GPU_NAME = args.gpu_name
    DATA_DIR = os.path.join("data", GPU_NAME)
    DATA_FILE = os.path.join(DATA_DIR, "regression-data.csv")

    if args.action in ("collect", "run"):
        collect()
    if args.action in ("fit", "run"):
        fit()


if __name__ == "__main__":
    main()
