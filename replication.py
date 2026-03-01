#!/usr/bin/env python3
"""
Replication package for alpha-flops experiments.

Each experiment collects forward-pass timing data for different layer
configurations, stores the results (with quartiles) in a pandas DataFrame
saved as CSV, and produces a PDF plot.

Usage:
    python replication.py collect A        # collect data for Experiment A
    python replication.py plot A           # plot Experiment A from saved CSV
    python replication.py run A            # collect + plot Experiment A
    python replication.py collect all      # collect all experiments
    python replication.py plot all         # plot all experiments
    python replication.py run all          # collect + plot everything
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# --- Configuration -------------------------------------------------------------

GPU_NAME = "rtx4090"
DATA_DIR = os.path.join("data", GPU_NAME)
FIGURES_DIR = os.path.join("figures", GPU_NAME)
NUM_ITERATIONS = 2000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Matplotlib styling
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "legend.fontsize": 14,
})


# --- Utility functions ---------------------------------------------------------

def _ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def _data_path(experiment_id: str) -> str:
    return os.path.join(DATA_DIR, f"experiment-{experiment_id}-data.csv")


def _figure_path(experiment_id: str) -> str:
    return os.path.join(FIGURES_DIR, f"experiment-{experiment_id}.pdf")


def create_conv_layer(in_channels: int, out_channels: int,
                      kernel_size: int = 1, padding=0) -> nn.Conv2d:
    layer = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding,
    )
    torch.manual_seed(0)
    layer.weight.data.normal_(mean=0.0, std=0.02)
    layer.bias.data.zero_()
    return layer.to(device)


def measure_forward_time(layer: nn.Module, input_tensor: torch.Tensor) -> float:
    # torch.cuda.synchronize()
    start = time.time()
    _ = layer(input_tensor)
    #torch.cuda.synchronize()
    end = time.time()
    return end - start


def benchmark(layer: nn.Module, input_tensor: torch.Tensor,
              n_iters: int = NUM_ITERATIONS):
    """Return (mean_time, q25, q75) over *n_iters* forward passes."""
    times = []
    for _ in range(n_iters):
        times.append(measure_forward_time(layer, input_tensor))
    return np.mean(times), np.percentile(times, 25), np.percentile(times, 75)


def alpha_model(S, K, F):
    """Predict forward-pass time (ms) using the alpha-FLOPs model."""
    if K == 1:
        beta = 0.00972514
        gamma = 1.1779239
        final = 0.00549217
    else:
        beta = 0.00785036
        gamma = 1.0601711
        final = 0.00401956

    Sk = np.log(K) + 1
    res = (Sk + beta * (S - Sk)) / S
    res = final * np.exp(gamma * np.log(res))
    return res * F


def compute_flops(W, H, K, C_in, C_out):
    """Compute FLOPs for a conv layer (in millions)."""
    return W * H * K * K * C_in * C_out / 1e6


# ===============================================================================
# Experiment A  (notebook: Figure 3b)
# Conv layers with increasing channel dimensions at equal FLOPs.
#   Conv_1x2 : K=1, H=1, W=2, Cin=2^i*100, Cout=2^i*50
#   Conv_2x2 : K=1, H=2, W=2, Cin=2^i*50,  Cout=2^i*50
#   Conv_4x4 : K=1, H=4, W=4, Cin=2^i*25,  Cout=2^i*25
# ===============================================================================

def collect_experiment_A() -> pd.DataFrame:
    _ensure_dirs()
    rows = []
    for i in range(9):
        scale = 2 ** i

        # Conv 1x2
        c_in, c_out = scale * 100, scale * 50
        layer = create_conv_layer(c_in, c_out, kernel_size=1)
        inp = torch.randn(1, c_in, 1, 2, device=device)
        avg, q25, q75 = benchmark(layer, inp)
        rows.append(dict(layer_type="Conv_1x2", K=1, Cin=c_in, Cout=c_out,
                         W=2, H=1, scale=scale,
                         avg_time=avg, q25=q25, q75=q75))
        print(f"  Conv_1x2  scale={scale} done")

        # Conv 2x2
        c_in = c_out = scale * 50
        layer = create_conv_layer(c_in, c_out, kernel_size=1)
        inp = torch.randn(1, c_in, 2, 2, device=device)
        avg, q25, q75 = benchmark(layer, inp)
        rows.append(dict(layer_type="Conv_2x2", K=1, Cin=c_in, Cout=c_out,
                         W=2, H=2, scale=scale,
                         avg_time=avg, q25=q25, q75=q75))
        print(f"  Conv_2x2  scale={scale} done")

        # Conv 4x4
        c_in = c_out = scale * 25
        layer = create_conv_layer(c_in, c_out, kernel_size=1)
        inp = torch.randn(1, c_in, 4, 4, device=device)
        avg, q25, q75 = benchmark(layer, inp)
        rows.append(dict(layer_type="Conv_4x4", K=1, Cin=c_in, Cout=c_out,
                         W=4, H=4, scale=scale,
                         avg_time=avg, q25=q25, q75=q75))
        print(f"  Conv_4x4  scale={scale} done")

    df = pd.DataFrame(rows)
    df.to_csv(_data_path("A"), index=False)
    print(f"Saved {_data_path('A')}")
    return df


def plot_experiment_A():
    df = pd.read_csv(_data_path("A"))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    configs = [
        ("Conv_1x2",
         r"H=1, W=2, K=1, $C_{in}=input \cdot 100$, $C_{out}=input \cdot 50$"),
        ("Conv_2x2",
         r"H=2, W=2, K=1, $C_{in}=input \cdot 50$, $C_{out}=input \cdot 50$"),
        ("Conv_4x4",
         r"H=4, W=4, K=1, $C_{in}=input \cdot 25$, $C_{out}=input \cdot 25$"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, (layer_type, label) in enumerate(configs):
        sub = df[df["layer_type"] == layer_type].sort_values("scale")
        ax.plot(sub["scale"], sub["avg_time"] * 1e3, marker="o",
                label=label, color=colors[idx])
        alpha_pred = sub.apply(
            lambda r: alpha_model(
                r["H"] * r["W"], r["K"],
                compute_flops(r["W"], r["H"], r["K"], r["Cin"], r["Cout"])),
            axis=1)
        ax.plot(sub["scale"], alpha_pred, marker="x", ls="--",
                color=colors[idx])

    ax.set_xlabel("Input Size (powers of 2)")
    ax.set_ylabel("Forward Pass Time (milliseconds)")
    ax.set_title("Layer Forward Pass Time vs Input Size")
    ax.grid(True, which="both", ls="--")
    ax.legend()

    fig.savefig(_figure_path("A"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {_figure_path('A')}")


# ===============================================================================
# Experiment B  (notebook: Figure 4b - Increase K, Decrease C)
# CxK held constant at {1500, 2100, 3000, 4500}; K varies 1..30; H=W=10.
# ===============================================================================

def collect_experiment_B() -> pd.DataFrame:
    _ensure_dirs()
    rows = []
    for CxK in [1500, 2100, 3000, 4500]:
        for K in range(1, 31):
            C = round(CxK / K)
            layer = create_conv_layer(C, C, kernel_size=K, padding="same")
            inp = torch.randn(1, C, 10, 10, device=device)
            avg, q25, q75 = benchmark(layer, inp)
            rows.append(dict(CxK=CxK, K=K, Cin=C, Cout=C, H=10, W=10,
                             avg_time=avg, q25=q25, q75=q75))
        print(f"  CxK={CxK} done")

    df = pd.DataFrame(rows)
    df.to_csv(_data_path("B"), index=False)
    print(f"Saved {_data_path('B')}")
    return df


def plot_experiment_B():
    df = pd.read_csv(_data_path("B"))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, cxk in enumerate([1500, 2100, 3000, 4500]):
        sub = df[df["CxK"] == cxk].sort_values("K")
        ax.plot(sub["K"], sub["avg_time"] * 1e3, marker="o",
                label=rf"$C \cdot K={cxk}$", color=colors[idx])
        alpha_pred = sub.apply(
            lambda r: alpha_model(
                r["H"] * r["W"], r["K"],
                compute_flops(r["W"], r["H"], r["K"], r["Cin"], r["Cout"])),
            axis=1)
        ax.plot(sub["K"], alpha_pred, marker="x", ls="--",
                color=colors[idx])

    ax.set_xlabel("K Size")
    ax.set_ylabel("Forward Pass Time (milliseconds)")
    ax.set_title(r"Time for increasing kernels and decreasing channels."
                 r" $H=W=10$, $C_{in}=C_{out}=C$")
    ax.grid(True, which="both", ls="--")
    ax.legend()

    fig.savefig(_figure_path("B"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {_figure_path('B')}")


# ===============================================================================
# Experiment C  (notebook: Figure 4b - H=W=300/K, fixed C)
# C in {50, 70, 100, 150}; K varies 1..30; H = W = round(300/K).
# ===============================================================================

def collect_experiment_C() -> pd.DataFrame:
    _ensure_dirs()
    rows = []
    for C in [50, 70, 100, 150]:
        for K in range(1, 31):
            H = round(300 / K)
            layer = create_conv_layer(C, C, kernel_size=K, padding="same")
            inp = torch.randn(1, C, H, H, device=device)
            avg, q25, q75 = benchmark(layer, inp)
            rows.append(dict(K=K, Cin=C, Cout=C, H=H, W=H,
                             avg_time=avg, q25=q25, q75=q75))
        print(f"  C={C} done")

    df = pd.DataFrame(rows)
    df.to_csv(_data_path("C"), index=False)
    print(f"Saved {_data_path('C')}")
    return df


def plot_experiment_C():
    df = pd.read_csv(_data_path("C"))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, C_val in enumerate([50, 70, 100, 150]):
        sub = df[df["Cin"] == C_val].sort_values("K")
        ax.plot(sub["K"], sub["avg_time"] * 1e3, marker="o",
                label=rf"$C_{{in}}=C_{{out}}={C_val}$", color=colors[idx])
        alpha_pred = sub.apply(
            lambda r: alpha_model(
                np.round(r["Cin"] / r["K"]) ** 2, r["K"],
                compute_flops(np.round(r["Cin"] / r["K"]),
                              np.round(r["Cin"] / r["K"]),
                              r["K"], r["Cin"], r["Cout"])),
            axis=1)
        ax.plot(sub["K"], alpha_pred, marker="x", ls="--",
                color=colors[idx])

    ax.set_xlabel("K Size")
    ax.set_ylabel("Forward Pass Time (milliseconds)")
    ax.set_title(r"Time for increasing kernels and decreasing input size."
                 r" $H=W=300/K$")
    ax.grid(True, which="both", ls="--")
    ax.legend()

    fig.savefig(_figure_path("C"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {_figure_path('C')}")


# ===============================================================================
# Experiment D  (notebook: Figure 5a - Vary W)
# W varies 1..649; H=100, K=3, Cout=100; Cin in {50, 100, 150}.
# ===============================================================================

def collect_experiment_D() -> pd.DataFrame:
    _ensure_dirs()
    rows = []
    for Cin in [50, 100, 150]:
        for W in range(1, 650):
            layer = create_conv_layer(Cin, 100, kernel_size=3, padding="same")
            inp = torch.randn(1, Cin, 100, W, device=device)
            avg, q25, q75 = benchmark(layer, inp)
            rows.append(dict(W=W, H=100, Cin=Cin, Cout=100, K=3,
                             avg_time=avg, q25=q25, q75=q75))
        print(f"  Cin={Cin} complete")

    df = pd.DataFrame(rows)
    df.to_csv(_data_path("D"), index=False)
    print(f"Saved {_data_path('D')}")
    return df


def plot_experiment_D():
    df = pd.read_csv(_data_path("D"))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, Cin in enumerate([50, 100, 150]):
        sub = df[df["Cin"] == Cin].sort_values("W")
        ax.scatter(sub["W"], sub["avg_time"] * 1e3, marker="o",
                   label=rf"$C_{{in}}={Cin}$", color=colors[idx])
        alpha_pred = sub.apply(
            lambda r: alpha_model(
                r["W"] * r["H"], r["K"],
                compute_flops(r["W"], r["H"], r["K"],
                              r["Cin"], r["Cout"])),
            axis=1)
        ax.plot(sub["W"], alpha_pred, ls="--", color=colors[idx])

    ax.set_xlabel("W size")
    ax.set_ylabel("Forward Pass Time (milliseconds)")
    ax.set_title(r"Time for increasing W with different channel counts."
                 r" $H=100$, $K=3$, $C_{out}=100$")
    ax.grid(True, which="both", ls="--")
    ax.legend()

    fig.savefig(_figure_path("D"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {_figure_path('D')}")


def plot_experiment_D_reduced():
    df = pd.read_csv(_data_path("D"))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    x_vals = [10] + [i * 20 for i in range(1, 33)]

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, Cin in enumerate([50, 100, 150]):
        sub = df[(df["Cin"] == Cin) & (df["W"].isin(x_vals))].sort_values("W")
        ax.plot(sub["W"], sub["avg_time"] * 1e3, marker="o",
                label=rf"$C_{{in}}={Cin}$", color=colors[idx])
        alpha_pred = sub.apply(
            lambda r: alpha_model(
                r["W"] * r["H"], r["K"],
                compute_flops(r["W"], r["H"], r["K"],
                              r["Cin"], r["Cout"])),
            axis=1)
        ax.plot(sub["W"], alpha_pred, ls="--", color=colors[idx])

    ax.set_xlabel("W size")
    ax.set_ylabel("Forward Pass Time (milliseconds)")
    ax.set_title(r"Time for increasing W with different channel counts."
                 r" $H=100$, $K=3$, $C_{out}=100$")
    ax.grid(True, which="both", ls="--")
    ax.legend()

    path = _figure_path("D-reduced")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ===============================================================================
# Experiment E  (notebook: Figure 5b - Vary Cout)
# Cout varies 1..649; W=100, H=100, K=3; Cin in {50, 100, 150}.
# ===============================================================================

def collect_experiment_E() -> pd.DataFrame:
    _ensure_dirs()
    rows = []
    for Cin in [50, 100, 150]:
        for Cout in range(1, 650):
            layer = create_conv_layer(Cin, Cout, kernel_size=3, padding="same")
            inp = torch.randn(1, Cin, 100, 100, device=device)
            avg, q25, q75 = benchmark(layer, inp)
            rows.append(dict(W=100, H=100, Cin=Cin, Cout=Cout, K=3,
                             avg_time=avg, q25=q25, q75=q75))
        print(f"  Cin={Cin} complete")

    df = pd.DataFrame(rows)
    df.to_csv(_data_path("E"), index=False)
    print(f"Saved {_data_path('E')}")
    return df


def plot_experiment_E():
    df = pd.read_csv(_data_path("E"))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, Cin in enumerate([50, 100, 150]):
        sub = df[df["Cin"] == Cin].sort_values("Cout")
        ax.scatter(sub["Cout"], sub["avg_time"] * 1e3, marker="o",
                   label=rf"$C_{{in}}={Cin}$", color=colors[idx])
        alpha_pred = sub.apply(
            lambda r: alpha_model(
                r["Cout"] * r["H"], r["K"],
                compute_flops(r["Cout"], r["H"], r["K"],
                              r["Cin"], r["Cin"])),
            axis=1)
        ax.plot(sub["Cout"], alpha_pred, ls="--", color=colors[idx])

    ax.set_xlabel(r"$C_{out}$ size")
    ax.set_ylabel("Forward Pass Time (milliseconds)")
    ax.set_title(r"Time for increasing $C_{out}$ with different"
                 r" $C_{in}$ counts. $H=W=100$")
    ax.grid(True, which="both", ls="--")
    ax.legend()

    fig.savefig(_figure_path("E"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {_figure_path('E')}")


def plot_experiment_E_reduced():
    df = pd.read_csv(_data_path("E"))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    x_vals = [10] + [i * 20 for i in range(1, 33)]

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, Cin in enumerate([50, 100, 150]):
        sub = df[(df["Cin"] == Cin) & (df["Cout"].isin(x_vals))].sort_values("Cout")
        ax.plot(sub["Cout"], sub["avg_time"] * 1e3, marker="o",
                label=rf"$C_{{in}}={Cin}$", color=colors[idx])
        alpha_pred = sub.apply(
            lambda r: alpha_model(
                r["Cout"] * r["H"], r["K"],
                compute_flops(r["Cout"], r["H"], r["K"],
                              r["Cin"], r["Cin"])),
            axis=1)
        ax.plot(sub["Cout"], alpha_pred, ls="--", color=colors[idx])

    ax.set_xlabel(r"$C_{out}$ size")
    ax.set_ylabel("Forward Pass Time (milliseconds)")
    ax.set_title(r"Time for increasing $C_{out}$ with different"
                 r" $C_{in}$ counts. $H=W=100$")
    ax.grid(True, which="both", ls="--")
    ax.legend()

    path = _figure_path("E-reduced")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ===============================================================================
# Experiment F  (notebook: Figure 6a - Vary Cout, K=1, different H)
# Cout varies 1..649; W=100, K=1, Cin=50; H in {100, 200, 300}.
# ===============================================================================

def collect_experiment_F() -> pd.DataFrame:
    _ensure_dirs()
    rows = []
    for H in [100, 200, 300]:
        for Cout in range(1, 650):
            layer = create_conv_layer(50, Cout, kernel_size=1, padding="same")
            inp = torch.randn(1, 50, H, 100, device=device)
            avg, q25, q75 = benchmark(layer, inp)
            rows.append(dict(W=100, H=H, Cin=50, Cout=Cout, K=1,
                             avg_time=avg, q25=q25, q75=q75))
        print(f"  H={H} complete")

    df = pd.DataFrame(rows)
    df.to_csv(_data_path("F"), index=False)
    print(f"Saved {_data_path('F')}")
    return df


def plot_experiment_F():
    df = pd.read_csv(_data_path("F"))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, H_val in enumerate([100, 200, 300]):
        sub = df[df["H"] == H_val].sort_values("Cout")
        ax.scatter(sub["Cout"], sub["avg_time"] * 1e3, marker="o",
                   label=rf"$H={H_val}$", color=colors[idx])
        alpha_pred = sub.apply(
            lambda r: alpha_model(
                r["H"] * r["W"], r["K"],
                compute_flops(r["W"], r["H"], r["K"],
                              r["Cin"], r["Cout"])),
            axis=1)
        ax.plot(sub["Cout"], alpha_pred, ls="--", color=colors[idx])

    ax.set_xlabel(r"$C_{out}$ size")
    ax.set_ylabel("Forward Pass Time (milliseconds)")
    ax.set_title(r"Time for increasing $C_{out}$ with different"
                 r" $H$ values. $W=100$, $C_{in}=50$, $K=1$")
    ax.grid(True, which="both", ls="--")
    ax.legend()

    fig.savefig(_figure_path("F"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {_figure_path('F')}")


def plot_experiment_F_reduced():
    df = pd.read_csv(_data_path("F"))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    x_vals = [10] + [i * 20 for i in range(1, 33)]

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, H_val in enumerate([100, 200, 300]):
        sub = df[(df["H"] == H_val) & (df["Cout"].isin(x_vals))].sort_values("Cout")
        ax.plot(sub["Cout"], sub["avg_time"] * 1e3, marker="o",
                label=rf"$H={H_val}$", color=colors[idx])
        alpha_pred = sub.apply(
            lambda r: alpha_model(
                r["H"] * r["W"], r["K"],
                compute_flops(r["W"], r["H"], r["K"],
                              r["Cin"], r["Cout"])),
            axis=1)
        ax.plot(sub["Cout"], alpha_pred, ls="--", color=colors[idx])

    ax.set_xlabel(r"$C_{out}$ size")
    ax.set_ylabel("Forward Pass Time (milliseconds)")
    ax.set_title(r"Time for increasing $C_{out}$ with different"
                 r" $H$ values. $W=100$, $C_{in}=50$, $K=1$")
    ax.grid(True, which="both", ls="--")
    ax.legend()

    path = _figure_path("F-reduced")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ===============================================================================
# Experiment G  (notebook: Figure 6b - Vary Cin, different K)
# Cin varies 1..999; W=10, H=10, Cout=1000; K in {1, 3, 5}.
# ===============================================================================

def collect_experiment_G() -> pd.DataFrame:
    _ensure_dirs()
    rows = []
    for K in [1, 3, 5]:
        for Cin in range(1, 1000):
            layer = create_conv_layer(Cin, 1000, kernel_size=K, padding="same")
            inp = torch.randn(1, Cin, 10, 10, device=device)
            avg, q25, q75 = benchmark(layer, inp)
            rows.append(dict(W=10, H=10, Cin=Cin, Cout=1000, K=K,
                             avg_time=avg, q25=q25, q75=q75))
        print(f"  K={K} done")

    df = pd.DataFrame(rows)
    df.to_csv(_data_path("G"), index=False)
    print(f"Saved {_data_path('G')}")
    return df


def plot_experiment_G():
    df = pd.read_csv(_data_path("G"))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, K_val in enumerate([1, 3, 5]):
        sub = df[df["K"] == K_val].sort_values("Cin")
        ax.scatter(sub["Cin"], sub["avg_time"] * 1e3, marker="o",
                   label=rf"$K={K_val}$", color=colors[idx])
        alpha_pred = sub.apply(
            lambda r: alpha_model(
                r["H"] * r["W"], r["K"],
                compute_flops(r["W"], r["H"], r["K"],
                              r["Cin"], r["Cout"])),
            axis=1)
        ax.plot(sub["Cin"], alpha_pred, ls="--", color=colors[idx])

    ax.set_xlabel(r"$C_{in}$ size")
    ax.set_ylabel("Forward Pass Time (milliseconds)")
    ax.set_title(r"Time for increasing $C_{in}$ with different"
                 r" $K$ values. $W=10, H=10$, $C_{out}=1000$")
    ax.grid(True, which="both", ls="--")
    ax.legend()

    fig.savefig(_figure_path("G"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {_figure_path('G')}")


def plot_experiment_G_reduced():
    df = pd.read_csv(_data_path("G"))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    x_vals = [i * 100 for i in range(1, 10)]

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, K_val in enumerate([1, 3, 5]):
        sub = df[(df["K"] == K_val) & (df["Cin"].isin(x_vals))].sort_values("Cin")
        ax.plot(sub["Cin"], sub["avg_time"] * 1e3, marker="o",
                label=rf"$K={K_val}$", color=colors[idx])
        alpha_pred = sub.apply(
            lambda r: alpha_model(
                r["H"] * r["W"], r["K"],
                compute_flops(r["W"], r["H"], r["K"],
                              r["Cin"], r["Cout"])),
            axis=1)
        ax.plot(sub["Cin"], alpha_pred, ls="--", color=colors[idx])

    ax.set_xlabel(r"$C_{in}$ size")
    ax.set_ylabel("Forward Pass Time (milliseconds)")
    ax.set_title(r"Time for increasing $C_{in}$ with different"
                 r" $K$ values. $W=10, H=10$, $C_{out}=1000$")
    ax.grid(True, which="both", ls="--")
    ax.legend()

    path = _figure_path("G-reduced")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ===============================================================================
# Dispatcher
# ===============================================================================

EXPERIMENTS = list("ABCDEFG")

COLLECTORS = {
    "A": collect_experiment_A,
    "B": collect_experiment_B,
    "C": collect_experiment_C,
    "D": collect_experiment_D,
    "E": collect_experiment_E,
    "F": collect_experiment_F,
    "G": collect_experiment_G,
}

PLOTTERS = {
    "A": [plot_experiment_A],
    "B": [plot_experiment_B],
    "C": [plot_experiment_C],
    "D": [plot_experiment_D, plot_experiment_D_reduced],
    "E": [plot_experiment_E, plot_experiment_E_reduced],
    "F": [plot_experiment_F, plot_experiment_F_reduced],
    "G": [plot_experiment_G, plot_experiment_G_reduced],
}


def main():
    parser = argparse.ArgumentParser(
        description="Replication package - CNN layer timing experiments")
    parser.add_argument("action", choices=["collect", "plot", "run"],
                        help="collect = gather timing data; "
                             "plot = generate PDF from CSV; "
                             "run = collect + plot")
    parser.add_argument("experiment", type=str,
                        help="Experiment ID (A..G) or 'all'")
    args = parser.parse_args()

    ids = EXPERIMENTS if args.experiment.lower() == "all" else [args.experiment.upper()]

    for eid in ids:
        if eid not in COLLECTORS:
            print(f"Unknown experiment: {eid}")
            continue
        print(f"\n{'='*60}")
        print(f" Experiment {eid}")
        print(f"{'='*60}")
        if args.action in ("collect", "run"):
            COLLECTORS[eid]()
        if args.action in ("plot", "run"):
            for plotter in PLOTTERS[eid]:
                plotter()


if __name__ == "__main__":
    main()
