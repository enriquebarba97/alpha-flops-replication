# Alpha-FLOPs Replication Package

## Introduction

This repository contains the replication package for the paper "FLOPs vs Real Work: The Importance of Replication for AI Performance Assessment":

The package reproduces the experiments presented in the paper, which
characterise the relationship between theoretical FLOPs and actual
forward-pass latency of Conv2d layers on GPU hardware. It includes:

- A **regression script** that collects a grid of Conv2d timings and fits the
  alpha-FLOPs model (separately for K=1 and K>1 kernels).
- A **replication script** that runs each of the seven experiments (A–G) from
  the paper, producing CSV data files and PDF plots that compare measured
  timings against the alpha-FLOPs model predictions.

## Directory Structure

```
.
├── Dockerfile              # Container definition (Ubuntu 24.04 + Python 3.13 + PyTorch)
├── LICENSE
├── README.md               # This file
├── convert_experiments.py   # Converts raw experiment summaries to replication-compatible CSVs
├── regression.py           # Regression: collect timing grid & fit alpha-FLOPs model
├── replication.py          # Replication: run experiments A–G, generate plots
├── data/
│   └── <gpu-name>/         # Collected data in CSV
└── figures/
    └── <gpu-name>/         # Generated PDF plots (one subdirectory per GPU)
        ├── experiment-A.pdf
        ├── ...
        └── experiment-G.pdf
```

## Building and Running the Docker Container

### Prerequisites

- Docker installed on the host machine.
- An NVIDIA GPU with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  installed (required for GPU access inside the container).

### Build the image

```bash
docker build -t alpha-flops-replication .
```

### Run the container

Mount the repository directory so that data and figures are persisted on the
host:

```bash
docker run --rm -it --gpus all -v "$(pwd)":/root alpha-flops-replication
```

This drops you into an interactive shell inside the container at `/root`,
where the repository files are available. All output files (CSVs, PDFs) are
written back to the host through the bind mount.

> **CPU-only mode:** If no GPU is available, omit `--gpus all`. PyTorch will
> fall back to CPU automatically, though timings will not be representative of
> the paper's results.

## Using the Regression Script

`regression.py` collects a grid of Conv2d forward-pass timings across
different spatial sizes, channel counts, and kernel sizes, then fits the
alpha-FLOPs model parameters via `scipy.optimize.curve_fit`.

### Commands

| Command   | Description                              |
|-----------|------------------------------------------|
| `collect` | Run the measurement sweep and save a CSV |
| `fit`     | Fit the model from a previously saved CSV|
| `run`     | Collect data and then fit (both steps)   |

### Examples

```bash
# Collect timing data (saved to data/<gpu-name>/regression-data.csv)
python regression.py collect

# Fit the model from existing data
python regression.py fit

# Collect and fit in one step
python regression.py run

# Use a custom GPU name for the output subdirectory
python regression.py run --gpu-name a100
```

### Output

- **CSV:** `data/<gpu-name>/regression-data.csv` — columns: `W`, `H`, `Cin`,
  `Cout`, `K`, `avg_time`, `FLOPs`.
- **Console:** Fitted parameters (`beta`, `gamma`, `final`) for the K=1 and
  K>1 groups.

## Using the Replication Script

`replication.py` runs the seven experiments (A–G) from the paper. Each
experiment benchmarks Conv2d forward passes under a specific configuration,
saves timing results to CSV, and produces PDF plots comparing measured times
to the alpha-FLOPs model predictions.

### Experiments

| ID | Description |
|----|-------------|
| A  | Increasing channel dimensions at equal FLOPs (K=1) |
| B  | Increasing kernel size with decreasing spatial size (H=W=300/K, fixed C) |
| C  | Increasing kernel size with decreasing channels (C·K constant, H=W=10) |
| D  | Varying spatial width W (H=100, K=3, C_out=100) |
| E  | Varying output channels C_out (H=W=100, K=3) |
| F  | Varying output channels C_out with different H (K=1, C_in=50) |
| G  | Varying input channels C_in with different K (H=W=10, C_out=1000) |

### Commands

| Command   | Description                              |
|-----------|------------------------------------------|
| `collect` | Run the timing benchmark and save CSV    |
| `plot`    | Generate PDF plots from existing CSV     |
| `run`     | Collect data and plot (both steps)       |

The second argument selects the experiment: a letter `A`–`G`, or `all`.

### Examples

```bash
# Run experiment A (collect + plot)
python replication.py run A

# Collect data for all experiments
python replication.py collect all

# Plot experiment D from previously collected data
python replication.py plot D

# Use a custom GPU name
python replication.py run all --gpu-name a100
```

### Output

- **CSV:** `data/<gpu-name>/experiment-<ID>-data.csv`
- **PDF:** `figures/<gpu-name>/experiment-<ID>.pdf`
  (experiments D–G also produce a `*-reduced.pdf` variant with fewer data
  points for clearer visualisation)
