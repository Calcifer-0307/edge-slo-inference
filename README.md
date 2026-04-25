# Edge-AI Multi-Tenant Simulation

This repository implements a CPU-only edge inference simulation for multi-tenant, multi-model serving with:

- SLO-aware scheduling
- model downgrade / adaptation
- admission control
- measurement-driven service-time profiling

The code is organized to match the development guide in `edge-ai-sim-dev-guide.md`.

## Repository Layout

```text
configs/         simulation configuration
data/            profiling-derived service times and accuracy proxy
profiling/       ONNX export and profiling scripts
sim/             simulation core, policies, metrics, experiments
results/         CSV outputs
figs/            generated figures
```

Large ONNX model artifacts are intentionally excluded from version control to keep the repository lightweight. They should be generated locally only when profiling is needed.

## Environment Used For Current Results

- Machine: Apple MacBook Pro, Apple M1 Pro
- OS Kernel: Darwin 25.0.0 arm64
- Python: 3.13.7
- onnxruntime: 1.25.0
- torch: 2.9.0
- torchvision: 0.24.0
- pandas: 2.3.3
- numpy: 2.2.6
- simpy: 4.1.1
- pyyaml: 6.0.2

## Install

```bash
pip install -r requirements.txt
```

## Profiling Workflow

1. Generate the local ONNX profiling models:

```bash
python3 profiling/generate_onnx.py
```

This step downloads pretrained `torchvision` weights on demand and exports local ONNX files into `profiling/models/`. Those binary files are not tracked by Git.

2. Run CPU profiling with warm-up and repeated measurements:

```bash
python3 profiling/profile_models.py
```

This writes:

- `data/service_time.csv`
- local ONNX artifacts under `profiling/models/`

If the local ONNX models are missing, `profile_models.py` will stop and ask you to run `python3 profiling/generate_onnx.py` first.

## Simulation Entry Points

Run one simulation:

```bash
python3 -m sim.simulate --config configs/default.yaml
```

Run Phase 2 strategy comparison:

```bash
python3 -m sim.plot --config configs/default.yaml
```

Run Phase 3 robustness / ablation experiments:

```bash
python3 -m sim.phase3 --config configs/default.yaml
```

## Current Strategy Set

- `FIFO + Fixed`
- `EDF + Fixed`
- `EDF + Threshold Downgrade`
- `EDF + Threshold Downgrade + Admission`
- `Proposed: EDF + Feasible Utility`
- `Proposed: EDF + Feasible Utility + Admission`

## Data Files

- `data/service_time.csv`: measured latency statistics from local profiling
- `data/accuracy.csv`: accuracy proxy for each model tier

If these files exist, the simulator loads them and overrides fallback values in `configs/default.yaml`.

## Repository Size Notes

- The repository no longer tracks `profiling/models/*.onnx` or `profiling/models/*.onnx.data`.
- This avoids committing large binary artifacts to GitHub.
- If you need to reproduce the profiling pipeline from scratch, regenerate the local models first and then rerun profiling.

## Reproducibility Notes

- Random seeds are fixed in batch experiments.
- `service_time.csv` comes from repeated inference runs after warm-up.
- The current profiling input shape is `1x3x224x224`.
- Accuracy values are proxies, not locally re-measured task accuracy.

## Threats To Validity

- Accuracy is represented by a proxy rather than task-specific evaluation.
- The simulator models a single non-preemptive CPU server, not full multi-core execution.
- Admission and downgrade decisions use expected service times rather than exact future runtimes.
