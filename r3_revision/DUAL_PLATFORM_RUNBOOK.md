# HAD-MC Dual-Platform Runbook

This runbook documents the added GPU/DCU execution entrypoints introduced for TPDS supplementary experiments.

## Goal

Use the same R3 experiment code on two heterogeneous accelerator platforms:
- DCU platform with DTK/ HIP runtime
- NVIDIA GPU platform such as V100 or A100

## Files Added

- `code/platform_probe.py`
- `run_probe_dcu.sh`
- `run_probe_gpu.sh`
- `run_full_dcu.sh`
- `run_full_gpu.sh`

## Probe First

### DCU

```bash
cd r3_revision
bash run_probe_dcu.sh /path/to/probe_output
```

### GPU

```bash
cd r3_revision
bash run_probe_gpu.sh /path/to/probe_output
```

The probe writes a JSON file with:
- torch version
- CUDA/HIP backend info
- device count and device name
- a small matrix-multiplication sanity benchmark

## Full Experiment Run

### DCU

```bash
cd r3_revision
bash run_full_dcu.sh /path/to/results_dcu
```

### GPU

```bash
cd r3_revision
bash run_full_gpu.sh /path/to/results_gpu
```

Both wrappers call:

```bash
python3 -u code/hadmc_experiments_complete.py \
  --platform-tag <platform> \
  --results-dir <results_dir> \
  --allow-missing-financial
```

## Output Structure

Each run writes to a platform-specific results directory:

- `RUN_METADATA.json`
- `models/`
- `COMPLETE_EXPERIMENT_RESULTS.json`
- any generated figures or logs layered on top by the caller

## Financial Dataset Note

The current R3 script still references a proprietary financial dataset on disk.
The new `--allow-missing-financial` flag allows the run to continue and explicitly record that the financial sub-experiment was skipped when those files are unavailable.

## Recommended Git Workflow

Use the local branch:

```bash
git checkout tpds-dual-platform
```

Do not push directly to `upstream`. Add a separate remote for your fork or target repository before publishing the code.