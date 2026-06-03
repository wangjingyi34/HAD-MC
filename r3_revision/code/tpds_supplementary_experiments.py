#!/usr/bin/env python3
"""TPDS supplementary experiments for the HAD-MC dual-platform revision."""

import argparse
import copy
import json
import os
import subprocess
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

import hadmc_experiments_complete as core


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TPDS supplementary experiments on the current hardware platform"
    )
    parser.add_argument(
        "--results-dir",
        default=os.environ.get(
            "HADMC_TPDS_RESULTS_DIR",
            os.path.expanduser("~/HAD-MC/r3_revision/results/tpds_supplementary")
        ),
        help="Directory used to store supplementary experiment outputs",
    )
    parser.add_argument(
        "--platform-tag",
        default=os.environ.get("HADMC_PLATFORM_TAG", core.PLATFORM_TAG),
        help="Logical platform label such as dcu, v100, or gpu",
    )
    parser.add_argument(
        "--seeds",
        default="11,22,33,44,55",
        help="Comma-separated seed list for the variance experiment",
    )
    parser.add_argument(
        "--neudet-num-per-class",
        type=int,
        default=300,
        help="Number of synthetic NEU-DET samples per class",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Mini-batch size used for training and evaluation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=int(os.environ.get("HADMC_NUM_WORKERS", "2")),
        help="DataLoader worker count",
    )
    parser.add_argument(
        "--baseline-epochs",
        type=int,
        default=40,
        help="Epoch count used to train each NEU-DET baseline model",
    )
    parser.add_argument(
        "--public-epochs",
        type=int,
        default=15,
        help="Epoch count used for the optional public benchmark baseline",
    )
    parser.add_argument(
        "--prune-ratio",
        type=float,
        default=0.5,
        help="Primary prune ratio used for HAD-MC compression",
    )
    parser.add_argument(
        "--lut-prune-ratios",
        default="0.1,0.2,0.3,0.4,0.5,0.6",
        help="Comma-separated prune ratios used to build LUT validation candidates",
    )
    parser.add_argument(
        "--lut-warmup",
        type=int,
        default=15,
        help="Warmup iterations used for model and operator latency measurement",
    )
    parser.add_argument(
        "--lut-runs",
        type=int,
        default=60,
        help="Repeated timing iterations used for model and operator latency measurement",
    )
    parser.add_argument(
        "--lut-batch-size",
        type=int,
        default=int(os.environ.get("HADMC_LUT_BATCH_SIZE", "32")),
        help=(
            "Batch size used for LUT validation. The per-operator LUT and the "
            "reference whole-model latency are both measured at this batch "
            "size so that compute dominates kernel-launch overhead and the "
            "per-operator additive model becomes physically meaningful."
        ),
    )
    parser.add_argument(
        "--public-train-subset",
        type=int,
        default=5000,
        help="Maximum number of CIFAR-10 training samples used for the public benchmark",
    )
    parser.add_argument(
        "--public-test-subset",
        type=int,
        default=1000,
        help="Maximum number of CIFAR-10 test samples used for the public benchmark",
    )
    parser.add_argument(
        "--skip-public-benchmark",
        action="store_true",
        help="Skip the optional CIFAR-10 public benchmark if runtime budget is tight",
    )
    return parser.parse_args()


ARGS = parse_args()
RESULTS_DIR = os.path.abspath(os.path.expanduser(ARGS.results_dir))
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
VARIANCE_MODELS_DIR = os.path.join(MODELS_DIR, "variance")
os.makedirs(VARIANCE_MODELS_DIR, exist_ok=True)


def parse_int_list(values):
    return [int(value.strip()) for value in values.split(",") if value.strip()]


def parse_float_list(values):
    return [float(value.strip()) for value in values.split(",") if value.strip()]


SEEDS = parse_int_list(ARGS.seeds)
LUT_PRUNE_RATIOS = parse_float_list(ARGS.lut_prune_ratios)


def save_json(relative_path, payload):
    output_path = os.path.join(RESULTS_DIR, relative_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(core.clean_for_json(payload), handle, indent=2)
    return output_path


def get_git_commit():
    env_commit = os.environ.get("HADMC_GIT_COMMIT")
    if env_commit:
        return env_commit
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def set_seed(seed):
    core.set_global_seed(seed)


def make_loader(dataset, shuffle):
    return DataLoader(
        dataset,
        batch_size=ARGS.batch_size,
        shuffle=shuffle,
        num_workers=ARGS.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def create_neudet_loaders(seed):
    set_seed(seed)
    train_imgs, train_lbls, test_imgs, test_lbls = core.create_neudet_dataset(
        num_per_class=ARGS.neudet_num_per_class,
        img_size=64,
        num_classes=6,
    )
    train_ds = TensorDataset(train_imgs, train_lbls)
    test_ds = TensorDataset(test_imgs, test_lbls)
    return make_loader(train_ds, shuffle=True), make_loader(test_ds, shuffle=False)


def build_hadmc_variants(teacher_model, train_loader, prune_ratio):
    compression_only = core.smart_structural_prune(teacher_model, prune_ratio=prune_ratio)
    pruning_info = getattr(compression_only, "_pruning_info", None)
    compression_only, distill_losses = core.distill_model(
        teacher_model,
        compression_only,
        train_loader,
        num_epochs=25,
        temperature=4.0,
        alpha=0.7,
        lr=0.01,
    )
    compression_only, ft_losses, ft_accs = core.train_model(
        compression_only,
        train_loader,
        num_epochs=15,
        lr=0.005,
        verbose=False,
    )
    fused_model = core.fuse_conv_bn(compression_only)
    full_model = core.quantize_model_int8(fused_model)
    quantization_info = getattr(full_model, "_quantization_info", None)
    info = {
        "prune_ratio": 1.0 - core.count_params(compression_only) / core.count_params(teacher_model),
        "distill_losses": distill_losses,
        "ft_losses": ft_losses,
        "ft_accs": ft_accs,
        "pruning_info": pruning_info,
        "quantization_info": quantization_info,
    }
    return compression_only, fused_model, full_model, info


def augment_hadmc_results(baseline_results, hadmc_results, info):
    hadmc_results = dict(hadmc_results)
    hadmc_results["compression_info"] = {
        "prune_ratio": info["prune_ratio"],
        "compression_ratio": 1.0 - hadmc_results["num_params"] / baseline_results["num_params"],
        "size_reduction": 1.0 - hadmc_results["model_size_mb"] / baseline_results["model_size_mb"],
        "speedup": baseline_results["latency_ms"] / hadmc_results["latency_ms"],
    }
    hadmc_results["effective_size_mb"] = round(hadmc_results["model_size_mb"] / 4, 4)
    hadmc_results["pruning_info"] = info.get("pruning_info")
    hadmc_results["quantization_info"] = info.get("quantization_info")
    return hadmc_results


def summarize_scalar_list(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean": round(float(arr.mean()), 6),
        "std": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 6),
        "min": round(float(arr.min()), 6),
        "max": round(float(arr.max()), 6),
    }


def save_seed_models(seed, baseline_model, compression_only_model, baseline_results, hadmc_results, info):
    base_prefix = os.path.join(VARIANCE_MODELS_DIR, f"seed_{seed}")
    torch.save(baseline_model.state_dict(), base_prefix + "_baseline.pth")
    torch.save(compression_only_model.state_dict(), base_prefix + "_compression_only.pth")
    metadata = {
        "seed": seed,
        "baseline_base_width": baseline_model.conv1.out_channels,
        "compression_only_base_width": compression_only_model.conv1.out_channels,
        "baseline_results": baseline_results,
        "hadmc_results": hadmc_results,
        "compression_info": info,
    }
    save_json(f"variance/seed_{seed}_metadata.json", metadata)


def load_reference_models(seed):
    metadata_path = os.path.join(RESULTS_DIR, f"variance/seed_{seed}_metadata.json")
    if not os.path.exists(metadata_path):
        raise RuntimeError(
            f"Missing variance metadata for seed {seed}: {metadata_path}. "
            "Variance experiment must complete before decomposition or LUT validation can run."
        )
    try:
        with open(metadata_path, "r") as handle:
            metadata = json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Failed to read variance metadata: {metadata_path}") from exc

    required_keys = {"baseline_base_width", "compression_only_base_width", "baseline_results", "hadmc_results"}
    missing_keys = sorted(required_keys.difference(metadata.keys()))
    if missing_keys:
        raise RuntimeError(
            f"Variance metadata is incomplete: {metadata_path}. Missing keys: {', '.join(missing_keys)}"
        )

    baseline = core.ResNet18Small(num_classes=6, base_width=metadata["baseline_base_width"])
    compression_only = core.ResNet18Small(num_classes=6, base_width=metadata["compression_only_base_width"])

    baseline_path = os.path.join(VARIANCE_MODELS_DIR, f"seed_{seed}_baseline.pth")
    compression_only_path = os.path.join(VARIANCE_MODELS_DIR, f"seed_{seed}_compression_only.pth")

    if not os.path.exists(baseline_path) or not os.path.exists(compression_only_path):
        raise RuntimeError(
            "Missing saved variance models required for downstream experiments: "
            f"{baseline_path}, {compression_only_path}"
        )

    baseline.load_state_dict(torch.load(baseline_path, map_location="cpu"))
    compression_only.load_state_dict(torch.load(compression_only_path, map_location="cpu"))
    return baseline, compression_only, metadata


def run_variance_experiment():
    print("\n" + "=" * 70)
    print("SUPPLEMENTARY EXPERIMENT 1: Multi-Seed Variance")
    print("=" * 70)

    runs = []
    for index, seed in enumerate(SEEDS, start=1):
        print(f"\n[Variance] Seed {seed} ({index}/{len(SEEDS)})")
        train_loader, test_loader = create_neudet_loaders(seed)

        baseline_model = core.ResNet18Small(num_classes=6, base_width=64)
        baseline_model, train_losses, train_accs = core.train_model(
            baseline_model,
            train_loader,
            num_epochs=ARGS.baseline_epochs,
            lr=0.01,
        )
        baseline_results = core.evaluate_model(baseline_model, test_loader)
        baseline_results["train_losses"] = train_losses
        baseline_results["train_accs"] = train_accs

        compression_only_model, _, full_model, info = build_hadmc_variants(
            baseline_model,
            train_loader,
            prune_ratio=ARGS.prune_ratio,
        )
        hadmc_results = augment_hadmc_results(
            baseline_results,
            core.evaluate_model(full_model, test_loader),
            info,
        )

        run_result = {
            "seed": seed,
            "baseline": baseline_results,
            "hadmc": hadmc_results,
        }
        runs.append(run_result)
        save_seed_models(seed, baseline_model, compression_only_model, baseline_results, hadmc_results, info)
        save_json("variance/partial_results.json", {"completed_runs": runs})

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "baseline_accuracy": summarize_scalar_list([run["baseline"]["accuracy"] for run in runs]),
        "hadmc_accuracy": summarize_scalar_list([run["hadmc"]["accuracy"] for run in runs]),
        "hadmc_latency_ms": summarize_scalar_list([run["hadmc"]["latency_ms"] for run in runs]),
        "compression_ratio": summarize_scalar_list([
            run["hadmc"]["compression_info"]["compression_ratio"] for run in runs
        ]),
        "speedup": summarize_scalar_list([
            run["hadmc"]["compression_info"]["speedup"] for run in runs
        ]),
    }
    results = {
        "seeds": SEEDS,
        "runs": runs,
        "summary": summary,
    }
    save_json("variance/variance_results.json", results)
    return results


def run_decomposition_experiment(reference_seed):
    print("\n" + "=" * 70)
    print("SUPPLEMENTARY EXPERIMENT 2: Compression vs Deployment Decomposition")
    print("=" * 70)

    train_loader, test_loader = create_neudet_loaders(reference_seed)
    baseline_model, compression_only_model, metadata = load_reference_models(reference_seed)
    baseline_model = baseline_model.to(core.device)
    compression_only_model = compression_only_model.to(core.device)

    variants = OrderedDict()
    variants["baseline_fp32"] = baseline_model
    variants["baseline_deployment_only"] = core.fuse_conv_bn(copy.deepcopy(baseline_model))
    variants["compressed_only"] = compression_only_model
    variants["compressed_plus_deployment"] = core.quantize_model_int8(
        core.fuse_conv_bn(copy.deepcopy(compression_only_model))
    )

    results = OrderedDict()
    for name, model in variants.items():
        results[name] = core.evaluate_model(model, test_loader)
        if "deployment" in name:
            results[name]["effective_size_mb"] = round(results[name]["model_size_mb"] / 4, 4)
        qinfo = getattr(model, "_quantization_info", None)
        if qinfo is not None:
            results[name]["quantization_info"] = qinfo
        pinfo = getattr(model, "_pruning_info", None)
        if pinfo is not None:
            results[name]["pruning_info"] = pinfo

    # Backfill pruning_info for compressed variants: state_dict load doesn't
    # carry Python attributes, so propagate from the seed's metadata payload.
    meta_pinfo = (
        metadata.get("hadmc_results", {}).get("pruning_info")
        if isinstance(metadata, dict)
        else None
    )
    if meta_pinfo is not None:
        for cname in ("compressed_only", "compressed_plus_deployment"):
            if cname in results and "pruning_info" not in results[cname]:
                results[cname]["pruning_info"] = meta_pinfo

    base_lat = results["baseline_fp32"]["latency_ms"]
    decomposition = {
        "runtime_only_gain": round(base_lat / results["baseline_deployment_only"]["latency_ms"], 6),
        "compression_only_gain": round(base_lat / results["compressed_only"]["latency_ms"], 6),
        "combined_gain": round(base_lat / results["compressed_plus_deployment"]["latency_ms"], 6),
    }
    decomposition["interaction_gain"] = round(
        decomposition["combined_gain"] / (
            decomposition["runtime_only_gain"] * decomposition["compression_only_gain"]
        ),
        6,
    )

    payload = {
        "reference_seed": reference_seed,
        "source_metadata": metadata,
        "variants": results,
        "decomposition": decomposition,
    }
    save_json("decomposition/decomposition_results.json", payload)
    return payload


def collect_operator_signatures(model, sample_input):
    records = []
    hooks = []

    def register(name, module):
        def hook(_module, inputs, _outputs):
            tensor = inputs[0]
            if isinstance(_module, nn.Conv2d):
                signature = {
                    "op": "conv2d",
                    "name": name,
                    "batch_size": int(tensor.shape[0]),
                    "input_shape": [int(tensor.shape[-2]), int(tensor.shape[-1])],
                    "in_channels": int(_module.in_channels),
                    "out_channels": int(_module.out_channels),
                    "kernel_size": list(_module.kernel_size),
                    "stride": list(_module.stride),
                    "padding": list(_module.padding),
                    "dilation": list(_module.dilation),
                    "groups": int(_module.groups),
                    "bias": bool(_module.bias is not None),
                }
            elif isinstance(_module, nn.Linear):
                signature = {
                    "op": "linear",
                    "name": name,
                    "batch_size": int(tensor.shape[0]),
                    "in_features": int(_module.in_features),
                    "out_features": int(_module.out_features),
                    "bias": bool(_module.bias is not None),
                }
            else:
                return
            records.append(signature)

        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(register(name, module)))

    model = model.to(core.device)
    model.eval()
    with torch.no_grad():
        _ = model(sample_input.to(core.device))

    for hook in hooks:
        hook.remove()
    return records


def signature_key(signature):
    return json.dumps(signature, sort_keys=True)


def benchmark_signature(signature):
    if signature["op"] == "conv2d":
        layer = nn.Conv2d(
            signature["in_channels"],
            signature["out_channels"],
            tuple(signature["kernel_size"]),
            stride=tuple(signature["stride"]),
            padding=tuple(signature["padding"]),
            dilation=tuple(signature["dilation"]),
            groups=signature["groups"],
            bias=signature["bias"],
        )
        dummy = torch.randn(
            signature["batch_size"],
            signature["in_channels"],
            signature["input_shape"][0],
            signature["input_shape"][1],
        )
    else:
        layer = nn.Linear(
            signature["in_features"],
            signature["out_features"],
            bias=signature["bias"],
        )
        dummy = torch.randn(signature["batch_size"], signature["in_features"])

    layer = layer.to(core.device)
    layer.eval()
    dummy = dummy.to(core.device)

    with torch.no_grad():
        for _ in range(ARGS.lut_warmup):
            _ = layer(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    latencies = []
    with torch.no_grad():
        for _ in range(ARGS.lut_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = core.time.perf_counter()
            _ = layer(dummy)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = core.time.perf_counter()
            latencies.append((end - start) * 1000.0)

    return {
        "latency_ms": round(float(np.mean(latencies)), 6),
        "latency_std_ms": round(float(np.std(latencies)), 6),
        "samples": len(latencies),
    }


def compute_spearman(x_values, y_values):
    def rank(values):
        order = np.argsort(values)
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(len(values), dtype=float)
        return ranks

    x_ranks = rank(np.asarray(x_values, dtype=float))
    y_ranks = rank(np.asarray(y_values, dtype=float))
    return float(np.corrcoef(x_ranks, y_ranks)[0, 1])


def _summarize_predictions(predicted, measured):
    predicted = np.asarray(predicted, dtype=float)
    measured = np.asarray(measured, dtype=float)
    errors = np.abs(predicted - measured) / np.maximum(measured, 1e-9) * 100.0
    return {
        "candidate_count": int(predicted.size),
        "mape": round(float(np.mean(errors)), 6),
        "median_abs_pct_error": round(float(np.median(errors)), 6),
        "p95_abs_pct_error": round(float(np.percentile(errors, 95)), 6),
        "max_abs_pct_error": round(float(np.max(errors)), 6),
        "pearson": round(float(np.corrcoef(predicted, measured)[0, 1]), 6),
        "spearman": round(compute_spearman(predicted, measured), 6),
    }, errors.tolist()


def calibrate_lut_predictions(raw_predicted, num_ops, measured):
    """Honest per-platform LUT calibration.

    Fits a 3-parameter affine model
        measured ≈ α * raw_predicted + β * num_ops + γ
    via ordinary least squares, then evaluates with leave-one-out so every
    candidate is predicted by parameters fitted **without** seeing it. This
    mirrors per-platform calibration in nn-Meter / MAPLE-style LUT systems
    and does not modify the underlying micro-benchmarks.
    """
    raw = np.asarray(raw_predicted, dtype=float)
    ops = np.asarray(num_ops, dtype=float)
    meas = np.asarray(measured, dtype=float)
    n = raw.size

    def _fit(idx_mask):
        X = np.stack([raw[idx_mask], ops[idx_mask], np.ones(idx_mask.sum())], axis=1)
        y = meas[idx_mask]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return coef  # [alpha, beta, gamma]

    # Full-fit coefficients (reported for transparency, NOT used to score predictions).
    coef_full = _fit(np.ones(n, dtype=bool))

    # Leave-one-out predictions.
    loo_pred = np.zeros(n, dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        alpha, beta, gamma = _fit(mask)
        loo_pred[i] = alpha * raw[i] + beta * ops[i] + gamma

    return {
        "method": "affine_per_platform_lut_calibration_loo",
        "features": ["raw_predicted_ms", "num_operator_calls", "1"],
        "full_fit_coefficients": {
            "alpha_raw_predicted": float(coef_full[0]),
            "beta_num_ops": float(coef_full[1]),
            "gamma_intercept": float(coef_full[2]),
        },
        "loo_predicted_ms": [round(float(v), 6) for v in loo_pred],
        "note": (
            "Per-candidate prediction comes from a least-squares fit on the "
            "other n-1 candidates only; the held-out candidate never "
            "influences its own predicted value."
        ),
    }


def _measure_model_latency_at_batch(model, input_tensor, num_warmup, num_runs):
    """Measure whole-model wall-clock latency at the exact batch size used to
    build the per-operator LUT. The supplementary core `evaluate_model` always
    times at batch=1 (deployment-style), but the LUT is built and compared at
    `--lut-batch-size`, so we re-measure here with matching conditions.
    """
    model = model.to(core.device)
    model.eval()
    inp = input_tensor.to(core.device)
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(inp)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = core.time.perf_counter()
            _ = model(inp)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = core.time.perf_counter()
            latencies.append((end - start) * 1000.0)
    return {
        "latency_ms": round(float(np.mean(latencies)), 6),
        "latency_std_ms": round(float(np.std(latencies)), 6),
        "samples": len(latencies),
    }


def run_lut_validation(reference_seed):
    print("\n" + "=" * 70)
    print("SUPPLEMENTARY EXPERIMENT 3: Whole-Model LUT Validation")
    print("=" * 70)

    _, test_loader = create_neudet_loaders(reference_seed)
    baseline_model, compression_only_model, _ = load_reference_models(reference_seed)

    candidates = OrderedDict()
    candidates["baseline_fp32"] = baseline_model
    candidates["baseline_fused"] = core.fuse_conv_bn(copy.deepcopy(baseline_model))
    candidates["baseline_int8"] = core.quantize_model_int8(core.fuse_conv_bn(copy.deepcopy(baseline_model)))
    candidates["reference_compressed"] = compression_only_model
    candidates["reference_full_hadmc"] = core.quantize_model_int8(
        core.fuse_conv_bn(copy.deepcopy(compression_only_model))
    )

    for ratio in LUT_PRUNE_RATIOS:
        pruned = core.smart_structural_prune(baseline_model, prune_ratio=ratio)
        ratio_tag = str(ratio).replace(".", "p")
        candidates[f"pruned_{ratio_tag}"] = pruned
        candidates[f"pruned_fused_{ratio_tag}"] = core.fuse_conv_bn(copy.deepcopy(pruned))
        candidates[f"pruned_int8_{ratio_tag}"] = core.quantize_model_int8(
            core.fuse_conv_bn(copy.deepcopy(pruned))
        )

    # Build a sample input at the LUT batch size so the operator signatures
    # carry batch_size=B and benchmark_signature() reproduces the exact tensor
    # shape that the full model sees. Pull as many samples as available from
    # the loader before padding with the last sample if needed.
    lut_batch = max(1, int(getattr(ARGS, "lut_batch_size", 32)))
    pool = []
    for x, _ in test_loader:
        pool.append(x)
        if sum(t.shape[0] for t in pool) >= lut_batch:
            break
    pool_tensor = torch.cat(pool, dim=0)
    if pool_tensor.shape[0] >= lut_batch:
        sample_input = pool_tensor[:lut_batch].clone()
    else:
        # Loader is smaller than lut_batch — repeat to fill.
        repeats = (lut_batch + pool_tensor.shape[0] - 1) // pool_tensor.shape[0]
        sample_input = pool_tensor.repeat(repeats, 1, 1, 1)[:lut_batch].clone()

    lut_cache = {}
    candidate_results = []

    for candidate_name, model in candidates.items():
        signatures = collect_operator_signatures(model, sample_input)
        predicted_ms = 0.0
        for signature in signatures:
            cache_key = signature_key(signature)
            if cache_key not in lut_cache:
                lut_cache[cache_key] = {
                    "signature": signature,
                    "benchmark": benchmark_signature(signature),
                }
            predicted_ms += lut_cache[cache_key]["benchmark"]["latency_ms"]

        measured = _measure_model_latency_at_batch(
            model, sample_input,
            num_warmup=ARGS.lut_warmup, num_runs=ARGS.lut_runs,
        )
        measured_ms = measured["latency_ms"]
        abs_pct_error = abs(predicted_ms - measured_ms) / max(measured_ms, 1e-9) * 100.0

        # Also report acc / params via the standard evaluator (batch=1 latency
        # discarded here; only acc/params are used for context).
        eval_ctx = core.evaluate_model(model, test_loader, num_warmup=2, num_runs=2)
        candidate_results.append({
            "candidate": candidate_name,
            "num_operator_calls": len(signatures),
            "predicted_latency_ms": round(predicted_ms, 6),
            "measured_latency_ms": measured_ms,
            "measured_latency_std_ms": measured["latency_std_ms"],
            "abs_pct_error": round(abs_pct_error, 6),
            "accuracy": eval_ctx["accuracy"],
            "num_params": eval_ctx["num_params"],
        })
        save_json("lut_validation/partial_results.json", {"candidates": candidate_results})

    predicted = [item["predicted_latency_ms"] for item in candidate_results]
    measured = [item["measured_latency_ms"] for item in candidate_results]
    num_ops = [item["num_operator_calls"] for item in candidate_results]

    raw_summary, raw_errors = _summarize_predictions(predicted, measured)

    calibration = calibrate_lut_predictions(predicted, num_ops, measured)
    calibrated_summary, cal_errors = _summarize_predictions(
        calibration["loo_predicted_ms"], measured
    )

    # Attach per-candidate calibrated predictions and error to the records so
    # downstream consumers can verify the calibration without re-running.
    for i, item in enumerate(candidate_results):
        item["raw_abs_pct_error"] = round(raw_errors[i], 6)
        item["calibrated_predicted_ms"] = calibration["loo_predicted_ms"][i]
        item["calibrated_abs_pct_error"] = round(cal_errors[i], 6)

    summary = {
        "lut_batch_size": lut_batch,
        "raw": raw_summary,
        "calibrated_loo": calibrated_summary,
        # Backwards-compatible top-level keys (= raw, the uncalibrated metric).
        "candidate_count": raw_summary["candidate_count"],
        "mape": raw_summary["mape"],
        "median_abs_pct_error": raw_summary["median_abs_pct_error"],
        "p95_abs_pct_error": raw_summary["p95_abs_pct_error"],
        "pearson": raw_summary["pearson"],
        "spearman": raw_summary["spearman"],
    }
    payload = {
        "reference_seed": reference_seed,
        "lut_batch_size": lut_batch,
        "measurement_protocol": {
            "model_latency": "_measure_model_latency_at_batch with matching lut_batch_size",
            "operator_latency": "benchmark_signature with batch_size from operator signature",
            "warmup_iterations": ARGS.lut_warmup,
            "timed_iterations": ARGS.lut_runs,
            "rationale": (
                "At batch=1 on this platform, HIP/CUDA kernel-launch overhead "
                "dominates per-operator latency, breaking the additive LUT "
                "model. Measuring at a deployment-style batch size lets "
                "compute dominate launch overhead so the per-operator "
                "additive model becomes physically meaningful."
            ),
        },
        "summary": summary,
        "calibration": calibration,
        "candidates": candidate_results,
        "operator_lut": list(lut_cache.values()),
    }
    save_json("lut_validation/lut_validation_results.json", payload)
    return payload


def run_public_benchmark():
    print("\n" + "=" * 70)
    print("SUPPLEMENTARY EXPERIMENT 4: Public Benchmark")
    print("=" * 70)

    if ARGS.skip_public_benchmark:
        payload = {
            "skipped": True,
            "reason": "Skipped by --skip-public-benchmark",
        }
        save_json("public_benchmark/public_benchmark_results.json", payload)
        return payload

    try:
        from torchvision import datasets, transforms
    except Exception as exc:
        payload = {
            "skipped": True,
            "reason": f"torchvision unavailable: {exc}",
        }
        save_json("public_benchmark/public_benchmark_results.json", payload)
        return payload

    data_root = os.path.join(RESULTS_DIR, "datasets")
    transform = transforms.Compose([transforms.ToTensor()])

    try:
        train_ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    except Exception as exc:
        payload = {
            "skipped": True,
            "reason": f"CIFAR-10 download failed: {exc}",
        }
        save_json("public_benchmark/public_benchmark_results.json", payload)
        return payload

    if ARGS.public_train_subset > 0 and ARGS.public_train_subset < len(train_ds):
        train_ds = Subset(train_ds, list(range(ARGS.public_train_subset)))
    if ARGS.public_test_subset > 0 and ARGS.public_test_subset < len(test_ds):
        test_ds = Subset(test_ds, list(range(ARGS.public_test_subset)))

    train_loader = make_loader(train_ds, shuffle=True)
    test_loader = make_loader(test_ds, shuffle=False)

    try:
        set_seed(SEEDS[0])
        baseline = core.ResNet18Small(num_classes=10, base_width=64)
        baseline, train_losses, train_accs = core.train_model(
            baseline,
            train_loader,
            num_epochs=ARGS.public_epochs,
            lr=0.01,
        )
        baseline_results = core.evaluate_model(baseline, test_loader)
        baseline_results["train_losses"] = train_losses
        baseline_results["train_accs"] = train_accs

        compression_only, _, full_model, info = build_hadmc_variants(
            baseline,
            train_loader,
            prune_ratio=ARGS.prune_ratio,
        )
        compressed_results = augment_hadmc_results(
            baseline_results,
            core.evaluate_model(full_model, test_loader),
            info,
        )

        payload = {
            "skipped": False,
            "dataset": "CIFAR-10",
            "train_size": len(train_ds),
            "test_size": len(test_ds),
            "baseline": baseline_results,
            "compressed": compressed_results,
            "compression_only_base_width": compression_only.conv1.out_channels,
        }
    except Exception as exc:
        payload = {
            "skipped": True,
            "reason": f"CIFAR-10 benchmark failed during training/evaluation: {exc}",
        }
    save_json("public_benchmark/public_benchmark_results.json", payload)
    return payload


def main():
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "platform_tag": ARGS.platform_tag,
        "device": str(core.device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "hip": getattr(torch.version, "hip", None),
        "pytorch": torch.__version__,
        "git_commit": get_git_commit(),
        "seeds": SEEDS,
        "results_dir": RESULTS_DIR,
    }
    metadata_path = save_json("RUN_METADATA.json", metadata)
    print(f"  Supplementary run metadata saved to: {metadata_path}")

    variance_results = run_variance_experiment()
    reference_seed = SEEDS[0]
    decomposition_results = run_decomposition_experiment(reference_seed)
    lut_results = run_lut_validation(reference_seed)
    public_benchmark_results = run_public_benchmark()

    all_results = {
        "metadata": metadata,
        "variance": variance_results,
        "decomposition": decomposition_results,
        "lut_validation": lut_results,
        "public_benchmark": public_benchmark_results,
    }
    summary_path = save_json("TPDS_SUPPLEMENTARY_RESULTS.json", all_results)
    print("\n" + "=" * 70)
    print("TPDS supplementary experiments completed")
    print(f"Results saved to: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()