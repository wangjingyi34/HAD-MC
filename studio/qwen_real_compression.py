#!/usr/bin/env python3
"""Run one real Qwen checkpoint compression policy on a DCU.

This adapter intentionally keeps the evidence boundary narrow: it loads an
official Qwen checkpoint, applies structured SwiGLU MLP pruning, symmetric
weight-only quantization and soft-logit calibration distillation, then measures
DCU forward latency plus output agreement. It does not claim benchmark
accuracy, an INT kernel, full fine tuning, or a trained PPO policy.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import time
from pathlib import Path

import torch
from torch import nn
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS = (
    "请用一句话解释结构化模型压缩。",
    "金融数据湖需要哪些可追溯性能力？",
    "Write a concise description of hardware-aware inference.",
)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def progress(path: Path, phase: str, **extra: object) -> None:
    payload = {"phase": phase, "updated_at_utc": utc_now(), **extra}
    write_json(path, payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def parameter_bytes(model: nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def cuda_sync() -> None:
    torch.cuda.synchronize()


def benchmark(model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, float]:
    model.eval()
    with torch.inference_mode():
        for _ in range(1):
            output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        cuda_sync()
        started = time.perf_counter()
        for _ in range(3):
            output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        cuda_sync()
    return output, (time.perf_counter() - started) * 1000.0 / 3.0


def calibration_logits(model: nn.Module, tokenizer: AutoTokenizer, device: torch.device) -> list[torch.Tensor]:
    outputs: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for prompt_text in PROMPTS:
            batch = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=96)
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs.append(model(**batch, use_cache=False).logits.float().cpu())
    return outputs


def round_width(value: int, multiple: int = 256) -> int:
    return max(multiple, (value // multiple) * multiple)


def normalized(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    return values / values.mean().clamp_min(1e-8)


def collect_mlp_sensitivity(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> dict[int, dict[str, torch.Tensor | float]]:
    """Measure Qwen MLP channel sensitivity from weights and calibration activations."""
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("The loaded checkpoint does not expose Qwen model.layers")
    activation_sums: dict[int, torch.Tensor] = {}
    activation_counts: dict[int, int] = {}
    handles = []

    def capture(layer_index: int):
        def hook(_module, _inputs, output):
            sample = output.detach().float().abs().mean(dim=(0, 1)).cpu()
            activation_sums[layer_index] = activation_sums.get(layer_index, torch.zeros_like(sample)) + sample
            activation_counts[layer_index] = activation_counts.get(layer_index, 0) + 1
        return hook

    for layer_index, layer in enumerate(layers):
        gate = getattr(getattr(layer, "mlp", None), "gate_proj", None)
        if not isinstance(gate, nn.Linear):
            raise RuntimeError(f"Layer {layer_index} does not expose Qwen gate_proj")
        handles.append(gate.register_forward_hook(capture(layer_index)))
    try:
        calibration_logits(model, tokenizer, device)
    finally:
        for handle in handles:
            handle.remove()

    sensitivity: dict[int, dict[str, torch.Tensor | float]] = {}
    raw_layer_scores: list[float] = []
    for layer_index, layer in enumerate(layers):
        mlp = layer.mlp
        weight_score = (
            mlp.gate_proj.weight.detach().float().abs().mean(dim=1).cpu()
            + mlp.up_proj.weight.detach().float().abs().mean(dim=1).cpu()
        )
        activation_score = activation_sums[layer_index] / max(activation_counts[layer_index], 1)
        channel_score = 0.55 * normalized(weight_score) + 0.45 * normalized(activation_score)
        raw_layer_score = float(
            torch.log1p(activation_score.mean()) * weight_score.mean()
        )
        raw_layer_scores.append(raw_layer_score)
        sensitivity[layer_index] = {
            "channels": channel_score,
            "layer_score": raw_layer_score,
            "activation_mean": float(activation_score.mean()),
            "weight_mean": float(weight_score.mean()),
        }
    mean_layer_score = sum(raw_layer_scores) / max(len(raw_layer_scores), 1)
    for values in sensitivity.values():
        values["layer_score"] = float(values["layer_score"]) / max(mean_layer_score, 1e-12)
    return sensitivity


def replacement_linear(source: nn.Linear, weight: torch.Tensor) -> nn.Linear:
    target = nn.Linear(
        weight.shape[1],
        weight.shape[0],
        bias=source.bias is not None,
        device=source.weight.device,
        dtype=source.weight.dtype,
    )
    with torch.no_grad():
        target.weight.copy_(weight)
        if source.bias is not None and target.bias is not None:
            target.bias.copy_(source.bias.detach()[: weight.shape[0]])
    return target


def structured_prune_qwen_mlp(
    model: nn.Module,
    prune_ratio: float,
    sensitivity: dict[int, dict[str, torch.Tensor | float]],
) -> dict:
    """Prune Qwen SwiGLU intermediate channels while keeping every block valid."""
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("The loaded checkpoint does not expose Qwen model.layers")
    selected_widths: list[int] = []
    with torch.no_grad():
        for layer_index, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            gate = getattr(mlp, "gate_proj", None)
            up = getattr(mlp, "up_proj", None)
            down = getattr(mlp, "down_proj", None)
            if not all(isinstance(module, nn.Linear) for module in (gate, up, down)):
                raise RuntimeError(f"Layer {layer_index} does not expose Qwen SwiGLU linear projections")
            original_width = gate.out_features
            retained_width = round_width(int(original_width * (1.0 - prune_ratio)))
            retained_width = min(retained_width, original_width)
            importance = sensitivity[layer_index]["channels"].to(gate.weight.device)
            indices = importance.topk(retained_width, largest=True).indices.sort().values
            mlp.gate_proj = replacement_linear(gate, gate.weight.detach().index_select(0, indices))
            mlp.up_proj = replacement_linear(up, up.weight.detach().index_select(0, indices))
            mlp.down_proj = replacement_linear(down, down.weight.detach().index_select(1, indices))
            selected_widths.append(retained_width)
    return {
        "method": "activation_weight_channel_sensitivity",
        "layers": len(selected_widths),
        "retained_intermediate_widths": selected_widths,
        "alignment": 256,
        "sensitivity_summary": [
            {
                "layer": layer_index,
                "score": round(float(values["layer_score"]), 7),
                "activation_mean": round(float(values["activation_mean"]), 7),
                "weight_mean": round(float(values["weight_mean"]), 7),
            }
            for layer_index, values in sensitivity.items()
        ],
    }


def weight_only_quantize(
    model: nn.Module,
    bits: int,
    sensitivity: dict[int, dict[str, torch.Tensor | float]],
    int4_layer_fraction: float,
) -> dict:
    """Apply sensitivity-guided symmetric quantization and dequantize for DCU FP16 kernels."""
    if bits not in (4, 8):
        raise ValueError("only INT4 and INT8 weight-only calibration are supported")
    if not 0.0 <= int4_layer_fraction <= 1.0:
        raise ValueError("int4_layer_fraction must be between 0 and 1")
    layer_order = sorted(sensitivity, key=lambda index: float(sensitivity[index]["layer_score"]))
    int4_layers = set(layer_order[: round(len(layer_order) * int4_layer_fraction)])
    quantized_bytes = 0
    scale_bytes = 0
    replaced_fp16_bytes = 0
    layers = 0
    bit_histogram = {"4": 0, "8": 0}
    weighted_mse = 0.0
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            layer_index = None
            if name.startswith("model.layers."):
                try:
                    layer_index = int(name.split(".")[2])
                except (IndexError, ValueError):
                    layer_index = None
            selected_bits = 4 if layer_index in int4_layers else bits
            qmax = float((1 << (selected_bits - 1)) - 1)
            squared_error = 0.0
            scale_count = 0
            for start in range(0, module.weight.shape[0], 1024):
                end = min(start + 1024, module.weight.shape[0])
                weight = module.weight[start:end].detach().float()
                scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
                codes = torch.round(weight / scale).clamp(-qmax, qmax)
                dequantized = codes * scale
                squared_error += torch.sum((weight - dequantized) ** 2).item()
                scale_count += scale.numel()
                module.weight[start:end].copy_(dequantized.to(module.weight.dtype))
                del weight, scale, codes, dequantized
            error = squared_error / max(module.weight.numel(), 1)
            sensitivity_weight = (
                float(sensitivity[layer_index]["layer_score"])
                if layer_index in sensitivity
                else 1.0
            )
            weighted_mse += error * sensitivity_weight
            quantized_bytes += int(module.weight.numel() * selected_bits / 8)
            scale_bytes += scale_count * 4
            replaced_fp16_bytes += module.weight.numel() * module.weight.element_size()
            bit_histogram[str(selected_bits)] += 1
            layers += 1
    return {
        "method": "sensitivity_guided_symmetric_per_channel_weight_only",
        "default_bits": bits,
        "int4_layer_fraction": int4_layer_fraction,
        "int4_transformer_layers": sorted(int4_layers),
        "bit_histogram": bit_histogram,
        "layers": layers,
        "replaced_fp16_storage_bytes": replaced_fp16_bytes,
        "logical_storage_bytes": quantized_bytes + scale_bytes,
        "sensitivity_weighted_mse": weighted_mse / max(layers, 1),
        "kernel": "FP16 dequantized execution; no INT kernel acceleration claim",
    }


def calibrate_distillation(model: nn.Module, teacher_outputs: list[torch.Tensor], student_outputs: list[torch.Tensor], temperature: float, alpha: float) -> dict:
    """Fit deployable per-token output scales using real teacher/student soft-logit KD."""
    teacher = torch.cat([item[:, -1, :].float() for item in teacher_outputs], dim=0)
    student = torch.cat([item[:, -1, :].float() for item in student_outputs], dim=0)
    scale = torch.ones(student.shape[-1], requires_grad=True)
    optimizer = torch.optim.Adam([scale], lr=0.02)
    before = None
    loss = torch.tensor(0.0)
    for _ in range(16):
        optimizer.zero_grad()
        loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student * scale.unsqueeze(0) / temperature, dim=-1),
            torch.nn.functional.softmax(teacher / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature * temperature)
        if before is None:
            before = float(loss.detach())
        (alpha * loss).backward()
        optimizer.step()
        with torch.no_grad():
            scale.clamp_(0.85, 1.15)
    with torch.no_grad():
        for start in range(0, model.lm_head.weight.shape[0], 4096):
            end = min(start + 4096, model.lm_head.weight.shape[0])
            token_scale = scale[start:end].to(model.lm_head.weight.device, dtype=model.lm_head.weight.dtype)
            model.lm_head.weight[start:end].mul_(token_scale.unsqueeze(1))
    return {
        "method": "per_token_soft_logit_calibration_distillation",
        "temperature": temperature,
        "alpha": alpha,
        "steps": 16,
        "loss_before": before,
        "loss_after": float(loss.detach()),
        "token_scale_min": float(scale.detach().min()),
        "token_scale_max": float(scale.detach().max()),
        "token_scale_mean": float(scale.detach().mean()),
    }


def ensure_checkpoint(model_id: str, model_dir: Path, progress_path: Path) -> Path:
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = model_dir.parent / ".qlight-qwen-download.lock"
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if (model_dir / "config.json").exists() and any(model_dir.glob("*.safetensors")):
            progress(progress_path, "checkpoint_ready", detail="reusing verified local Qwen checkpoint")
            return model_dir
        if model_dir.exists():
            shutil.rmtree(model_dir)
        progress(progress_path, "downloading_checkpoint", detail=f"downloading {model_id} to shared model cache")
        snapshot_download(
            repo_id=model_id,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        if not (model_dir / "config.json").exists() or not any(model_dir.glob("*.safetensors")):
            raise RuntimeError("checkpoint download finished without config.json and safetensors weights")
        progress(progress_path, "checkpoint_ready", detail="official checkpoint is present in shared model cache")
    return model_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-cache", default="/models/qwen2.5-7b-instruct")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--prune-ratio", required=True, type=float)
    parser.add_argument("--quality-weight", required=True, type=float)
    parser.add_argument("--storage-weight", required=True, type=float)
    parser.add_argument("--latency-weight", required=True, type=float)
    parser.add_argument("--quant-bits", type=int, default=8)
    parser.add_argument("--int4-layer-fraction", type=float, default=0.0)
    parser.add_argument("--distillation-temperature", type=float, default=4.0)
    parser.add_argument("--distillation-alpha", type=float, default=0.5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    progress_path = output_dir / "QLIGHT_REAL_QWEN_PROGRESS.json"
    result_path = output_dir / "QLIGHT_REAL_QWEN_CANDIDATE.json"
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("DCU runtime is unavailable to PyTorch")
        device = torch.device("cuda:0")
        progress(progress_path, "starting", candidate_id=args.candidate_id, model_id=args.model_id)
        checkpoint_dir = ensure_checkpoint(args.model_id, Path(args.model_cache), progress_path)
        progress(progress_path, "loading_model", detail="loading the official checkpoint into a DCU process")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_dir,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        ).to(device)
        model.eval()
        cuda_sync()
        baseline_params = parameter_count(model)
        baseline_bytes = parameter_bytes(model)
        input_ids = torch.randint(0, model.config.vocab_size, (1, 32), device=device)
        attention_mask = torch.ones_like(input_ids)
        progress(progress_path, "baseline_measured", parameters=baseline_params, storage_bytes=baseline_bytes)
        baseline_output, baseline_latency = benchmark(model, input_ids, attention_mask)
        reference_outputs = calibration_logits(model, tokenizer, device)
        progress(progress_path, "scoring_sensitivity", detail="measuring Qwen MLP activation and weight sensitivity")
        sensitivity = collect_mlp_sensitivity(model, tokenizer, device)
        progress(progress_path, "scoring_structure", detail="ranking actual Qwen SwiGLU channels by measured sensitivity")
        pruning_info = structured_prune_qwen_mlp(model, args.prune_ratio, sensitivity)
        cuda_sync()
        progress(progress_path, "structured_pruning_complete", pruning=pruning_info)
        progress(progress_path, "quantizing_weights", bits=args.quant_bits)
        quantization_info = weight_only_quantize(model, args.quant_bits, sensitivity, args.int4_layer_fraction)
        cuda_sync()
        candidate_outputs = calibration_logits(model, tokenizer, device)
        progress(progress_path, "distilling_calibration", temperature=args.distillation_temperature, alpha=args.distillation_alpha)
        distillation_info = calibrate_distillation(model, reference_outputs, candidate_outputs, args.distillation_temperature, args.distillation_alpha)
        candidate_output, candidate_latency = benchmark(model, input_ids, attention_mask)
        candidate_outputs = calibration_logits(model, tokenizer, device)
        cosine_scores = [
            torch.nn.functional.cosine_similarity(reference.flatten(), candidate.flatten(), dim=0).item()
            for reference, candidate in zip(reference_outputs, candidate_outputs)
        ]
        candidate_params = parameter_count(model)
        candidate_bytes = parameter_bytes(model)
        effective_storage_bytes = (
            candidate_bytes
            - quantization_info["replaced_fp16_storage_bytes"]
            + quantization_info["logical_storage_bytes"]
        )
        structural_reduction = 1.0 - candidate_params / baseline_params
        storage_reduction = 1.0 - candidate_bytes / baseline_bytes
        effective_storage_reduction = 1.0 - effective_storage_bytes / baseline_bytes
        speedup = baseline_latency / candidate_latency
        quality_retention = sum(cosine_scores) / len(cosine_scores)
        calibration_precision_loss = 1.0 - quality_retention
        threshold_penalty = max(0.0, 0.98 - quality_retention) * 12.0
        reward = (
            args.quality_weight * quality_retention
            + args.storage_weight * effective_storage_reduction
            + args.latency_weight * max(0.0, min(speedup - 1.0, 1.0))
            - threshold_penalty
        )
        result = {
            "product": "QLight",
            "experiment": "real_qwen_hadmc_structured_compression",
            "evidence_scope": {
                "checkpoint": "official Qwen checkpoint loaded from Hugging Face",
                "hardware": "measured on one allocated Hygon DCU through the DTK/HYHAL container",
                "sensitivity": "real activation and weight sensitivity measured on calibration prompts",
                "compression": "real sensitivity-ranked structured SwiGLU MLP pruning on Qwen Transformer modules",
                "quality": "calibration logit agreement; this is not a benchmark accuracy claim",
                "quantization": "real weight-only quantization/dequantization; no INT kernel acceleration claim is made",
                "distillation": "real soft-logit calibration distillation on calibration prompts; not full fine tuning",
            },
            "model": {
                "id": args.model_id,
                "config_model_type": getattr(model.config, "model_type", None),
                "hidden_size": getattr(model.config, "hidden_size", None),
                "layers": getattr(model.config, "num_hidden_layers", None),
                "baseline_parameters": baseline_params,
                "candidate_parameters": candidate_params,
                "baseline_fp16_storage_bytes": baseline_bytes,
                "candidate_fp16_storage_bytes": candidate_bytes,
                "candidate_effective_storage_bytes": effective_storage_bytes,
            },
            "candidate": {
                "candidate_id": args.candidate_id,
                "target_prune_ratio": args.prune_ratio,
                "reward_weights": {
                    "quality": args.quality_weight,
                    "storage": args.storage_weight,
                    "latency": args.latency_weight,
                },
                "structural_compression_pct": round(structural_reduction * 100.0, 3),
                "fp16_storage_reduction_pct": round(storage_reduction * 100.0, 3),
                "effective_storage_reduction_pct": round(effective_storage_reduction * 100.0, 3),
                "calibration_logit_cosine": round(quality_retention, 6),
                "calibration_retention_pct": round(quality_retention * 100.0, 3),
                "calibration_precision_loss_pct": round(calibration_precision_loss * 100.0, 3),
                "baseline_latency_ms": round(baseline_latency, 3),
                "candidate_latency_ms": round(candidate_latency, 3),
                "speedup": round(speedup, 4),
                "latency_reduction_pct": round((1.0 - candidate_latency / baseline_latency) * 100.0, 3),
                "reward": round(reward, 7),
                "retention_threshold_pct": 98.0,
                "retention_threshold_met": quality_retention >= 0.98,
                "threshold_penalty": round(threshold_penalty, 7),
                "pruning": pruning_info,
                "quantization": quantization_info,
                "distillation": distillation_info,
            },
            "runtime": {
                "device": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "completed_at_utc": utc_now(),
            },
        }
        write_json(result_path, result)
        progress(progress_path, "completed", result_file=result_path.name, candidate=result["candidate"])
    except Exception as exc:
        progress(progress_path, "failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
