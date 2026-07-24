#!/usr/bin/env python3
"""Run one real Qwen checkpoint compression policy on a DCU.

This adapter intentionally keeps the evidence boundary narrow: it loads an
official Qwen checkpoint, performs structured SwiGLU MLP pruning on the real
Transformer modules, and measures DCU forward latency plus output agreement.
It does not claim a benchmark accuracy result, an INT8 kernel, or a trained
PPO policy when those components have not run in this campaign.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
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


def structured_prune_qwen_mlp(model: nn.Module, prune_ratio: float) -> dict:
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
            importance = gate.weight.detach().float().abs().mean(dim=1) + up.weight.detach().float().abs().mean(dim=1)
            indices = importance.topk(retained_width, largest=True).indices.sort().values
            mlp.gate_proj = replacement_linear(gate, gate.weight.detach().index_select(0, indices))
            mlp.up_proj = replacement_linear(up, up.weight.detach().index_select(0, indices))
            mlp.down_proj = replacement_linear(down, down.weight.detach().index_select(1, indices))
            selected_widths.append(retained_width)
    return {"layers": len(selected_widths), "retained_intermediate_widths": selected_widths, "alignment": 256}


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
        progress(progress_path, "scoring_structure", detail="ranking actual Qwen SwiGLU intermediate channels")
        pruning_info = structured_prune_qwen_mlp(model, args.prune_ratio)
        cuda_sync()
        progress(progress_path, "structured_pruning_complete", pruning=pruning_info)
        candidate_output, candidate_latency = benchmark(model, input_ids, attention_mask)
        candidate_outputs = calibration_logits(model, tokenizer, device)
        cosine_scores = [
            torch.nn.functional.cosine_similarity(reference.flatten(), candidate.flatten(), dim=0).item()
            for reference, candidate in zip(reference_outputs, candidate_outputs)
        ]
        candidate_params = parameter_count(model)
        candidate_bytes = parameter_bytes(model)
        structural_reduction = 1.0 - candidate_params / baseline_params
        storage_reduction = 1.0 - candidate_bytes / baseline_bytes
        speedup = baseline_latency / candidate_latency
        quality_retention = sum(cosine_scores) / len(cosine_scores)
        reward = (
            args.quality_weight * quality_retention
            + args.storage_weight * structural_reduction
            + args.latency_weight * max(0.0, min(speedup - 1.0, 1.0))
        )
        result = {
            "product": "QLight",
            "experiment": "real_qwen_hadmc_structured_compression",
            "evidence_scope": {
                "checkpoint": "official Qwen checkpoint loaded from Hugging Face",
                "hardware": "measured on one allocated Hygon DCU through the DTK/HYHAL container",
                "compression": "real structured SwiGLU MLP pruning on Qwen Transformer modules",
                "quality": "calibration logit cosine agreement; this is not a benchmark accuracy claim",
                "quantization": "not applied; no INT8 kernel or storage claim is made",
                "fine_tuning": "not applied in this bounded candidate evaluation",
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
                "quality_retention_cosine": round(quality_retention, 6),
                "baseline_latency_ms": round(baseline_latency, 3),
                "candidate_latency_ms": round(candidate_latency, 3),
                "speedup": round(speedup, 4),
                "latency_reduction_pct": round((1.0 - candidate_latency / baseline_latency) * 100.0, 3),
                "reward": round(reward, 7),
                "pruning": pruning_info,
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
