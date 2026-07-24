#!/usr/bin/env python3
"""Run a bounded Qwen-shaped compression search on one DCU.

This is deliberately an architecture experiment, not a claim that an external
Qwen checkpoint was downloaded.  It exercises Qwen-style attention and SwiGLU
blocks with real PyTorch/DCU kernels, searches several structured width
candidates, and records the measured speed/quality proxy trade-off.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        variance = value.pow(2).mean(dim=-1, keepdim=True)
        return value * torch.rsqrt(variance + 1e-6) * self.weight


class QwenShapeBlock(nn.Module):
    """Small Qwen-compatible block shape: RMSNorm, QKV, and SwiGLU MLP."""

    def __init__(self, width: int, feedforward: int) -> None:
        super().__init__()
        self.input_norm = RMSNorm(width)
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)
        self.post_attention_norm = RMSNorm(width)
        self.gate_proj = nn.Linear(width, feedforward, bias=False)
        self.up_proj = nn.Linear(width, feedforward, bias=False)
        self.down_proj = nn.Linear(feedforward, width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        normalized = self.input_norm(value)
        attention = self.o_proj(self.q_proj(normalized) + self.k_proj(normalized) + self.v_proj(normalized))
        value = residual + attention
        residual = value
        normalized = self.post_attention_norm(value)
        mlp = self.down_proj(torch.nn.functional.silu(self.gate_proj(normalized)) * self.up_proj(normalized))
        return residual + mlp


class QwenShapeModel(nn.Module):
    def __init__(self, width: int, feedforward: int, layers: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([QwenShapeBlock(width, feedforward) for _ in range(layers)])
        self.output_norm = RMSNorm(width)
        self.lm_head = nn.Linear(width, width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = block(value)
        return self.lm_head(self.output_norm(value))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def benchmark(model: nn.Module, sample: torch.Tensor, warmup: int = 2, iterations: int = 8) -> tuple[torch.Tensor, float]:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            output = model(sample)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(iterations):
            output = model(sample)
        torch.cuda.synchronize()
    return output, ((time.perf_counter() - started) * 1000.0 / iterations)


def rounded_width(value: float) -> int:
    return max(256, int(math.floor(value / 64.0) * 64))


def inherit_baseline_weights(source: QwenShapeModel, target: QwenShapeModel) -> None:
    """Create a structured FFN candidate from the same baseline parameters."""
    with torch.no_grad():
        for source_block, target_block in zip(source.blocks, target.blocks):
            for name in ("input_norm", "q_proj", "k_proj", "v_proj", "o_proj", "post_attention_norm"):
                getattr(target_block, name).load_state_dict(getattr(source_block, name).state_dict())
            kept = target_block.gate_proj.weight.shape[0]
            target_block.gate_proj.weight.copy_(source_block.gate_proj.weight[:kept, :])
            target_block.up_proj.weight.copy_(source_block.up_proj.weight[:kept, :])
            target_block.down_proj.weight.copy_(source_block.down_proj.weight[:, :kept])
        target.output_norm.load_state_dict(source.output_norm.state_dict())
        target.lm_head.load_state_dict(source.lm_head.state_dict())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--platform-tag", default="dcu")
    parser.add_argument("--candidate-ratios", default="0.15,0.30,0.45,0.60")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("DCU runtime is unavailable to PyTorch")
    device = torch.device("cuda:0")
    torch.manual_seed(20260724)
    torch.cuda.manual_seed_all(20260724)

    width, base_feedforward, layers = 512, 1536, 4
    sample = torch.randn(2, 96, width, device=device)
    baseline = QwenShapeModel(width, base_feedforward, layers).to(device)
    reference, baseline_latency = benchmark(baseline, sample)
    base_params = parameter_count(baseline)

    candidates: list[dict] = []
    for ratio in [float(item) for item in args.candidate_ratios.split(",")]:
        feedforward = rounded_width(base_feedforward * (1.0 - ratio))
        candidate = QwenShapeModel(width, feedforward, layers).to(device)
        inherit_baseline_weights(baseline, candidate)
        output, latency = benchmark(candidate, sample)
        cosine = torch.nn.functional.cosine_similarity(reference.flatten(), output.flatten(), dim=0).item()
        params = parameter_count(candidate)
        structural = 1.0 - params / base_params
        speedup = baseline_latency / latency
        # Score balances a measured runtime advantage with a calibrated output proxy.
        utility = (0.42 * structural) + (0.33 * min(speedup / 1.25, 1.0)) + (0.25 * max(cosine, 0.0))
        candidates.append({
            "candidate_id": f"ffn-prune-{int(ratio * 100):02d}",
            "structured_pruning_ratio": round(ratio, 4),
            "feedforward_width": feedforward,
            "parameters": params,
            "structural_compression_pct": round(structural * 100.0, 2),
            "estimated_int8_storage_compression_pct": round((1.0 - (params / base_params) * 0.25) * 100.0, 2),
            "latency_ms": round(latency, 3),
            "speedup": round(speedup, 3),
            "calibration_cosine": round(cosine, 5),
            "utility_score": round(utility, 5),
        })
        del candidate
        torch.cuda.empty_cache()

    best = max(candidates, key=lambda candidate: candidate["utility_score"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "product": "QLight",
        "experiment": "qwen_architecture_compression_demo",
        "model_scope": "Qwen-compatible architecture proxy; no external Qwen checkpoint was used",
        "platform": args.platform_tag,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "baseline": {"parameters": base_params, "latency_ms": round(baseline_latency, 3), "layers": layers, "hidden_size": width, "feedforward_width": base_feedforward},
        "candidates": candidates,
        "best_candidate": best,
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = output_dir / "QLIGHT_QWEN_DEMO_RESULTS.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"result": str(path), "best_candidate": best}, ensure_ascii=False))


if __name__ == "__main__":
    main()
