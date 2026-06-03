#!/usr/bin/env python3
"""Re-run only the *extra* supplementary phases (public benchmark, reward
weight sensitivity, baseline fairness) against an existing TPDS supplementary
results directory. Preserves the variance / decomposition / LUT phases that
were produced earlier and that take much longer to retrain.

Usage::

    python3 tpds_rerun_extras.py \\
        --results-dir /root/HAD-MC-worktree/r3_revision/results/tpds_full_dcu_detached2 \\
        --platform-tag dcu \\
        --seeds 11,22,33,44,55 \\
        --lut-batch-size 32
"""

import json
import os
import sys
from datetime import datetime

if __name__ == "__main__":
    sys.argv = [sys.argv[0]] + sys.argv[1:]

import tpds_supplementary_experiments as sup  # noqa: E402


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def main():
    results_dir = sup.RESULTS_DIR
    reference_seed = sup.SEEDS[0]

    variance_results = _load_json(os.path.join(results_dir, "variance", "variance_results.json"))
    decomposition_results = _load_json(os.path.join(results_dir, "decomposition", "decomposition_results.json"))
    lut_results = _load_json(os.path.join(results_dir, "lut_validation", "lut_validation_results.json"))

    if variance_results is None:
        raise FileNotFoundError(
            "Missing variance/variance_results.json. Run the full pipeline first."
        )
    if decomposition_results is None:
        raise FileNotFoundError(
            "Missing decomposition/decomposition_results.json. Run the decomposition phase first."
        )
    if lut_results is None:
        raise FileNotFoundError(
            "Missing lut_validation/lut_validation_results.json. Run the LUT validation phase first."
        )

    # Re-run only the new / previously-skipped phases.
    public_benchmark_results = sup.run_public_benchmark()

    if sup.ARGS.skip_reward_sensitivity:
        reward_sensitivity_results = {
            "skipped": True,
            "reason": "Skipped by --skip-reward-sensitivity",
        }
        sup.save_json(
            "reward_sensitivity/reward_sensitivity_results.json",
            reward_sensitivity_results,
        )
    else:
        reward_sensitivity_results = sup.run_reward_sensitivity_experiment(lut_results)

    baseline_fairness_results = sup.run_baseline_fairness_experiment(reference_seed)

    metadata_path = os.path.join(results_dir, "RUN_METADATA.json")
    base_metadata = _load_json(metadata_path) or {}
    base_metadata["rerun_timestamp"] = datetime.now().isoformat()
    base_metadata["rerun_phases"] = [
        "public_benchmark",
        "reward_sensitivity",
        "baseline_fairness",
    ]

    combined = {
        "metadata": base_metadata,
        "variance": variance_results,
        "decomposition": decomposition_results,
        "lut_validation": lut_results,
        "public_benchmark": public_benchmark_results,
        "reward_sensitivity": reward_sensitivity_results,
        "baseline_fairness": baseline_fairness_results,
    }
    out_path = os.path.join(results_dir, "TPDS_SUPPLEMENTARY_RESULTS.json")
    with open(out_path, "w") as handle:
        json.dump(sup.core.clean_for_json(combined), handle, indent=2)
    print(f"\nExtras re-run complete. Combined results written to: {out_path}")


if __name__ == "__main__":
    main()
