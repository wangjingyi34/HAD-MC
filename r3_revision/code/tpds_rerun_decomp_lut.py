#!/usr/bin/env python3
"""Re-run only the decomposition + LUT validation phases against an existing
TPDS supplementary results directory, preserving the already-trained variance
seeds. Used to apply algorithmic upgrades (e.g. LUT affine calibration)
without re-training the 5-seed variance pool.

Usage::

    python3 tpds_rerun_decomp_lut.py \\
        --results-dir /root/.../tpds_full_dcu_detached2 \\
        --platform-tag dcu \\
        --seeds 11,22,33,44,55 \\
        --skip-public-benchmark
"""

import json
import os
import sys
from datetime import datetime

# Reuse the supplementary module's setup. argparse runs at import time, so we
# pass the same flags via sys.argv before importing.
if __name__ == "__main__":
    # Pre-parse: forward all incoming args to the supplementary module.
    sys.argv = [sys.argv[0]] + sys.argv[1:]

import tpds_supplementary_experiments as sup  # noqa: E402


def main():
    results_dir = sup.RESULTS_DIR
    reference_seed = sup.SEEDS[0]

    variance_results_path = os.path.join(results_dir, "variance", "variance_results.json")
    if not os.path.exists(variance_results_path):
        raise FileNotFoundError(
            f"Cannot re-run decomposition+LUT: missing {variance_results_path}. "
            "Run the full pipeline first to populate the variance phase."
        )
    with open(variance_results_path) as handle:
        variance_results = json.load(handle)

    decomposition_results = sup.run_decomposition_experiment(reference_seed)
    lut_results = sup.run_lut_validation(reference_seed)

    # Preserve existing public_benchmark payload if present; otherwise mark skipped.
    public_path = os.path.join(results_dir, "public_benchmark", "public_benchmark_results.json")
    if os.path.exists(public_path):
        with open(public_path) as handle:
            public_results = json.load(handle)
    else:
        public_results = {"skipped": True, "reason": "Not run in this re-run"}

    metadata_path = os.path.join(results_dir, "RUN_METADATA.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as handle:
            base_metadata = json.load(handle)
    else:
        base_metadata = {}
    base_metadata["rerun_timestamp"] = datetime.now().isoformat()
    base_metadata["rerun_phases"] = ["decomposition", "lut_validation"]

    combined = {
        "metadata": base_metadata,
        "variance": variance_results,
        "decomposition": decomposition_results,
        "lut_validation": lut_results,
        "public_benchmark": public_results,
    }
    out_path = os.path.join(results_dir, "TPDS_SUPPLEMENTARY_RESULTS.json")
    with open(out_path, "w") as handle:
        json.dump(sup.core.clean_for_json(combined), handle, indent=2)
    print(f"\nRe-run complete. Combined results written to: {out_path}")


if __name__ == "__main__":
    main()
