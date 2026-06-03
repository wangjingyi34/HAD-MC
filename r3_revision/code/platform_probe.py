#!/usr/bin/env python3

import argparse
import json
import os
import platform
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Probe accelerator/runtime information for HAD-MC dual-platform experiments")
    parser.add_argument("--output-dir", required=True, help="Directory used to store the probe JSON output")
    parser.add_argument("--platform-tag", default=None, help="Logical platform label such as dcu, v100, a100, or gpu")
    parser.add_argument("--matrix-size", type=int, default=2048, help="Square matrix size for the quick GEMM benchmark")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch

    platform_tag = args.platform_tag
    if platform_tag is None:
        if getattr(torch.version, 'hip', None):
            platform_tag = 'dcu'
        elif torch.cuda.is_available():
            platform_tag = 'gpu'
        else:
            platform_tag = 'cpu'

    os.makedirs(args.output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    probe = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'platform_tag': platform_tag,
        'hostname': platform.node(),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'cuda': getattr(torch.version, 'cuda', None),
        'hip': getattr(torch.version, 'hip', None),
        'cuda_available': bool(torch.cuda.is_available()),
        'device': device,
        'device_count': int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        probe['device_name'] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        probe['device_memory_gb'] = round(props.total_memory / 1e9, 2)

        matrix_size = args.matrix_size
        a = torch.randn(matrix_size, matrix_size, device=device)
        b = torch.randn(matrix_size, matrix_size, device=device)

        for _ in range(3):
            _ = a @ b
        torch.cuda.synchronize()

        start = time.perf_counter()
        c = a @ b
        torch.cuda.synchronize()
        end = time.perf_counter()

        probe['matmul_seconds'] = end - start
        probe['matmul_checksum'] = float(c.sum().detach().cpu())

    output_path = os.path.join(args.output_dir, f'platform_probe_{platform_tag}.json')
    with open(output_path, 'w') as f:
        json.dump(probe, f, indent=2)

    print(json.dumps(probe, indent=2))
    print(f'Probe saved to: {output_path}')


if __name__ == '__main__':
    main()