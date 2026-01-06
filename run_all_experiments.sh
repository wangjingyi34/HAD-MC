#!/bin/bash
# HAD-MC Complete Experiment Pipeline
# This script runs all experiments for reproducibility

set -e

echo "=========================================="
echo "HAD-MC Complete Experiment Pipeline"
echo "=========================================="

# Check Python version
python3 --version

# Install dependencies
echo "[1/6] Installing dependencies..."
pip3 install -r requirements.txt

# Clone YOLOv5 if not exists
if [ ! -d "yolov5" ]; then
    echo "[2/6] Cloning YOLOv5..."
    git clone https://github.com/ultralytics/yolov5.git
    cd yolov5
    pip3 install -r requirements.txt
    cd ..
fi

# Download COCO128 dataset
echo "[3/6] Preparing dataset..."
cd yolov5
python3 -c "from utils.dataloaders import create_dataloader; print('Dataset ready')" || \
    python3 -c "import torch; torch.hub.download_url_to_file('https://ultralytics.com/assets/coco128.zip', 'coco128.zip'); import zipfile; zipfile.ZipFile('coco128.zip').extractall('../datasets/')"
cd ..

# Run FP32 Baseline
echo "[4/6] Training FP32 Baseline (100 epochs)..."
cd yolov5
python3 train.py --img 640 --batch 16 --epochs 100 --data coco128.yaml --weights yolov5s.pt --project ../experiments/results --name fp32_baseline
cd ..

# Run HAD-MC Ultra Optimized
echo "[5/6] Running HAD-MC Ultra Optimized..."
python3 experiments/run_hadmc_ultra_optimized.py

# Run Statistical Analysis
echo "[6/6] Running Statistical Analysis..."
python3 experiments/run_statistical_analysis.py

echo "=========================================="
echo "All experiments completed!"
echo "Results saved to experiments/results/"
echo "=========================================="
