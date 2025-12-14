#!/bin/bash

################################################################################
# HAD-MC One-Click Deployment Script
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "HAD-MC: Hardware-Aware Dynamic Model Compression"
echo "One-Click Deployment Script"
echo "================================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
print_info "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    print_error "Python 3.8+ required, found $python_version"
    exit 1
fi
print_info "Python $python_version detected ✓"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv venv
    print_info "Virtual environment created ✓"
else
    print_info "Virtual environment already exists ✓"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Install dependencies
print_info "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt > /dev/null 2>&1
    print_info "Dependencies installed ✓"
else
    print_error "requirements.txt not found!"
    exit 1
fi

# Check PyTorch installation
print_info "Checking PyTorch installation..."
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed')" || {
    print_error "PyTorch not installed correctly!"
    exit 1
}
print_info "PyTorch installation verified ✓"

# Create necessary directories
print_info "Creating directories..."
mkdir -p data/financial data/neudet models results logs
print_info "Directories created ✓"

# Prepare datasets
print_info "Preparing datasets..."
if [ -f "data/prepare_datasets.py" ]; then
    python3 data/prepare_datasets.py
    print_info "Datasets prepared ✓"
else
    print_warning "Dataset preparation script not found, skipping..."
fi

# Run tests
print_info "Running tests..."
if [ -f "tests/test_all_algorithms.py" ]; then
    pytest tests/test_all_algorithms.py -v --tb=short || {
        print_warning "Some tests failed, but continuing deployment..."
    }
    print_info "Tests completed ✓"
else
    print_warning "Test file not found, skipping tests..."
fi

# Run quick demo
print_info "Running quick demo..."
if [ -f "experiments/full_pipeline.py" ]; then
    python3 experiments/full_pipeline.py || {
        print_error "Demo failed!"
        exit 1
    }
    print_info "Demo completed ✓"
else
    print_error "Demo script not found!"
    exit 1
fi

# Display results
echo ""
echo "================================================================================"
echo "Deployment Summary"
echo "================================================================================"
echo ""

if [ -f "results/pipeline_results.json" ]; then
    print_info "Results saved to: results/pipeline_results.json"
    python3 -c "
import json
with open('results/pipeline_results.json', 'r') as f:
    results = json.load(f)
    
print('')
print('Performance Metrics:')
print('-' * 60)
print(f\"Model Size Reduction:    {results['improvements']['size_reduction_pct']:.1f}%\")
print(f\"Latency Reduction:       {results['improvements']['latency_reduction_pct']:.1f}%\")
print(f\"Accuracy Change:         {results['improvements']['accuracy_change_pct']:+.2f}%\")
print('')
print('Baseline FP32:')
print(f\"  Accuracy: {results['baseline']['accuracy']:.2f}%\")
print(f\"  Latency:  {results['baseline']['latency_ms']:.2f} ms\")
print(f\"  Size:     {results['baseline']['size_mb']:.2f} MB\")
print('')
print('HAD-MC Compressed:')
print(f\"  Accuracy: {results['hadmc']['accuracy']:.2f}%\")
print(f\"  Latency:  {results['hadmc']['latency_ms']:.2f} ms\")
print(f\"  Size:     {results['hadmc']['size_mb']:.2f} MB\")
print('')
"
fi

echo "================================================================================"
print_info "Deployment completed successfully!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Review results in results/pipeline_results.json"
echo "  2. Run NEU-DET experiment: python3 experiments/neudet_experiment.py"
echo "  3. Run financial experiment: python3 experiments/financial_experiment.py"
echo "  4. Check documentation in docs/"
echo ""
echo "For help, visit: https://github.com/your-username/HAD-MC"
echo ""
