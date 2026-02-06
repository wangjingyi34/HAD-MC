#!/bin/bash
################################################################################
# HAD-MC 2.0 Third Review - Complete Experiment Runner
#
# This script runs all experiments for the third review:
# 1. SOTA baseline comparisons (AMC, HAQ, DECORE)
# 2. Ablation study
# 3. Cross-dataset experiments
# 4. Cross-platform experiments
# 5. Pareto frontier analysis
# 6. Statistical significance analysis
#
# Usage:
#   bash run_all_experiments.sh [options]
#
# Options:
#   --quick        Quick validation mode (1 epoch, 1 run)
#   --baseline     Only run baseline comparisons
#   --ablation     Only run ablation study
#   --cross-ds     Only run cross-dataset experiments
#   --cross-plat   Only run cross-platform experiments
#   --dry-run      Print commands without executing
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
QUICK=false
RUN_BASELINE=true
RUN_ABLATION=true
RUN_CROSS_DS=true
RUN_CROSS_PLAT=true
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK=true
            shift
            ;;
        --baseline)
            RUN_BASELINE=true
            RUN_ABLATION=false
            RUN_CROSS_DS=false
            RUN_CROSS_PLAT=false
            shift
            ;;
        --ablation)
            RUN_BASELINE=false
            RUN_ABLATION=true
            RUN_CROSS_DS=false
            RUN_CROSS_PLAT=false
            shift
            ;;
        --cross-ds)
            RUN_BASELINE=false
            RUN_ABLATION=false
            RUN_CROSS_DS=true
            RUN_CROSS_PLAT=false
            shift
            ;;
        --cross-plat)
            RUN_BASELINE=false
            RUN_ABLATION=false
            RUN_CROSS_DS=false
            RUN_CROSS_PLAT=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Configuration
PROJECT_ROOT="/home/HAD-MC"
EXPERIMENT_DIR="${PROJECT_ROOT}/experiments_r3"
RESULTS_DIR="${EXPERIMENT_DIR}/results"
LOG_DIR="${EXPERIMENT_DIR}/logs"
PYTHON="${PROJECT_ROOT}/venv/bin/python"  # Adjust if using different python

# Quick mode settings
if [ "$QUICK" = true ]; then
    NUM_EPOCHS=1
    NUM_RUNS=1
    NUM_EPISODES=10
else
    NUM_EPOCHS=100
    NUM_RUNS=5
    NUM_EPISODES=1000
fi

# Helper functions
print_header() {
    echo -e "${BLUE}========================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

run_command() {
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] $@"
    else
        echo "[RUNNING] $@"
        eval "$@"
    fi}

# Main execution
main() {
    print_header "HAD-MC 2.0 Third Review - Complete Experiment Pipeline"

    # Create directories
    mkdir -p "${RESULTS_DIR}"/{baselines,ablation,cross_dataset,cross_platform,pareto,statistical}
    mkdir -p "${LOG_DIR}"
    mkdir -p "${EXPERIMENT_DIR}/checkpoints"

    print_success "Directories created"

    # Activate virtual environment if exists
    if [ -f "${PROJECT_ROOT}/venv/bin/activate" ]; then
        source "${PROJECT_ROOT}/venv/bin/activate"
        print_success "Virtual environment activated"
    fi

    # Record start time
    START_TIME=$(date +%s)

    # ========================================================================
    # 1. SOTA Baseline Comparisons
    # ========================================================================
    if [ "$RUN_BASELINE" = true ]; then
        print_header "1. SOTA Baseline Comparisons"

        print_header "1.1 AMC (AutoML for Model Compression)"
        run_command "${PYTHON} -m experiments_r3.baselines.amc --num-episodes ${NUM_EPISODES} --num-runs ${NUM_RUNS} --output ${RESULTS_DIR}/baselines/amc_results.json"

        print_header "1.2 HAQ (Hardware-Aware Automated Quantization)"
        run_command "${PYTHON} -m experiments_r3.baselines.haq --num-episodes ${NUM_EPISODES} --num-runs ${NUM_RUNS} --output ${RESULTS_DIR}/baselines/haq_results.json"

        print_header "1.3 DECORE (Deep Compression with Reinforcement Learning)"
        run_command "${PYTHON} -m experiments_r3.baselines.decore --num-episodes ${NUM_EPISODES} --num-runs ${NUM_RUNS} --output ${RESULTS_DIR}/baselines/decore_results.json"

        print_success "Baseline comparisons completed"
    fi

    # ========================================================================
    # 2. Ablation Study
    # ========================================================================
    if [ "$RUN_ABLATION" = true ]; then
        print_header "2. Ablation Study"

        print_header "2.1 Running all ablation variants"
        run_command "${PYTHON} -m experiments_r3.ablation.ablation_runner --num-runs ${NUM_RUNS} --output ${RESULTS_DIR}/ablation/"

        print_success "Ablation study completed"
    fi

    # ========================================================================
    # 3. Cross-Dataset Experiments
    # ========================================================================
    if [ "$RUN_CROSS_DS" = true ]; then
        print_header "3. Cross-Dataset Generalization Experiments"

        print_header "3.1 Running cross-dataset experiments"
        run_command "${PYTHON} -m experiments_r3.cross_dataset.cross_dataset_experiment --num-runs ${NUM_RUNS} --output ${RESULTS_DIR}/cross_dataset/"

        print_success "Cross-dataset experiments completed"
    fi

    # ========================================================================
    # 4. Cross-Platform Experiments
    # ========================================================================
    if [ "$RUN_CROSS_PLAT" = true ]; then
        print_header "4. Cross-Platform Validation Experiments"

        print_header "4.1 Running cross-platform experiments"
        run_command "${PYTHON} -m experiments_r3.cross_platform.cross_platform_experiment --num-runs ${NUM_RUNS} --output ${RESULTS_DIR}/cross_platform/"

        print_success "Cross-platform experiments completed"
    fi

    # ========================================================================
    # 5. Pareto Frontier Analysis
    # ========================================================================
    print_header "5. Pareto Frontier Analysis"

    print_header "5.1 Running Pareto analysis"
    run_command "${PYTHON} -m experiments_r3.pareto.pareto_frontier --input ${RESULTS_DIR} --output ${RESULTS_DIR}/pareto/"

    print_success "Pareto frontier analysis completed"

    # ========================================================================
    # 6. Statistical Significance Analysis
    # ========================================================================
    print_header "6. Statistical Significance Analysis"

    print_header "6.1 Running statistical analysis"
    run_command "${PYTHON} -m experiments_r3.utils.statistical --input ${RESULTS_DIR} --output ${RESULTS_DIR}/statistical/"

    print_success "Statistical analysis completed"

    # ========================================================================
    # 7. Generate Final Report
    # ========================================================================
    print_header "7. Generate Final Report"

    print_header "7.1 Compiling all results"
    run_command "${PYTHON} -m experiments_r3.scripts.generate_report --results-dir ${RESULTS_DIR} --output ${RESULTS_DIR}/final_report.json"

    print_success "Final report generated"

    # Calculate total time
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))

    # ========================================================================
    # Summary
    # ========================================================================
    print_header "Experiment Pipeline Completed"

    echo ""
    echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    echo ""
    echo "Results saved to: ${RESULTS_DIR}"
    echo "Logs saved to: ${LOG_DIR}"
    echo ""
    echo "Summary:"
    echo "  - SOTA Baselines:      ${RESULTS_DIR}/baselines/"
    echo "  - Ablation Study:     ${RESULTS_DIR}/ablation/"
    echo "  - Cross-Dataset:      ${RESULTS_DIR}/cross_dataset/"
    echo "  - Cross-Platform:     ${RESULTS_DIR}/cross_platform/"
    echo "  - Pareto Analysis:     ${RESULTS_DIR}/pareto/"
    echo "  - Statistical:        ${RESULTS_DIR}/statistical/"
    echo ""
    print_success "All experiments completed successfully!"
}

# Run main function
main "$@"
