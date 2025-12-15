#!/bin/bash
#SBATCH --job-name=prior_analysis
#SBATCH --output=logs/prior_analysis_%j.out
#SBATCH --error=logs/prior_analysis_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Prior vs Posterior Analysis Script
# This analyzes saved model states and generates diagnostic plots

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR="./testing/output"
OUTPUT_DIR="./testing/output/prior_analysis"
SJ_ID="chr6:34236964:34237203:+"
MODALITY_NAME="splicing_sj"

# If you want to specify model files manually, uncomment and edit these:
# MODEL_FILES=(
#     "./testing/output/model_crispra_only.pkl"
#     "./testing/output/model_both_groups.pkl"
#     "./testing/output/model_uniform_priors.pkl"
# )
# CONDITION_NAMES=(
#     "crispra_only"
#     "both_groups"
#     "uniform_priors"
# )

# =============================================================================
# Setup
# =============================================================================

echo "=========================================="
echo "Prior vs Posterior Analysis"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Create logs directory
mkdir -p logs

# Activate conda environment (adjust path as needed)
# Option 1: If you have a conda environment
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate bayesdream

# Option 2: If using a specific Python installation
# export PATH=/path/to/python/bin:$PATH

# =============================================================================
# Run Analysis
# =============================================================================

echo ""
echo "Running prior analysis..."
echo "Base directory: $BASE_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Splice junction: $SJ_ID"
echo ""

# Auto-detect mode (default)
if [ -z "${MODEL_FILES+x}" ]; then
    echo "Auto-detecting model files in $BASE_DIR"
    python analyze_priors_cluster.py \
        --base_dir "$BASE_DIR" \
        --sj_id "$SJ_ID" \
        --modality_name "$MODALITY_NAME" \
        --output_dir "$OUTPUT_DIR"
else
    # Manual specification mode
    echo "Using manually specified model files:"
    for i in "${!MODEL_FILES[@]}"; do
        echo "  ${CONDITION_NAMES[$i]}: ${MODEL_FILES[$i]}"
    done

    python analyze_priors_cluster.py \
        --sj_id "$SJ_ID" \
        --modality_name "$MODALITY_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --model_files "${MODEL_FILES[@]}" \
        --condition_names "${CONDITION_NAMES[@]}"
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=========================================="
echo "Analysis complete!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"
echo ""
echo "To view results, check:"
echo "  - Individual prior plots: priors_<condition>_<sj_id>.png"
echo "  - Comparison plots: compare_*.png"
echo "  - Summary table: summary_table_<sj_id>.csv"
echo ""
