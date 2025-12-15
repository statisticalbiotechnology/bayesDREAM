# Prior vs Posterior Analysis

This directory contains scripts for analyzing and comparing prior vs posterior distributions across different fitting conditions.

## Files

- `plot_binomial_priors.py`: Core plotting functions
- `analyze_priors_cluster.py`: Analysis script for cluster execution
- `submit_prior_analysis.sh`: SLURM submission script

## Quick Start

### Auto-detection Mode (Recommended)

If your model files follow standard naming patterns:

```bash
# On cluster
sbatch submit_prior_analysis.sh
```

The script will automatically find models matching these patterns:
- `*crispra*model*.pkl` → CRISPRa only
- `*crispri*model*.pkl` → CRISPRi only
- `*both*model*.pkl` → Both groups
- `*uniform*model*.pkl` → Uniform priors
- `*min*denom*3*model*.pkl` → min_denominator=3
- `*min*denom*0*model*.pkl` → min_denominator=0

### Manual Specification Mode

If your files have different names, edit `submit_prior_analysis.sh` and uncomment the `MODEL_FILES` and `CONDITION_NAMES` sections:

```bash
MODEL_FILES=(
    "./testing/output/my_crispra_model.pkl"
    "./testing/output/my_both_groups_model.pkl"
    "./testing/output/my_uniform_model.pkl"
)
CONDITION_NAMES=(
    "crispra_only"
    "both_groups"
    "uniform_priors"
)
```

### Direct Python Execution

You can also run the analysis script directly:

```bash
# Auto-detection
python analyze_priors_cluster.py \
    --base_dir ./testing/output \
    --sj_id chr6:34236964:34237203:+ \
    --output_dir ./testing/output/prior_analysis

# Manual specification
python analyze_priors_cluster.py \
    --sj_id chr6:34236964:34237203:+ \
    --output_dir ./testing/output/prior_analysis \
    --model_files ./testing/output/model1.pkl ./testing/output/model2.pkl \
    --condition_names condition1 condition2
```

## Output Files

The analysis generates three types of outputs:

### 1. Individual Prior Plots

`priors_<condition>_<sj_id>.png` - Shows priors vs posteriors for each condition:
- **A** (baseline): Beta prior vs posterior
- **Vmax_a**: Beta prior vs posterior
- **Vmax_b**: Beta prior vs posterior
- **K_a** (EC50): Gamma prior vs posterior
- **K_b** (EC50): Gamma prior vs posterior
- **Data overview**: Guide-level PSI distribution

### 2. Comparison Plots

`compare_<condition1>_vs_<condition2>_<sj_id>.png` - Direct comparisons:
- CRISPRa only vs Both groups
- Data-driven priors vs Uniform priors
- min_denom=0 vs min_denom=3

Each shows overlaid histograms for all parameters (A, Vmax_a, Vmax_b, K_a, K_b, n_a, n_b).

### 3. Summary Table

`summary_table_<sj_id>.csv` - Quantitative summary with:
- Posterior means
- 95% credible intervals (2.5th - 97.5th percentiles)
- CI widths
- Whether CI includes zero (important for n_a, n_b)

## Interpreting Results

### Diagnosing Loss of Power

Check the comparison plots for:

1. **Wider credible intervals** with 2 groups → Less certainty
2. **n_a and n_b posteriors** shifted to include zero → No detected dose-response
3. **Prior-posterior conflict** → Prior may be too strong or misspecified

### Key Parameters

- **A**: Baseline PSI (should match low guides)
- **Vmax_a, Vmax_b**: Amplitudes of positive/negative Hill components
- **K_a, K_b**: EC50 values (where half-maximal effect occurs)
- **n_a, n_b**: Hill coefficients (steepness of dose-response)
  - If 95% CI includes zero → No detectable effect
  - Magnitude indicates steepness

### Common Issues

**Stuck at floor**: If Vmax_a or Vmax_b mean < 1e-4, the model found no effect
**Wide K posteriors**: Often happens when there's no dose-response signal
**Prior-posterior mismatch**: Check if data-driven priors are appropriate for your data

## Configuration

Edit these variables in `submit_prior_analysis.sh`:

```bash
BASE_DIR="./testing/output"           # Where your model files are
OUTPUT_DIR="./testing/output/prior_analysis"  # Where to save plots
SJ_ID="chr6:34236964:34237203:+"      # Which splice junction to analyze
MODALITY_NAME="splicing_sj"           # Modality name
```

## Requirements

- Saved bayesDREAM model objects (`.pkl` files)
- Models must have `posterior_samples_trans` attribute
- Python packages: torch, numpy, pandas, matplotlib, scipy

## Troubleshooting

**No models found**: Check that files match the naming patterns or use manual specification

**SJ not found**: Verify the `coord.intron` value exists in your modality's feature_meta

**Missing posteriors**: Ensure `fit_trans()` was run before saving the model

**Import errors**: Make sure `plot_binomial_priors.py` is in the same directory

## Interactive Usage

For interactive exploration (not on cluster):

```python
import matplotlib.pyplot as plt
from plot_binomial_priors import plot_additive_hill_priors, compare_one_vs_two_groups

# Single model
fig = plot_additive_hill_priors(model, sj_id='chr6:34236964:34237203:+')
plt.show()

# Compare two models
fig = compare_one_vs_two_groups(
    model_1grp=model_crispra,
    model_2grp=model_both,
    sj_id='chr6:34236964:34237203:+'
)
plt.show()
```
