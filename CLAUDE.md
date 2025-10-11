# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

bayesDREAM is a Bayesian framework for modeling CRISPR perturbation effects across multiple molecular modalities. The model consists of three sequential steps:

1. **Technical fit** (`fit_technical`): Models technical variation in non-targeting controls (NTC) to estimate gene-specific overdispersion parameters (`alpha_y`)
2. **Cis fit** (`fit_cis`): Models direct effects on the targeted gene expression (`model_x`)
3. **Trans fit** (`fit_trans`): Models downstream effects on other genes as a function of the cis gene expression (`model_y`)

The codebase uses PyTorch and Pyro for probabilistic programming and variational inference.

## Repository Structure

```
bayesDREAM_forClaude/
├── bayesDREAM/
│   ├── __init__.py          # Package exports
│   ├── model.py             # Core bayesDREAM class (~2250 lines)
│   ├── modality.py          # Modality class for multi-modal data
│   ├── multimodal.py        # MultiModalBayesDREAM wrapper class
│   └── splicing.py          # Splicing data processing (donor/acceptor/exon skip)
├── run_pipeline/
│   ├── prepare_inputs.py    # Data preparation and sum factor calculation
│   ├── run_technical.py     # Step 1: Technical fitting
│   ├── run_cis.py           # Step 2: Cis effect fitting
│   ├── run_trans.py         # Step 3: Trans effect fitting
│   ├── submit_jobs.sh       # Legacy job submission script
│   └── submit_jobs_new.sh   # Current SLURM job submission script
├── splicing code/
│   └── CodeDump.R           # R functions for splicing analysis
├── examples/
│   └── multimodal_example.py # Multi-modal usage examples
└── toydata/                 # Test datasets (genes, splicing, metadata)
```

## Core Architecture

### bayesDREAM Class

The main class in `bayesDREAM/model.py` implements the three-step modeling pipeline:

**Initialization:**
- Takes metadata DataFrame (`meta`) with columns: `L_cell_barcode`, `guide`, `cell_line`, `target`, etc.
- Takes counts DataFrame (`counts`) with genes as rows, cell barcodes as columns
- Creates guide-level metadata by grouping cells by guide and specified covariates
- Supports both CPU and CUDA devices

**Key Methods:**

- `fit_technical(covariates, sum_factor_col, ...)`: Fits NTC-only model to estimate `alpha_y_prefit`
- `set_alpha_x(alpha_x, is_posterior, covariates)`: Sets cis gene overdispersion parameters
- `set_alpha_y(alpha_y, is_posterior, covariates)`: Sets trans gene overdispersion parameters
- `adjust_ntc_sum_factor(covariates, ...)`: Adjusts NTC sum factors for covariates
- `fit_cis(sum_factor_col, ...)`: Fits cis effects using `_model_x`
- `set_x_true(x_true, is_posterior)`: Sets true cis expression for trans modeling
- `permute_genes(genes2permute, ...)`: Permutes guide-gene associations for null testing
- `refit_sumfactor(covariates, ...)`: Re-estimates sum factors based on posterior cis expression
- `fit_trans(sum_factor_col, function_type, ...)`: Fits trans effects using `_model_y`

**Function Types for Trans Modeling:**

The `fit_trans` method supports multiple functional forms for modeling how trans gene expression depends on cis gene expression:

- `single_hill`: Single Hill equation (positive or negative)
- `additive_hill`: Additive combination of positive and negative Hill functions
- `polynomial`: Polynomial function with configurable degree (default: 6)

### Probabilistic Models

Three Pyro models implement the statistical framework:

1. **`_model_technical`**: Models NTC cells to estimate baseline overdispersion
   - Negative binomial likelihood with log-normal priors
   - Estimates per-gene `alpha_y` parameters

2. **`_model_x`**: Models cis effects on the targeted gene
   - Accounts for guide-level and cell-line-level variation
   - Estimates true gene expression `x_true` for each guide
   - Uses sum factors for normalization

3. **`_model_y`**: Models trans effects as functions of cis expression
   - Supports Hill-based functions or polynomials
   - Includes sparsity priors (gamma distribution on effect sizes)
   - Models gene-specific dose-response curves

### Pipeline Scripts

**`prepare_inputs.py`**: Data preprocessing
- Takes metadata and count matrices
- Calls R script to calculate sum factors using `scran::calculateSumFactors`
- Splits data into technical (NTC only) and cis-specific subsets
- Usage: `python prepare_inputs.py <label> <outdir>`

**`run_technical.py`**: Technical fitting
- Loads NTC-only data
- Fits `bayesDREAM.fit_technical()` with cell_line covariates
- Saves `alpha_y_prefit.pt` for use in subsequent steps
- Usage: `python run_technical.py --inlabel <label> --label <label_i> --outdir <outdir> --cores <cores>`

**`run_cis.py`**: Cis effect fitting
- Loads gene-specific data (NTC + one cis gene)
- Sets alpha parameters from technical fit
- Adjusts NTC sum factors for covariates
- Fits cis model and saves posterior samples and `x_true`
- Supports subsetting by CRISPRa/CRISPRi or NTC-only
- Usage: `python run_cis.py --inlabel <label> --label <label_i> --outdir <outdir> --cis_gene <gene> --cores <cores> [--subset <NTC|CRISPRa|CRISPRi>]`

**`run_trans.py`**: Trans effect fitting
- Loads cis gene posteriors and sets `x_true`
- Optionally permutes guide-gene associations for null distribution
- Refits sum factors based on posterior cis expression (when not subsetting)
- Fits trans model with specified function type
- Usage: `python run_trans.py --inlabel <label> --label <label_i> --outdir <outdir> --cis_gene <gene> --permtype <none|All|gene_name> --function_type <additive_hill|polynomial> --cores <cores> [--subset <NTC|CRISPRa|CRISPRi>]`

### Job Submission

`submit_jobs_new.sh` provides a flexible SLURM workflow:

**Flags:**
- `--prepare`: Run data preparation
- `--tech`: Run technical fitting
- `--cis`: Run cis fitting for each gene
- `--trans`: Enable trans fitting
- `--full`: Run trans on full dataset (combine with `--full-none` and/or `--full-all`)
- `--full-none`: Run trans with no permutation (real data)
- `--full-all`: Run trans with all genes permuted (global null)
- `--each-permutation`: Run array job permuting each trans gene individually
- `--subsets`: Run separate analyses for CRISPRa and CRISPRi cell lines
- `--function-types <type1,type2>`: Specify function types (default: additive_hill,polynomial)
- `--array-max <N>`: Maximum concurrent array tasks (default: 50)
- `--start-reps <N>`, `--end-reps <N>`: Run multiple replicates with different random seeds
- `--label <name>`: Base output label
- `--outdir <path>`: Output directory
- `--cis-genes <gene1,gene2>`: Comma-separated list of cis genes
- `--cores <N>`: Number of CPU cores per job

**Typical Workflow:**

1. Prepare inputs (once per dataset):
   ```bash
   bash submit_jobs_new.sh --prepare --label run_20250613 --outdir /path/to/output
   ```

2. Run full pipeline with multiple replicates:
   ```bash
   bash submit_jobs_new.sh --tech --cis --trans --full --full-none --full-all \
     --function-types additive_hill,polynomial \
     --start-reps 1 --end-reps 10 \
     --label run_20250613 --outdir /path/to/output \
     --cis-genes GFI1B,TET2,MYB,NFE2
   ```

3. Run permutation testing (one job per trans gene):
   ```bash
   bash submit_jobs_new.sh --trans --full --each-permutation \
     --function-types additive_hill \
     --label run_20250613 --outdir /path/to/output
   ```

### Threading and Performance

All pipeline scripts use `set_max_threads(cores)` to control CPU usage:
- Sets environment variables for OpenBLAS, MKL, NumExpr
- Calls `torch.set_num_threads(cores)`
- Each SLURM job should specify `--cpus-per-task` matching the `--cores` argument

### Random Seeds

Each script derives a deterministic random seed from input parameters (label, gene, permutation type) using hashing:
```python
seed = abs(hash(args.label + args.cis_gene)) % (2**32)
pyro.set_rng_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
```

This ensures reproducibility while allowing multiple replicates with `--start-reps`/`--end-reps`.

### Output Structure

```
outdir/
├── <label>/
│   ├── meta_technical.csv
│   ├── counts_technical.csv
│   ├── meta_cis_<gene>.csv
│   ├── counts_cis_<gene>.csv
│   └── logs/
└── <label>_<rep>/
    ├── alpha_y_prefit.pt
    ├── posterior_samples_technical.pt
    ├── <gene>_run/
    │   ├── x_true.pt
    │   ├── alpha_x_prefit.pt
    │   ├── alpha_y_prefit.pt
    │   ├── posterior_samples_cis.pt
    │   └── posterior_samples_trans_<function_type>_<permtype>.pt
    └── logs/
```

## Common Development Tasks

### Running the Pipeline Locally

For testing with toy data:

```bash
# Prepare inputs
python bayesDREAM_forClaude/run_pipeline/prepare_inputs.py test_run ./output

# Run technical fit
python bayesDREAM_forClaude/run_pipeline/run_technical.py \
  --inlabel test_run --label test_run_1 --outdir ./output --cores 4

# Run cis fit
python bayesDREAM_forClaude/run_pipeline/run_cis.py \
  --inlabel test_run --label test_run_1 --outdir ./output \
  --cis_gene GFI1B --cores 4

# Run trans fit
python bayesDREAM_forClaude/run_pipeline/run_trans.py \
  --inlabel test_run --label test_run_1 --outdir ./output \
  --cis_gene GFI1B --permtype none --function_type additive_hill --cores 4
```

### Modifying the Model

When adding new functionality to `bayesDREAM/model.py`:

1. Helper functions go at the top of the file (before the class definition)
2. New Pyro models should follow the naming convention `_model_<name>`
3. Public methods should include docstrings explaining parameters
4. Use `self.device` for tensor placement
5. Save important intermediate results with `torch.save()`

### Adding New Function Types

To add a new dose-response function:

1. Define the function in the helper section (e.g., `def my_function(x, params)`)
2. Add a conditional branch in `_model_y` to handle the new function type
3. Update `fit_trans` to set appropriate priors and optimization settings
4. Add the new type to `FUNCTION_TYPES` in `submit_jobs_new.sh`

### Testing Changes

The `toydata/` directory contains small test datasets. Use these for quick validation before running on full data:

- `gene_counts.csv`, `gene_meta.csv`: Gene expression data
- `SJ_counts.csv`, `SJ_meta.csv`: Splice junction data
- `SpliZ_counts.csv`, `SpliZ_meta.csv`: Splicing quantification

## Multi-Modal Architecture

bayesDREAM now supports multiple molecular modalities beyond gene expression. This allows modeling of transcripts, splicing, and custom measurements within a unified framework.

### Modality Class

The `Modality` class (`bayesDREAM/modality.py`) provides a standardized container for different data types:

**Supported Distributions:**
- `negbinom`: Negative binomial (gene counts, transcript counts)
- `multinomial`: Categorical/proportional data (isoform usage, donor/acceptor usage)
- `binomial`: Binary outcomes with denominators (exon skipping PSI)
- `normal`: Continuous measurements (SpliZ scores)
- `mvnormal`: Multivariate normal (SpliZVD with 3D vectors)

**Data Structures:**
- **2D data** (negbinom, normal, binomial): `(features, cells)` or `(cells, features)`
- **3D multinomial**: `(features, cells, categories)` - e.g., donor sites × cells × acceptor options
- **3D mvnormal**: `(features, cells, dimensions)` - e.g., genes × cells × 3 (z0, z1, z2)
- **Binomial**: 2D counts + 2D denominator array

**Key Features:**
- Feature-level metadata (gene names, junction coordinates, etc.)
- Cell subsetting and feature subsetting
- Automatic validation of shapes and distribution requirements
- Conversion to PyTorch tensors

### MultiModalBayesDREAM Class

The `MultiModalBayesDREAM` class (`bayesDREAM/multimodal.py`) extends the base `bayesDREAM` class:

**Initialization:**
```python
from bayesDREAM import MultiModalBayesDREAM

model = MultiModalBayesDREAM(
    meta=cell_metadata,
    counts=gene_counts,              # Primary modality (genes)
    cis_gene='GFI1B',
    primary_modality='gene',         # Which modality drives cis/trans effects
    output_dir='./output',
    label='multimodal_run'
)
```

**Adding Modalities:**

1. **Transcript counts** (as isoform usage or independent counts):
   ```python
   model.add_transcript_modality(
       transcript_counts=tx_counts,
       transcript_meta=tx_meta,      # Must have: transcript_id, gene
       use_isoform_usage=True        # True: multinomial, False: negbinom
   )
   ```

2. **Splicing data** (donor/acceptor/exon skipping):
   ```python
   model.add_splicing_modality(
       sj_counts=sj_counts,
       sj_meta=sj_meta,              # Must have: coord.intron, chrom, intron_start, intron_end, strand
       splicing_types=['donor', 'acceptor', 'exon_skip'],
       gene_of_interest='GFI1B',
       min_cell_total=1,             # Min reads for donor/acceptor
       min_total_exon=2              # Min reads for exon skipping
   )
   ```

3. **Custom modalities** (SpliZ, SpliZVD, etc.):
   ```python
   # SpliZ scores (normal distribution)
   model.add_custom_modality(
       name='spliz',
       counts=spliz_scores,          # 2D: genes × cells
       feature_meta=gene_meta,
       distribution='normal'
   )

   # SpliZVD (multivariate normal, 3D)
   model.add_custom_modality(
       name='splizvd',
       counts=splizvd_array,         # 3D: genes × cells × 3
       feature_meta=gene_meta,
       distribution='mvnormal'
   )
   ```

**Working with Modalities:**
```python
# List all modalities
print(model.list_modalities())

# Access specific modality
donor_mod = model.get_modality('splicing_donor')
print(donor_mod.dims)                    # {'n_features': 100, 'n_cells': 500, 'n_categories': 10}
print(donor_mod.feature_meta.head())     # Metadata: chrom, strand, donor, acceptors, etc.

# Subset modality
subset = donor_mod.get_feature_subset(['feature1', 'feature2'])
```

### Splicing Processing

The `splicing.py` module wraps R functions from `CodeDump.R` to compute splicing metrics:

**Donor Usage**: Groups splice junctions by donor site (5' splice site). Returns multinomial counts showing which acceptor is used for each donor.
- Distribution: `multinomial`
- Dimensions: `(n_donors, n_cells, max_acceptors_per_donor)`
- Metadata: `chrom`, `strand`, `donor`, `acceptors` (list), `n_acceptors`

**Acceptor Usage**: Groups junctions by acceptor site (3' splice site). Returns multinomial counts showing which donor is used for each acceptor.
- Distribution: `multinomial`
- Dimensions: `(n_acceptors, n_cells, max_donors_per_acceptor)`
- Metadata: `chrom`, `strand`, `acceptor`, `donors` (list), `n_donors`

**Exon Skipping**: Detects cassette exon triplets (inc1, inc2, skip) and computes inclusion/total counts.
- Distribution: `binomial`
- Dimensions: `(n_events, n_cells)` for both inclusion and total
- Metadata: `trip_id`, `chrom`, `strand`, `d1`, `a2`, `d2`, `a3`, `sj_inc1`, `sj_inc2`, `sj_skip`

**R Function Requirements:**
- `psi_donor_usage_strand()`: Computes donor PSI
- `psi_acceptor_usage_strand()`: Computes acceptor PSI
- `psi_exon_skipping_strand()`: Finds cassette exons and computes PSI

**SJ Metadata Requirements:**
- `coord.intron`: Junction ID (e.g., "chr1:12345:67890:+")
- `chrom`: Chromosome
- `intron_start`, `intron_end`: Junction coordinates
- `strand`: Strand ('+', '-', 1, or 2)
- Optional: `gene_short_name.start`, `gene_short_name.end` for gene filtering

### Current Limitations

1. **Cis/Trans fitting**: Currently only the primary modality (usually genes) is used for cis and trans modeling. Future versions will support modality-specific fits.

2. **Technical fitting**: Only the primary modality supports `fit_technical()`. Other modalities require manual specification of overdispersion parameters.

3. **Permutation testing**: `permute_genes()` operates on the primary modality.

4. **Sum factors**: Calculated only for gene-level data. Other modalities may need alternative normalization strategies.

### Example Workflows

**Comprehensive example** (`examples/multimodal_example.py`):
```python
from bayesDREAM import MultiModalBayesDREAM

# Load data
meta = pd.read_csv('meta.csv')
gene_counts = pd.read_csv('gene_counts.csv', index_col=0)
sj_counts = pd.read_csv('SJ_counts.csv', index_col=0)
sj_meta = pd.read_csv('SJ_meta.csv')

# Create multi-modal model
model = MultiModalBayesDREAM(
    meta=meta,
    counts=gene_counts,
    cis_gene='GFI1B',
    output_dir='./output',
    label='multimodal_run'
)

# Add splicing modalities
model.add_splicing_modality(
    sj_counts=sj_counts,
    sj_meta=sj_meta,
    splicing_types=['donor', 'acceptor', 'exon_skip'],
    gene_of_interest='GFI1B'
)

# Inspect modalities
print(model.list_modalities())

# Run standard pipeline (operates on primary 'gene' modality)
model.fit_technical(covariates=['cell_line'])
model.fit_cis(sum_factor_col='sum_factor')
model.fit_trans(sum_factor_col='sum_factor_adj', function_type='additive_hill')

# Access splicing data for downstream analysis
donor_modality = model.get_modality('splicing_donor')
donor_counts = donor_modality.counts        # 3D array
donor_meta = donor_modality.feature_meta    # Donor site annotations
```

See `examples/multimodal_example.py` for complete examples including transcripts, custom modalities, and advanced usage.
