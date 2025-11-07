<p align="center">
  <img src="bayesDREAM logo.pdf" alt="bayesDREAM Logo" width="600"/>
</p>

# bayesDREAM: Bayesian Dosage Response Effects Across Modalities

A Bayesian framework for modeling perturbation effects across multiple molecular modalities using PyTorch and Pyro.

## Overview

bayesDREAM models how perturbations propagate through molecular layers using a three-step Bayesian framework:

1. **Technical fit**: Model technical variation in non-targeting controls
2. **Cis fit**: Model direct effects on targeted features (genes, ATAC regions, etc.)
3. **Trans fit**: Model downstream effects as dose-response functions

Supports multiple molecular modalities including genes, transcripts, splicing, ATAC-seq, and custom measurements.

### Key Innovations

**🧬 Flexible Cis Feature Selection**
- Cis feature can be any molecular measurement (gene, ATAC peak, junction, etc.)
- Use `cis_gene='GFI1B'` for genes or `cis_feature='chr9:132283881-132284881'` for ATAC peaks
- Model chromatin accessibility, regulatory elements, or any feature as the direct perturbation target

**🔬 Prior-Informed Fitting** (🚧 In Development - ~30% complete)
- Use prior datasets to inform Bayesian priors on guide effects
- Guide-level priors: Provide expected log2FC for specific guides from prior experiments
- Hyperparameter-level priors: Use prior dataset statistics to inform hierarchical parameters
- Main use case: Prior GEX data → improve ATAC inference (or vice versa)
- See [CODEBASE_EVOLUTION.md](CODEBASE_EVOLUTION.md#2c-prior-informed-cis-fitting--in-development) for implementation status

**🔬 Multi-Modal Integration**
- Add transcripts, splicing, ATAC-seq, and custom modalities to any analysis
- Model cross-layer effects (e.g., chromatin accessibility → gene expression)
- Unified interface across all data types

**📊 Comprehensive Visualization**
- Interactive x-y data plots with k-NN smoothing
- Prior-posterior comparisons with distribution metrics
- Model diagnostics and residual analysis
- Trans function overlays for all function types
- See [docs/PLOTTING_GUIDE.md](docs/PLOTTING_GUIDE.md) for complete guide

## Features

- 🧬 **Gene expression modeling** with negative binomial likelihood
- 🔬 **ATAC-seq integration** for chromatin accessibility and regulatory element analysis
- 📊 **Transcript-level analysis** (isoform usage or independent counts)
- ✂️ **Splicing analysis** (donor usage, acceptor usage, exon skipping, raw junction counts)
- 📈 **Custom modalities** (SpliZ, SpliZVD, user-defined measurements)
- 🎨 **Rich visualization suite** with interactive plots, prior-posterior comparisons, and model diagnostics
- 🔄 **Multiple dose-response functions** (Hill equations, polynomials)
- 🎯 **Guide-level inference** with technical variation modeling
- 🔀 **Permutation testing** for statistical significance
- 💾 **Save/load pipeline stages** with modality-specific control
- 🚀 **GPU support** via PyTorch

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/bayesDREAM.git
cd bayesDREAM

# Install dependencies
pip install torch pyro-ppl pandas numpy scipy scikit-learn matplotlib

# Install in development mode
pip install -e .
```

## Quick Start

### Basic Usage (Gene Expression)

```python
from bayesDREAM import bayesDREAM
import pandas as pd

# Load data
meta = pd.read_csv('meta.csv')
gene_counts = pd.read_csv('gene_counts.csv', index_col=0)

# Create model
model = bayesDREAM(
    meta=meta,
    counts=gene_counts,
    feature_meta=gene_meta,  # Optional: gene annotations (gene, gene_name, gene_id)
    cis_gene='GFI1B',
    guide_covariates=['cell_line'],
    output_dir='./output',
    label='my_run'
)

# Run 3-step pipeline
model.set_technical_groups(['cell_line'])
model.fit_technical(sum_factor_col='sum_factor')
model.fit_cis(sum_factor_col='sum_factor')
model.fit_trans(sum_factor_col='sum_factor_adj', function_type='additive_hill')

# Save results
model.save_technical_fit()
model.save_cis_fit()
model.save_trans_fit()
```

### Multi-Modal Analysis

```python
from bayesDREAM import bayesDREAM

# Initialize with genes (feature_meta optional but recommended)
model = bayesDREAM(
    meta=meta,
    counts=gene_counts,
    feature_meta=gene_meta,  # Optional: gene, gene_name, gene_id
    cis_gene='GFI1B',
    guide_covariates=['cell_line']
)

# Add ATAC-seq data
model.add_atac_modality(
    atac_counts=atac_counts,
    atac_meta=atac_meta,
    genes_to_peaks={'GFI1B': ['chr9:132283881-132284881']}
)

# Add splicing modalities
model.add_splicing_modality(
    sj_counts=sj_counts,
    sj_meta=sj_meta,
    splicing_types=['sj', 'donor', 'acceptor', 'exon_skip']
)

# Add custom modalities
model.add_custom_modality('spliz', spliz_scores, gene_meta, 'normal')

# View all modalities
print(model.list_modalities())
# Output: ['cis', 'gene', 'atac', 'splicing_sj', 'splicing_donor',
#          'splicing_acceptor', 'splicing_exon_skip', 'spliz']

# Run pipeline (operates on primary gene modality)
model.set_technical_groups(['cell_line'])
model.fit_technical(sum_factor_col='sum_factor')
model.fit_cis(sum_factor_col='sum_factor')
model.fit_trans(sum_factor_col='sum_factor_adj', function_type='additive_hill')

# Access other modalities
donor_mod = model.get_modality('splicing_donor')
atac_mod = model.get_modality('atac')
```

### ATAC-Guided Analysis

```python
# Use ATAC peak as cis feature to model regulatory elements
model = bayesDREAM(
    meta=meta,
    counts=atac_counts,
    feature_meta=atac_meta,
    modality_name='atac',
    cis_feature='chr9:132283881-132284881',  # Regulatory region
    guide_covariates=['cell_line'],
    output_dir='./output'
)

# Add genes to model chromatin → transcription effects
model.add_gene_modality(gene_counts=gene_counts, gene_meta=gene_meta)

# Fit: Models how ATAC accessibility drives gene expression
model.fit_cis()  # Cis = ATAC peak accessibility
model.fit_trans(modality_name='gene')  # Trans = Genes regulated by peak
```

### Prior-Informed Fitting (🚧 In Development)

**Use Case:** You have prior information about guide effects (e.g., from a previous GEX experiment) and want to use that to improve inference on a new dataset (e.g., ATAC-seq).

**Current Status:** Infrastructure ~30% complete. Parameters exist but not yet integrated into Pyro model.

```python
# Scenario: Prior GEX experiment → inform current ATAC experiment
# Same guides, same cell line, different modality

# Step 1: Extract guide effects from prior GEX dataset
prior_gex_model = bayesDREAM(meta=prior_meta, counts=prior_gex_counts, cis_gene='GFI1B')
prior_gex_model.fit_cis()

# Get guide-level statistics
prior_guide_effects = pd.DataFrame({
    'guide': guide_names,
    'log2FC': prior_gex_model.posterior_samples_cis['x_eff_g'].mean(dim=0).cpu().numpy()
})

# Step 2: Use as priors for current ATAC dataset
current_atac_model = bayesDREAM(
    meta=current_meta,
    counts=current_atac_counts,
    modality_name='atac',
    cis_feature='chr9:132283881-132284881',
    guide_covariates=['cell_line']
)

# Fit with guide-level priors (🚧 NOT YET FUNCTIONAL)
current_atac_model.fit_cis(
    sum_factor_col='sum_factor',
    manual_guide_effects=prior_guide_effects,  # Parameter exists
    prior_strength=2.0  # Parameter exists
)
# NOTE: Parameters are accepted but priors not yet integrated into Pyro model
```

**Implementation Status:**
- ✅ Parameters added to `fit_cis()`: `manual_guide_effects`, `prior_strength`
- ✅ Tensor preparation and validation
- ❌ Integration into Pyro model (in progress)
- ❌ Hyperparameter-level priors (planned)

See [CODEBASE_EVOLUTION.md](CODEBASE_EVOLUTION.md#2c-prior-informed-cis-fitting--in-development) for detailed roadmap.

### Visualization

```python
# Plot x-y data with trans function overlay
model.plot_xy_data(
    feature='TET2',
    window=100,
    show_correction='both',      # Side-by-side corrected/uncorrected
    show_hill_function=True      # Overlay fitted trans function
)

# Plot ATAC peak
model.plot_xy_data(
    feature='chr9:132283881-132284881',
    modality_name='atac',
    show_ntc_gradient=True       # Color by NTC proportion
)

# Prior-posterior comparison
from bayesDREAM.plotting import plot_prior_posterior
plot_prior_posterior(
    model,
    features=['TET2', 'MYB', 'GAPDH'],
    params=['A', 'alpha', 'beta']
)
```

See **[docs/PLOTTING_GUIDE.md](docs/PLOTTING_GUIDE.md)** for complete visualization documentation.

### Staged Pipeline with Save/Load

```python
# Stage 1: Technical fit
model.set_technical_groups(['cell_line'])
model.fit_technical(sum_factor_col='sum_factor')
model.save_technical_fit()

# Stage 2: Cis fit (in new session)
model2 = bayesDREAM(meta=meta, counts=gene_counts, cis_gene='GFI1B',
                    guide_covariates=['cell_line'])
model2.load_technical_fit()
model2.fit_cis(sum_factor_col='sum_factor')
model2.save_cis_fit()

# Stage 3: Trans fit (in new session)
model3 = bayesDREAM(meta=meta, counts=gene_counts, cis_gene='GFI1B',
                    guide_covariates=['cell_line'])
model3.load_technical_fit()
model3.load_cis_fit()
model3.fit_trans(sum_factor_col='sum_factor_adj')
model3.save_trans_fit()

# Selective modality saving (model-level saved automatically when primary modality included)
model.save_technical_fit(modalities=['gene', 'atac'])
model.load_technical_fit(modalities=['gene'])
```

See **[docs/SAVE_LOAD_GUIDE.md](docs/SAVE_LOAD_GUIDE.md)** for complete save/load documentation.


## Documentation

### User Guides
- **[docs/QUICKSTART_MULTIMODAL.md](docs/QUICKSTART_MULTIMODAL.md)** - Quick reference guide
- **[docs/PLOTTING_GUIDE.md](docs/PLOTTING_GUIDE.md)** - Comprehensive visualization guide
- **[docs/SAVE_LOAD_GUIDE.md](docs/SAVE_LOAD_GUIDE.md)** - Save/load pipeline stages with modality-specific control
- **[docs/API_REFERENCE.md](docs/API_REFERENCE.md)** - Complete API reference with all functions
- **[docs/DATA_ACCESS.md](docs/DATA_ACCESS.md)** - Guide to accessing and working with data
- **[examples/](examples/)** - Example scripts for staged pipeline execution

### Technical Documentation
- **[CLAUDE.md](CLAUDE.md)** - Complete architecture documentation for Claude Code
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture and design patterns
- **[docs/archive/](docs/archive/)** - Historical design documents and refactoring summaries

## Data Requirements

### Cell Metadata
- Required columns: `cell`, `guide`, `target`, `sum_factor`, `cell_line`
- Use `'ntc'` for non-targeting control guides

### Count Matrices
- Genes: rows = genes, columns = cells
- Splice junctions: rows = junctions, columns = cells
- Transcripts: rows = transcripts, columns = cells

### Feature Metadata (Optional but Recommended)
- Rows correspond to features in counts matrix
- For genes, recommended columns: `gene`, `gene_name`, `gene_id`
- For ATAC, include: `chrom`, `start`, `end`, peak annotations
- Can use index as feature identifier if named
- Enables feature-level annotations for downstream analysis
- Passed via `feature_meta` parameter during initialization

See [docs/QUICKSTART_MULTIMODAL.md](docs/QUICKSTART_MULTIMODAL.md) for detailed format specifications.

## Supported Distributions

| Distribution | Use Case | Data Structure |
|--------------|----------|----------------|
| `negbinom` | Gene/transcript counts, ATAC peaks | 2D: (features, cells) |
| `multinomial` | Isoform/donor/acceptor usage | 3D: (features, cells, categories) |
| `binomial` | Exon skipping PSI, raw SJ counts | 2D + denominator |
| `normal` | Continuous scores (SpliZ) | 2D: (features, cells) |
| `mvnormal` | Multivariate scores (SpliZVD) | 3D: (features, cells, dims) |

## Technical Notes

### Cis Modality Design

bayesDREAM uses a dedicated **'cis' modality** internally for modeling direct perturbation effects:

**Initialization Behavior:**
- The 'cis' modality is automatically extracted from your primary modality during initialization
- The primary modality retains only **trans** features (cis feature excluded)
- Additional modalities added later (e.g., via `add_atac_modality()`) do not undergo cis extraction
- All modalities are automatically subset to cells present in the 'cis' modality

**Fitting Behavior:**
- `fit_technical()`: Uses primary modality with BOTH cis and trans features for cell-line effect estimation
- `fit_cis()`: Always uses the 'cis' modality (consistent interface regardless of data type)
- `fit_trans()`: Uses primary modality (trans features only)

**Example**: Specifying `cis_gene='GFI1B'` with 92 genes creates:
- **'cis' modality**: Just GFI1B (for cis modeling)
- **'gene' modality**: Remaining 91 genes (for trans modeling)
- **Technical fit**: Uses all 92 genes, extracts alpha_x for GFI1B, stores alpha_y for other 91

**Parameters:**
- `cis_gene`: For gene modality (e.g., `cis_gene='GFI1B'`)
- `cis_feature`: Generic parameter for any modality (e.g., `cis_feature='chr9:132283881-132284881'` for ATAC)

This design ensures consistent cis/trans separation across all data types while maintaining a unified user interface.

## Citation

If you use bayesDREAM in your research, please cite:

```
[Citation to be added]
```

## Development

### Running Tests

```bash
# Test multi-modal infrastructure
/opt/anaconda3/envs/pyroenv/bin/python tests/test_multimodal_fitting.py

# Test filtering
/opt/anaconda3/envs/pyroenv/bin/python tests/test_filtering_simple.py

# Test per-modality fitting
/opt/anaconda3/envs/pyroenv/bin/python tests/test_per_modality_fitting.py
```

See [tests/README.md](tests/README.md) for complete testing documentation.

### Project Structure

```
bayesDREAM/
├── bayesDREAM/           # Core package
│   ├── __init__.py       # Package exports
│   ├── model.py          # Main bayesDREAM class (~311 lines)
│   ├── core.py           # _BayesDREAMCore base class (~909 lines)
│   ├── modality.py       # Modality data structure for multi-modal support
│   ├── distributions.py  # Distribution-specific observation samplers
│   ├── splicing.py       # Splicing processing (pure Python, no R dependencies)
│   ├── utils.py          # General utility functions
│   ├── fitting/          # Modular fitting methods
│   │   ├── __init__.py
│   │   ├── technical.py  # TechnicalFitter class
│   │   ├── cis.py        # CisFitter class (includes prior-informed fitting infrastructure)
│   │   └── trans.py      # TransFitter class
│   ├── io/               # Save/load functionality
│   │   ├── __init__.py
│   │   ├── save.py       # ModelSaver class
│   │   └── load.py       # ModelLoader class
│   ├── modalities/       # Modality-specific methods
│   │   ├── __init__.py
│   │   ├── transcript.py     # Transcript modality methods
│   │   ├── splicing_modality.py  # Splicing modality methods
│   │   ├── atac.py       # ATAC-seq integration methods
│   │   └── custom.py     # Custom modality methods
│   └── plotting/         # Comprehensive plotting infrastructure
│       ├── __init__.py
│       ├── xy_plots.py       # X-Y data plots (all distributions, k-NN smoothing)
│       ├── prior_posterior.py # Prior-posterior comparison plots
│       ├── prior_sampling.py # Prior sampling utilities
│       ├── model_plots.py    # Model diagnostics and residual plots
│       └── utils.py          # Plotting helper functions
├── examples/             # Usage examples
├── tests/                # Test suite
├── docs/                 # Documentation
└── toydata/              # Test datasets
```

**Note**: The codebase was refactored from a single 4,537-line file into a modular structure (93% reduction in model.py). This improves maintainability while preserving complete backward compatibility. See [docs/archive/](docs/archive/) for refactoring history and [CODEBASE_EVOLUTION.md](CODEBASE_EVOLUTION.md) for a detailed overview of enhancements.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

[Add license information]

## Contact

[Add contact information]

## Acknowledgments

This work uses:
- [PyTorch](https://pytorch.org/) for tensor computations
- [Pyro](https://pyro.ai/) for probabilistic programming
