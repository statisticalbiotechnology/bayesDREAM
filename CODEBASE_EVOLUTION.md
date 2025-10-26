# bayesDREAM Codebase Evolution: From Uni-Modal to Multi-Modal Framework

**Prepared for:** PI Review
**Date:** 2025-01-26
**Summary:** Evolution from gene-specific implementation to a flexible, multi-modal Bayesian framework

---

## 🎯 Executive Summary

bayesDREAM has evolved from a **single-modality, gene-specific implementation** into a **mature, extensible multi-modal framework** that supports diverse molecular measurements and flexible cis-feature specification. The codebase has been completely refactored for maintainability while preserving full backward compatibility.

**Key Achievements:**
- ✅ Multi-modal support (genes, transcripts, splicing, ATAC-seq, custom)
- ✅ Flexible cis-feature specification (genes, ATAC regions, any feature)
- ✅ Prior-guided cis selection (e.g., using ATAC to identify regulatory regions)
- ✅ Comprehensive plotting infrastructure with interactive visualizations
- ✅ Modular architecture (93% reduction in main file size)
- ✅ Full backward compatibility with original API

---

## 📊 What You Saw Before: Uni-Modal Gene-Specific

**Original Implementation (What you've seen):**

```python
# Limited to gene expression only
model = bayesDREAM(
    meta=meta,
    counts=gene_counts,  # ONLY genes
    cis_gene='GFI1B'     # MUST be a gene
)

model.fit_technical()
model.fit_cis()
model.fit_trans()
```

**Limitations of Original Version:**
- ❌ Only supported gene expression data
- ❌ Cis feature **must** be a gene
- ❌ No integration with other molecular modalities
- ❌ Limited visualization capabilities
- ❌ Monolithic 4,537-line file (difficult to maintain)
- ❌ Could not leverage prior knowledge (e.g., ATAC) for cis selection

---

## 🚀 What's New: Multi-Modal Framework

### 1. **Multi-Modal Data Support**

The framework now supports **5 distribution types** covering diverse molecular measurements:

| Modality | Distribution | Use Case | Example |
|----------|-------------|----------|---------|
| **Genes** | Negative binomial | Gene expression counts | RNA-seq, 10x |
| **Transcripts** | Negative binomial / Multinomial | Isoform-level analysis | Long-read sequencing |
| **Splicing** | Multinomial / Binomial | Alternative splicing events | Donor/acceptor usage, exon skipping |
| **ATAC-seq** | Negative binomial | Chromatin accessibility | Peak counts, regulatory regions |
| **Custom** | Normal / Multivariate normal | Continuous scores | SpliZ, SpliZVD, custom metrics |

**Example: Integrating Multiple Modalities**

```python
from bayesDREAM import bayesDREAM

# Initialize with gene expression
model = bayesDREAM(
    meta=meta,
    counts=gene_counts,
    gene_meta=gene_meta,  # NEW: Optional gene annotations
    cis_gene='GFI1B'
)

# Add ATAC-seq data
model.add_atac_modality(
    atac_counts=atac_counts,
    atac_meta=atac_meta,
    genes_to_peaks=genes_to_peaks  # Link peaks to genes
)

# Add splicing analysis
model.add_splicing_modality(
    sj_counts=sj_counts,
    sj_meta=sj_meta,
    splicing_types=['donor', 'acceptor', 'exon_skip', 'sj']
)

# Add custom continuous measurements
model.add_custom_modality(
    name='spliz',
    counts=spliz_scores,
    feature_meta=gene_meta,
    distribution='normal'
)

# View all modalities
print(model.list_modalities())
# Output: ['cis', 'gene', 'atac', 'splicing_donor', 'splicing_acceptor',
#          'splicing_exon_skip', 'splicing_sj', 'spliz']
```

---

### 2. **Flexible Cis-Feature Specification**

**Major Innovation:** Cis feature is no longer limited to genes!

#### 2a. Basic Usage: Gene as Cis Feature

```python
# Standard: Gene as cis feature
model = bayesDREAM(
    meta=meta,
    counts=gene_counts,
    cis_gene='GFI1B'  # Traditional gene-based approach
)
```

#### 2b. Advanced: ATAC Peak as Cis Feature

```python
# NEW: ATAC peak as cis feature
model = bayesDREAM(
    meta=meta,
    counts=atac_counts,
    atac_meta=atac_meta,
    primary_modality='atac',  # ATAC is primary
    cis_feature='chr9:132283881-132284881',  # Regulatory region
    output_dir='./output'
)

# Add gene expression as secondary modality
model.add_gene_modality(
    gene_counts=gene_counts,
    gene_meta=gene_meta
)

# Fit: Models how ATAC accessibility drives gene expression
model.fit_cis()  # Cis = ATAC peak accessibility
model.fit_trans(modality_name='gene')  # Trans = Gene expression changes
```

**Why This Matters:**
- Model regulatory elements directly (enhancers, promoters)
- Capture chromatin-mediated effects
- Test hypotheses about regulatory architecture

#### 2c. Prior-Guided Cis Selection: ATAC + GEX Integration

**Use Case:** You have both ATAC and gene expression, and you want to use **ATAC data to select the cis feature** (regulatory region driving effects).

```python
# Step 1: Identify cis peak from ATAC data
# (e.g., differential accessibility analysis, prior knowledge, etc.)
cis_peak = 'chr9:132283881-132284881'  # GFI1B promoter region

# Step 2: Initialize with ATAC as primary, peak as cis
model = bayesDREAM(
    meta=meta,
    counts=atac_counts,
    atac_meta=atac_meta,
    primary_modality='atac',
    cis_feature=cis_peak,  # Selected based on ATAC prior
    output_dir='./output'
)

# Step 3: Add gene expression for trans effects
model.add_gene_modality(
    gene_counts=gene_counts,
    gene_meta=gene_meta
)

# Step 4: Model chromatin → transcription cascade
model.set_technical_groups(['cell_line'])
model.fit_technical()  # Technical variation in ATAC
model.fit_cis()        # Model cis ATAC accessibility
model.fit_trans(modality_name='gene')  # How ATAC drives gene expression

# Interpretation:
# - x_true = accessibility of chr9:132283881-132284881
# - Trans effects = how this accessibility regulates genome-wide gene expression
```

**Alternative: Gene as Cis, ATAC as Prior**

```python
# Use ATAC to inform which gene to model
# Example: Select gene with most variable promoter accessibility
most_variable_promoter = identify_from_atac(atac_counts, atac_meta)
target_gene = 'GFI1B'  # Mapped from ATAC peak

model = bayesDREAM(
    meta=meta,
    counts=gene_counts,
    gene_meta=gene_meta,
    cis_gene=target_gene,  # Selected using ATAC prior
    output_dir='./output'
)

# Add ATAC as secondary modality
model.add_atac_modality(
    atac_counts=atac_counts,
    atac_meta=atac_meta,
    genes_to_peaks=genes_to_peaks
)

# Fit: Model gene expression → ATAC changes
model.fit_cis()  # Cis = GFI1B expression
model.fit_trans(modality_name='atac')  # Trans = Chromatin changes
```

---

### 3. **Comprehensive Plotting Infrastructure**

**NEW:** Interactive, publication-ready visualizations for all modalities.

#### 3a. X-Y Data Plots (Raw Data Visualization)

```python
# Plot gene expression vs cis gene
model.plot_xy_data(
    feature='TET2',
    window=100,
    show_correction='both',  # Corrected + uncorrected
    show_hill_function=True  # Overlay fitted trans function
)

# Plot ATAC peak vs cis gene
model.plot_xy_data(
    feature='chr1:12345:67890',
    modality_name='atac',
    min_counts=5
)

# Plot splice junction with NTC gradient coloring
model.plot_xy_data(
    feature='chr1:999788:999865',
    modality_name='splicing_sj',
    show_ntc_gradient=True  # Color by NTC proportion
)
```

#### 3b. Prior-Posterior Comparison

```python
from bayesDREAM.plotting import plot_prior_posterior

# Compare prior and posterior distributions
plot_prior_posterior(
    model,
    features=['TET2', 'MYB', 'GAPDH'],
    params=['A', 'alpha', 'beta'],
    ci_level=0.95
)

# Summary metrics
from bayesDREAM.plotting.utils import compute_all_metrics
metrics = compute_all_metrics(prior_samples, posterior_samples)
# Returns: overlap, KL divergence, posterior coverage
```

#### 3c. Model Diagnostic Plots

```python
from bayesDREAM.plotting import plot_posterior_predictive, plot_residuals

# Check model fit
plot_posterior_predictive(model, features=['TET2', 'MYB'])
plot_residuals(model, features=['TET2'], plot_type='qq')
```

**Plotting Features:**
- ✅ Automatic technical group coloring (CRISPRa/CRISPRi)
- ✅ k-NN smoothing with configurable window
- ✅ NTC gradient visualization
- ✅ Trans function overlay (all function types)
- ✅ Side-by-side corrected/uncorrected views
- ✅ Works with all modalities (genes, ATAC, splicing, custom)

**See:** [docs/PLOTTING_GUIDE.md](docs/PLOTTING_GUIDE.md) for complete documentation

---

### 4. **Modular, Maintainable Architecture**

**Before:** Single monolithic file (4,537 lines)
**After:** Clean modular structure (93% reduction in main file)

```
bayesDREAM/
├── model.py              # 311 lines (was 4,537!)
├── core.py               # Base class
├── fitting/              # Fitting methods
│   ├── technical.py      # Technical fitting
│   ├── cis.py            # Cis fitting
│   └── trans.py          # Trans fitting
├── io/                   # Save/load
│   ├── save.py
│   └── load.py
├── modalities/           # Modality-specific methods
│   ├── transcript.py
│   ├── splicing_modality.py
│   ├── atac.py           # NEW: ATAC support
│   └── custom.py
├── plotting/             # NEW: Complete plotting infrastructure
│   ├── xy_plots.py       # X-Y data plots
│   ├── prior_posterior.py # Prior-posterior comparison
│   ├── model_plots.py    # Model diagnostics
│   └── utils.py          # Plotting utilities
└── distributions.py      # Distribution samplers
```

**Benefits:**
- ✅ Easier to test individual components
- ✅ Clearer separation of concerns
- ✅ Faster development of new features
- ✅ Better code documentation
- ✅ Full backward compatibility maintained

---

### 5. **Enhanced Data Handling**

#### Gene Metadata Support

```python
# NEW: Optional gene metadata for better annotations
gene_meta = pd.DataFrame({
    'gene': ['ENSG00000...', ...],
    'gene_name': ['GFI1B', 'TET2', ...],
    'gene_id': ['ENSG00000...', ...]
})

model = bayesDREAM(
    meta=meta,
    counts=gene_counts,
    gene_meta=gene_meta,  # Optional but recommended
    cis_gene='GFI1B'
)

# Supports flexible identifier matching:
# - By gene name: 'GFI1B'
# - By Ensembl ID: 'ENSG00000165702'
# - By index position
```

#### ATAC Peak-Gene Linkage

```python
# Link ATAC peaks to genes for integrated analysis
genes_to_peaks = {
    'GFI1B': ['chr9:132283881-132284881', 'chr9:132290000-132291000'],
    'TET2': ['chr4:106196000-106197000']
}

model.add_atac_modality(
    atac_counts=atac_counts,
    atac_meta=atac_meta,
    genes_to_peaks=genes_to_peaks
)

# Access peaks for specific gene
gfi1b_peaks = model.get_modality('atac').get_peaks_for_gene('GFI1B')
```

#### Splicing Analysis (Pure Python)

```python
# No R dependencies! Pure Python splicing analysis
model.add_splicing_modality(
    sj_counts=sj_counts,
    sj_meta=sj_meta,
    splicing_types=['sj', 'donor', 'acceptor', 'exon_skip'],
    gene_counts=gene_counts  # Optional: for SJ normalization
)

# Automatic computation of:
# - Raw SJ counts (binomial with gene expression denominator)
# - Donor usage (multinomial)
# - Acceptor usage (multinomial)
# - Exon skipping PSI (binomial)
```

---

## 📈 Use Cases Enabled by New Framework

### Use Case 1: Multi-Modal Perturbation Profiling

**Goal:** Understand how GFI1B perturbation affects multiple molecular layers

```python
model = bayesDREAM(meta=meta, counts=gene_counts, cis_gene='GFI1B')

# Add all modalities
model.add_atac_modality(atac_counts, atac_meta, genes_to_peaks)
model.add_splicing_modality(sj_counts, sj_meta, ['donor', 'acceptor', 'exon_skip'])
model.add_custom_modality('spliz', spliz_scores, gene_meta, 'normal')

# Fit once, analyze everywhere
model.fit_technical()
model.fit_cis()
model.fit_trans(modality_name='gene')
model.fit_trans(modality_name='atac')
model.fit_trans(modality_name='splicing_donor')
model.fit_trans(modality_name='spliz')

# Cross-modality analysis
# - Which genes respond to GFI1B perturbation?
# - Which chromatin regions open/close?
# - Which splicing events change?
# - Which splicing factors are affected?
```

### Use Case 2: Regulatory Element Mapping

**Goal:** Test if specific ATAC peak drives gene expression

```python
# Hypothesis: chr9:132283881-132284881 is GFI1B's key regulatory element

model = bayesDREAM(
    meta=meta,
    counts=atac_counts,
    atac_meta=atac_meta,
    primary_modality='atac',
    cis_feature='chr9:132283881-132284881'  # Candidate enhancer
)

model.add_gene_modality(gene_counts, gene_meta)

# Model: Peak accessibility → Gene expression
model.fit_cis()  # Model peak accessibility
model.fit_trans(modality_name='gene', function_type='additive_hill')

# Visualize: Does this peak's accessibility predict GFI1B expression?
model.plot_xy_data('GFI1B', modality_name='gene', show_hill_function=True)
```

### Use Case 3: Dose-Response Discovery

**Goal:** Find genes with non-linear responses to perturbation

```python
# Fit with multiple function types
model.fit_trans(function_type='additive_hill')  # Non-linear
model.save_trans_fit(suffix='additive_hill')

model.fit_trans(function_type='polynomial', degree=6)
model.save_trans_fit(suffix='polynomial')

# Compare fits
from bayesDREAM.plotting import plot_model_comparison
plot_model_comparison(
    model,
    features=['TET2', 'MYB'],
    function_types=['additive_hill', 'polynomial']
)
```

---

## 🔧 Technical Improvements

### Robust Feature Lookup

**Problem:** Features could be stored in index, columns, or attributes
**Solution:** Unified lookup system that checks all locations

```python
# Works regardless of where feature is stored:
# - feature_meta.index
# - feature_meta['gene'] column
# - feature_meta['coord.intron'] column
# - modality.feature_names attribute

model.plot_xy_data('chr1:999788:999865', modality_name='splicing_sj')
# ✅ Finds feature even if stored in feature_names (not index)
```

### Type-Safe Tensor Operations

**Problem:** PyTorch tensors incompatible with pandas operations
**Solution:** Automatic scalar conversion

```python
# Internal helper ensures compatibility
def _to_scalar(val):
    if hasattr(val, 'item'):  # PyTorch tensor
        return val.item()
    elif isinstance(val, np.ndarray):  # NumPy array
        return float(val)
    return float(val)  # Already scalar

# Applied throughout plotting pipeline for seamless tensor/pandas integration
```

### Modality-Specific Save/Load

```python
# Save only what you need
model.save_technical_fit(modalities=['gene', 'atac'])

# Load selectively
model.load_technical_fit(modalities=['gene'])

# Automatic validation and error handling
try:
    model.load_technical_fit(modalities=['missing_modality'])
except ValueError as e:
    print(e)  # "Modality 'missing_modality' not found"
```

---

## 📚 Documentation Improvements

**New Comprehensive Guides:**
- [QUICKSTART_MULTIMODAL.md](docs/QUICKSTART_MULTIMODAL.md) - Quick reference
- [PLOTTING_GUIDE.md](docs/PLOTTING_GUIDE.md) - Complete plotting documentation
- [SAVE_LOAD_GUIDE.md](docs/SAVE_LOAD_GUIDE.md) - Pipeline stage management
- [API_REFERENCE.md](docs/API_REFERENCE.md) - Full API documentation
- [DATA_ACCESS.md](docs/DATA_ACCESS.md) - Data handling guide
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - System design

**Updated Developer Docs:**
- [CLAUDE.md](CLAUDE.md) - Architecture for AI assistance
- [tests/README.md](tests/README.md) - Testing infrastructure

---

## ✅ Backward Compatibility

**Critical:** All original code still works!

```python
# Original uni-modal code (from before):
model = bayesDREAM(meta=meta, counts=gene_counts, cis_gene='GFI1B')
model.fit_technical()
model.fit_cis()
model.fit_trans()

# ✅ Still works exactly as before
# ✅ No breaking changes
# ✅ Existing scripts don't need modification
```

---

## 🎓 Key Takeaways

| Aspect | Before | After |
|--------|--------|-------|
| **Modalities** | Gene expression only | Genes, transcripts, splicing, ATAC, custom |
| **Cis Feature** | Must be a gene | Any feature type (genes, peaks, junctions) |
| **Prior Integration** | Not supported | ATAC-guided cis selection |
| **Plotting** | Minimal | Comprehensive visualization suite |
| **Architecture** | 4,537-line monolith | Modular (311-line main file) |
| **Testing** | Limited | Comprehensive test suite |
| **Documentation** | Basic | 7 detailed guides + API reference |

---

## 🚀 Future Directions

**Near-term:**
- Per-modality technical fitting (separate alpha_y for each modality)
- Cross-modality trans effects (e.g., gene → splicing)
- Automated model comparison and selection

**Long-term:**
- Integration with single-cell multi-omics (CITE-seq, TEA-seq)
- Hierarchical modeling of cell types
- Real-time interactive visualization dashboard

---

## 📊 Impact Summary

The refactored bayesDREAM framework transforms a gene-specific tool into a **flexible platform for multi-modal perturbation analysis**. Key innovations include:

1. **Scientific Flexibility**: Model any molecular layer as cis or trans
2. **Prior Integration**: Leverage ATAC or other data for hypothesis-driven cis selection
3. **Comprehensive Visualization**: Publication-ready plots for all modalities
4. **Maintainable Architecture**: Modular design enabling rapid feature development
5. **Backward Compatibility**: Zero disruption to existing workflows

This positions bayesDREAM as a **mature, production-ready framework** for multi-modal CRISPR screens and perturbation studies.
