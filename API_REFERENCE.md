# bayesDREAM API Reference

Complete reference for all classes, methods, and functions in bayesDREAM.

## Table of Contents

- [MultiModalBayesDREAM Class](#multimodalbayesdream-class)
- [Modality Class](#modality-class)
- [Distribution Functions](#distribution-functions)
- [Splicing Functions](#splicing-functions)
- [Utility Functions](#utility-functions)

---

## MultiModalBayesDREAM Class

The main class for multi-modal Bayesian modeling of CRISPR perturbation effects.

### Initialization

```python
MultiModalBayesDREAM(
    meta,
    counts,
    cis_gene,
    primary_modality='gene',
    output_dir=None,
    label=None,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
```

**Parameters:**
- `meta` (pd.DataFrame): Cell metadata with required columns:
  - `cell`: Cell barcode
  - `guide`: Guide RNA identifier
  - `target`: Target gene (use `'ntc'` for non-targeting controls)
  - `sum_factor`: Normalization factor per cell
  - `cell_line`: Cell line identifier
- `counts` (pd.DataFrame): Count matrix with features as rows, cells as columns
- `cis_gene` (str): Name of the gene to model cis effects for
- `primary_modality` (str, optional): Name for primary modality. Default: `'gene'`
- `output_dir` (str, optional): Directory for saving results
- `label` (str, optional): Label for this analysis run
- `device` (str, optional): PyTorch device (`'cuda'` or `'cpu'`)

**Returns:** MultiModalBayesDREAM instance

---

### Adding Modalities

#### add_modality()

```python
model.add_modality(name, modality, overwrite=False)
```

Add a pre-constructed Modality object.

**Parameters:**
- `name` (str): Name for this modality
- `modality` (Modality): Pre-constructed Modality object
- `overwrite` (bool): Whether to overwrite if name exists. Default: False

**Example:**
```python
custom_mod = Modality(
    name='my_data',
    counts=data_array,
    feature_meta=feature_df,
    distribution='normal'
)
model.add_modality('my_data', custom_mod)
```

---

#### add_transcript_modality()

```python
model.add_transcript_modality(
    transcript_counts,
    transcript_meta,
    modality_types='counts',
    gene_col=None
)
```

Add transcript-level data as counts and/or isoform usage.

**Parameters:**
- `transcript_counts` (pd.DataFrame): Transcript counts (transcripts × cells)
- `transcript_meta` (pd.DataFrame): Transcript metadata with required columns:
  - `transcript_id`: Transcript identifier
  - One of: `gene`, `gene_name`, or `gene_id` (gene assignment)
- `modality_types` (str or list): Type(s) to create:
  - `'counts'`: Transcript counts as negative binomial (2D)
  - `'usage'`: Isoform usage as multinomial (3D, grouped by gene)
  - `['counts', 'usage']`: Create both
- `gene_col` (str, optional): Column name for gene assignment. Auto-detects from: `gene`, `gene_name`, `gene_id`

**Modalities Created:**
- `'transcript_counts'`: Distribution = `negbinom`, shape = (transcripts, cells)
- `'transcript_usage'`: Distribution = `multinomial`, shape = (genes, cells, max_isoforms)

**Example:**
```python
model.add_transcript_modality(
    transcript_counts=tx_counts,
    transcript_meta=tx_meta,
    modality_types=['counts', 'usage']
)
```

---

#### add_splicing_modality()

```python
model.add_splicing_modality(
    sj_counts,
    sj_meta,
    splicing_types=['donor', 'acceptor'],
    gene_counts=None,
    min_cell_total=1,
    min_total_exon=2,
    exon_aggregate='min'
)
```

Add splicing data (splice junctions, donor/acceptor usage, exon skipping).

**Parameters:**
- `sj_counts` (pd.DataFrame): Splice junction counts (junctions × cells)
- `sj_meta` (pd.DataFrame): Junction metadata with required columns:
  - `coord.intron`: Junction ID (e.g., "chr1:12345:67890:+")
  - `chrom`: Chromosome
  - `intron_start`, `intron_end`: Junction coordinates
  - `strand`: Strand ('+', '-', 1, or 2)
  - `gene_name_start`, `gene_name_end`: Gene names at junction boundaries
  - Optional: `gene_id_start`, `gene_id_end` (Ensembl IDs)
- `splicing_types` (list): Types to create:
  - `'sj'`: Raw SJ counts (binomial with gene expression denominator)
  - `'donor'`: Donor site usage (multinomial)
  - `'acceptor'`: Acceptor site usage (multinomial)
  - `'exon_skip'`: Exon skipping events (binomial)
- `gene_counts` (pd.DataFrame, optional): Gene counts for 'sj' type denominator. Defaults to `self.counts`
- `min_cell_total` (int): Minimum reads per donor/acceptor. Default: 1
- `min_total_exon` (int): Minimum reads for exon skipping. Default: 2
- `exon_aggregate` (str): How to aggregate inc1/inc2 for exon skipping: `'min'` or `'mean'`

**Modalities Created:**
- `'splicing_sj'`: Distribution = `binomial`, shape = (junctions, cells)
- `'splicing_donor'`: Distribution = `multinomial`, shape = (donors, cells, acceptors)
- `'splicing_acceptor'`: Distribution = `multinomial`, shape = (acceptors, cells, donors)
- `'splicing_exon_skip'`: Distribution = `binomial`, shape = (events, cells)

**Example:**
```python
model.add_splicing_modality(
    sj_counts=sj_counts,
    sj_meta=sj_meta,
    splicing_types=['sj', 'donor', 'acceptor', 'exon_skip']
)
```

---

#### add_custom_modality()

```python
model.add_custom_modality(
    name,
    counts,
    feature_meta,
    distribution,
    denominator=None
)
```

Add a custom modality with user-defined measurements.

**Parameters:**
- `name` (str): Name for this modality
- `counts` (np.ndarray or pd.DataFrame): Measurement data
  - 2D for: `negbinom`, `normal`, `binomial`
  - 3D for: `multinomial`, `mvnormal`
- `feature_meta` (pd.DataFrame): Feature-level metadata
- `distribution` (str): Distribution type:
  - `'negbinom'`: Negative binomial (count data)
  - `'normal'`: Normal (continuous measurements)
  - `'binomial'`: Binomial (proportions with denominator)
  - `'multinomial'`: Multinomial (categorical)
  - `'mvnormal'`: Multivariate normal (multi-dimensional)
- `denominator` (np.ndarray, optional): Required for `binomial` distribution

**Example:**
```python
# SpliZ scores (normal distribution)
model.add_custom_modality(
    name='spliz',
    counts=spliz_scores,
    feature_meta=gene_meta,
    distribution='normal'
)

# SpliZVD (multivariate normal, 3D)
model.add_custom_modality(
    name='splizvd',
    counts=splizvd_array,  # shape: (genes, cells, 3)
    feature_meta=gene_meta,
    distribution='mvnormal'
)
```

---

### Accessing Modalities

#### get_modality()

```python
model.get_modality(name)
```

Retrieve a specific modality by name.

**Parameters:**
- `name` (str): Modality name

**Returns:** Modality object

**Raises:** KeyError if modality doesn't exist

**Example:**
```python
donor_mod = model.get_modality('splicing_donor')
print(donor_mod.dims)  # {'n_features': 100, 'n_cells': 500, 'n_categories': 10}
```

---

#### list_modalities()

```python
model.list_modalities()
```

Get a summary table of all modalities.

**Returns:** pd.DataFrame with columns:
- `name`: Modality name
- `distribution`: Distribution type
- `n_features`: Number of features
- `n_cells`: Number of cells
- `n_categories` (if applicable): Number of categories (multinomial)
- `n_dims` (if applicable): Number of dimensions (mvnormal)

**Example:**
```python
print(model.list_modalities())
#                   name distribution  n_features  n_cells  n_categories
# 0                 gene     negbinom        1000      500           NaN
# 1    splicing_donor  multinomial          50      500          10.0
# 2                spliz       normal        1000      500           NaN
```

---

### Modeling Pipeline

#### fit_technical()

```python
model.fit_technical(
    covariates,
    sum_factor_col=None,
    distribution='negbinom',
    denominator=None,
    n_steps=5000,
    lr=0.01,
    device=None
)
```

Fit technical model to estimate baseline overdispersion from non-targeting controls.

**Parameters:**
- `covariates` (list of str): Covariates to model (e.g., `['cell_line']`)
- `sum_factor_col` (str, optional): Column in `meta` with normalization factors. Not required for `normal`/`mvnormal`
- `distribution` (str): Distribution type. Default: `'negbinom'`
  - `'negbinom'`: Negative binomial (requires `sum_factor_col`)
  - `'normal'`: Normal (no sum factors needed)
  - `'binomial'`: Binomial (requires `denominator`)
  - `'multinomial'`: Multinomial (3D data)
  - `'mvnormal'`: Multivariate normal (3D data)
- `denominator` (np.ndarray, optional): Required for `binomial`
- `n_steps` (int): Number of optimization steps. Default: 5000
- `lr` (float): Learning rate. Default: 0.01
- `device` (str, optional): PyTorch device. Defaults to `self.device`

**Side Effects:**
- Sets `self.alpha_y_prefit` (overdispersion parameters)
- For `normal`/`mvnormal`, sets `self.sigma_y_prefit` or `self.cov_y_prefit`

**Example:**
```python
# Gene counts (negbinom)
model.fit_technical(covariates=['cell_line'], sum_factor_col='sum_factor')

# Continuous scores (normal)
model.fit_technical(covariates=['cell_line'], distribution='normal')
```

---

#### fit_cis()

```python
model.fit_cis(
    sum_factor_col='sum_factor',
    n_steps=5000,
    lr=0.01,
    device=None
)
```

Fit cis model to estimate direct effects on targeted gene.

**Parameters:**
- `sum_factor_col` (str): Column with normalization factors
- `n_steps` (int): Number of optimization steps. Default: 5000
- `lr` (float): Learning rate. Default: 0.01
- `device` (str, optional): PyTorch device

**Side Effects:**
- Sets `self.x_true` (posterior cis expression per guide)
- Sets `self.posterior_samples_cis` (full posterior samples)

**Example:**
```python
model.fit_cis(sum_factor_col='sum_factor')
```

---

#### fit_trans()

```python
model.fit_trans(
    sum_factor_col=None,
    function_type='additive_hill',
    distribution='negbinom',
    denominator=None,
    n_steps=10000,
    lr=0.01,
    device=None
)
```

Fit trans model to estimate downstream effects as a function of cis expression.

**Parameters:**
- `sum_factor_col` (str, optional): Column with normalization factors. Required for `negbinom`
- `function_type` (str): Dose-response function type:
  - `'single_hill'`: Single Hill equation
  - `'additive_hill'`: Sum of positive and negative Hill functions
  - `'polynomial'`: Polynomial (default degree: 6)
- `distribution` (str): Distribution type. Default: `'negbinom'`
- `denominator` (np.ndarray, optional): Required for `binomial`
- `n_steps` (int): Number of optimization steps. Default: 10000
- `lr` (float): Learning rate. Default: 0.01
- `device` (str, optional): PyTorch device

**Side Effects:**
- Sets `self.posterior_samples_trans` (dose-response parameters)

**Example:**
```python
# Gene counts (negbinom)
model.fit_trans(
    sum_factor_col='sum_factor_adj',
    function_type='additive_hill',
    distribution='negbinom'
)

# Continuous (normal)
model.fit_trans(
    distribution='normal',
    function_type='polynomial'
)

# Exon skipping (binomial)
model.fit_trans(
    distribution='binomial',
    denominator=total_counts,
    function_type='single_hill'
)
```

---

### Utility Methods

#### adjust_ntc_sum_factor()

```python
model.adjust_ntc_sum_factor(
    covariates,
    sum_factor_col_old='sum_factor',
    sum_factor_col_adj='sum_factor_adj'
)
```

Adjust sum factors based on NTC mean within covariate groups.

**Parameters:**
- `covariates` (list of str): Covariates to group by
- `sum_factor_col_old` (str): Input sum factor column. Default: `'sum_factor'`
- `sum_factor_col_adj` (str): Output adjusted column. Default: `'sum_factor_adj'`

**Side Effects:**
- Adds `sum_factor_col_adj` column to `self.meta`
- Adds `adjustment_factor` column to `self.meta`

**Example:**
```python
model.adjust_ntc_sum_factor(covariates=['lane', 'cell_line'])
```

---

#### set_alpha_x()

```python
model.set_alpha_x(alpha_x, is_posterior=False, covariates=None)
```

Set overdispersion parameters for cis gene.

**Parameters:**
- `alpha_x` (array-like): Overdispersion values
- `is_posterior` (bool): Whether these are posterior samples. Default: False
- `covariates` (list, optional): Covariate names if applicable

---

#### set_alpha_y()

```python
model.set_alpha_y(alpha_y, is_posterior=False, covariates=None)
```

Set overdispersion parameters for trans genes.

**Parameters:**
- `alpha_y` (array-like): Overdispersion values
- `is_posterior` (bool): Whether these are posterior samples. Default: False
- `covariates` (list, optional): Covariate names if applicable

---

#### set_x_true()

```python
model.set_x_true(x_true, is_posterior=False)
```

Set cis expression values (e.g., from external estimates).

**Parameters:**
- `x_true` (array-like): Cis expression values
- `is_posterior` (bool): Whether these are posterior samples. Default: False

---

#### permute_genes()

```python
model.permute_genes(
    genes2permute,
    sum_factor_col='sum_factor',
    function_type='additive_hill',
    n_steps=10000,
    lr=0.01
)
```

Permute guide-gene associations for null testing.

**Parameters:**
- `genes2permute` (list): Gene names to permute
- `sum_factor_col` (str): Normalization factor column
- `function_type` (str): Dose-response function
- `n_steps` (int): Optimization steps
- `lr` (float): Learning rate

**Side Effects:**
- Sets `self.posterior_samples_trans_permuted`

---

#### refit_sumfactor()

```python
model.refit_sumfactor(
    covariates,
    sum_factor_col_old='sum_factor',
    sum_factor_col_new='sum_factor_refit'
)
```

Re-estimate sum factors based on posterior cis expression.

**Parameters:**
- `covariates` (list): Covariates for adjustment
- `sum_factor_col_old` (str): Input sum factor column
- `sum_factor_col_new` (str): Output refitted column

---

## Modality Class

Container for modality-specific data and metadata.

### Initialization

```python
Modality(
    name,
    counts,
    feature_meta,
    distribution,
    denominator=None
)
```

**Parameters:**
- `name` (str): Modality name
- `counts` (np.ndarray): Measurement data
  - 2D `(features, cells)`: negbinom, normal, binomial
  - 3D `(features, cells, categories)`: multinomial
  - 3D `(features, cells, dims)`: mvnormal
- `feature_meta` (pd.DataFrame): Feature-level metadata
- `distribution` (str): One of: `'negbinom'`, `'normal'`, `'binomial'`, `'multinomial'`, `'mvnormal'`
- `denominator` (np.ndarray, optional): For binomial only

**Attributes:**
- `name`: Modality name
- `counts`: Data array
- `feature_meta`: Metadata DataFrame
- `distribution`: Distribution type
- `denominator`: Denominator array (binomial only)
- `dims`: Dictionary with dimensions:
  - `n_features`: Number of features
  - `n_cells`: Number of cells
  - `n_categories` (multinomial): Number of categories
  - `n_dims` (mvnormal): Number of dimensions

---

### Methods

#### to_tensor()

```python
modality.to_tensor(device='cpu')
```

Convert counts to PyTorch tensor.

**Parameters:**
- `device` (str): PyTorch device

**Returns:** torch.Tensor

---

#### get_feature_subset()

```python
modality.get_feature_subset(feature_indices)
```

Subset to specific features.

**Parameters:**
- `feature_indices` (list or array): Feature indices or names

**Returns:** New Modality object

---

#### get_cell_subset()

```python
modality.get_cell_subset(cell_indices)
```

Subset to specific cells.

**Parameters:**
- `cell_indices` (list or array): Cell indices

**Returns:** New Modality object

---

## Distribution Functions

From `bayesDREAM.distributions` module.

### get_observation_sampler()

```python
get_observation_sampler(distribution, model_type)
```

Get appropriate observation sampler for a distribution and model.

**Parameters:**
- `distribution` (str): Distribution name
- `model_type` (str): Either `'technical'` or `'trans'`

**Returns:** Sampler function

**Example:**
```python
from bayesDREAM import get_observation_sampler
sampler = get_observation_sampler('multinomial', 'trans')
```

---

### requires_denominator()

```python
requires_denominator(distribution)
```

Check if distribution requires a denominator.

**Parameters:**
- `distribution` (str): Distribution name

**Returns:** bool (True for `'binomial'`)

---

### requires_sum_factor()

```python
requires_sum_factor(distribution)
```

Check if distribution requires sum factors.

**Parameters:**
- `distribution` (str): Distribution name

**Returns:** bool (True for `'negbinom'`)

---

### is_3d_distribution()

```python
is_3d_distribution(distribution)
```

Check if distribution uses 3D data.

**Parameters:**
- `distribution` (str): Distribution name

**Returns:** bool (True for `'multinomial'` and `'mvnormal'`)

---

### supports_cell_line_effects()

```python
supports_cell_line_effects(distribution)
```

Check if distribution supports cell-line covariate effects.

**Parameters:**
- `distribution` (str): Distribution name

**Returns:** bool

---

### get_cell_line_effect_type()

```python
get_cell_line_effect_type(distribution)
```

Get how cell-line effects are applied for a distribution.

**Parameters:**
- `distribution` (str): Distribution name

**Returns:** str - One of:
- `'multiplicative'`: Effect multiplies mu (negbinom)
- `'additive'`: Effect adds to mu (normal, mvnormal)
- `'logit'`: Effect on logit scale (binomial)
- `None`: Not supported (multinomial)

---

## Splicing Functions

From `bayesDREAM.splicing` module.

### create_splicing_modality()

```python
create_splicing_modality(
    sj_counts,
    sj_meta,
    splicing_type,
    gene_counts=None,
    min_cell_total=1,
    min_total_exon=2,
    exon_aggregate='min'
)
```

High-level function to create a splicing modality.

**Parameters:**
- `sj_counts` (pd.DataFrame): Splice junction counts
- `sj_meta` (pd.DataFrame): Junction metadata
- `splicing_type` (str): Type to create (`'sj'`, `'donor'`, `'acceptor'`, `'exon_skip'`)
- `gene_counts` (pd.DataFrame, optional): Gene counts for 'sj' denominator
- `min_cell_total` (int): Minimum reads for donor/acceptor
- `min_total_exon` (int): Minimum reads for exon skipping
- `exon_aggregate` (str): Aggregation method for exon inclusion

**Returns:** Modality object

---

### process_donor_usage()

```python
process_donor_usage(sj_counts, sj_meta, min_cell_total=1)
```

Process splice junctions into donor site usage.

**Returns:** Tuple of (counts_3d, metadata_df)

---

### process_acceptor_usage()

```python
process_acceptor_usage(sj_counts, sj_meta, min_cell_total=1)
```

Process splice junctions into acceptor site usage.

**Returns:** Tuple of (counts_3d, metadata_df)

---

### process_exon_skipping()

```python
process_exon_skipping(
    sj_counts,
    sj_meta,
    min_total=2,
    aggregate='min'
)
```

Detect cassette exon triplets and compute inclusion/total.

**Returns:** Tuple of (inclusion_2d, total_2d, metadata_df)

---

### process_sj_counts()

```python
process_sj_counts(sj_counts, sj_meta, gene_counts)
```

Process raw splice junction counts with gene expression denominator.

**Returns:** Tuple of (sj_counts_2d, gene_denominator_2d, metadata_df)

---

## Utility Functions

From `bayesDREAM.utils` module.

### Hill_based_positive()

```python
Hill_based_positive(x, params)
```

Positive Hill equation (activation).

**Parameters:**
- `x` (torch.Tensor): Input values
- `params` (torch.Tensor): Parameters [B, K, EC50]

**Returns:** torch.Tensor

---

### Hill_based_negative()

```python
Hill_based_negative(x, params)
```

Negative Hill equation (repression).

**Parameters:**
- `x` (torch.Tensor): Input values
- `params` (torch.Tensor): Parameters [B, K, IC50]

**Returns:** torch.Tensor

---

### Hill_based_piecewise()

```python
Hill_based_piecewise(x, params)
```

Piecewise Hill equation (can be positive or negative).

**Parameters:**
- `x` (torch.Tensor): Input values
- `params` (torch.Tensor): Parameters [B, K, xc, indicator]

**Returns:** torch.Tensor

---

### Polynomial_function()

```python
Polynomial_function(x, params, degree=6)
```

Polynomial dose-response function.

**Parameters:**
- `x` (torch.Tensor): Input values
- `params` (torch.Tensor): Polynomial coefficients
- `degree` (int): Polynomial degree. Default: 6

**Returns:** torch.Tensor

---

### find_beta()

```python
find_beta(mu, phi, method='mle', tolerance=1e-9)
```

Numerical solver for beta parameter in negative binomial.

**Parameters:**
- `mu` (float): Mean parameter
- `phi` (float): Overdispersion parameter
- `method` (str): Solving method. Default: 'mle'
- `tolerance` (float): Convergence tolerance

**Returns:** float

---

### check_tensor()

```python
check_tensor(x, name='tensor')
```

Check tensor for NaN/Inf values.

**Parameters:**
- `x` (torch.Tensor): Tensor to check
- `name` (str): Name for error message

**Raises:** ValueError if NaN or Inf detected

---

## Distribution Registry

The `DISTRIBUTION_REGISTRY` dictionary maps distribution names to their samplers:

```python
from bayesDREAM import DISTRIBUTION_REGISTRY

print(DISTRIBUTION_REGISTRY.keys())
# dict_keys(['negbinom', 'multinomial', 'binomial', 'normal', 'mvnormal'])
```

Each entry contains:
- `'technical'`: Sampler function for technical model
- `'trans'`: Sampler function for trans model
- `'requires_denominator'`: bool
- `'requires_sum_factor'`: bool
- `'is_3d'`: bool
- `'supports_cell_line_effects'`: bool
- `'cell_line_effect_type'`: str or None
