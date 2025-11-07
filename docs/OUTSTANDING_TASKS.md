# Outstanding Tasks for bayesDREAM

This document tracks remaining implementation tasks and known issues.

Last updated: 2025-01-07

---

## Recent Changes

### Cis Modality Design (2025-01-22)

**Change**: bayesDREAM now uses a separate 'cis' modality for cis gene/feature modeling.

**Design**:
- During `bayesDREAM()` initialization, the 'cis' modality is extracted from the primary modality
- The primary modality contains only trans features (cis feature excluded)
- `fit_cis()` always uses the 'cis' modality, regardless of primary modality type

**Benefits**:
- Consistent interface: `fit_cis()` works the same for gene, ATAC, or any modality
- Clear separation: cis vs trans features are explicitly separated
- Extensibility: Easy to support new modality types

**Parameters**:
```python
# For gene modality (default)
model = bayesDREAM(
    meta=meta,
    counts=gene_counts,
    cis_gene='GFI1B',
    guide_covariates=['cell_line']
)
# Creates: 'cis' modality (just GFI1B) + 'gene' modality (trans genes)

# For ATAC as primary modality (implemented via generic negbinom)
model = bayesDREAM(
    meta=meta,
    counts=atac_counts,
    modality_name='atac',
    cis_feature='chr9:123-456',
    feature_meta=region_meta,
    guide_covariates=['cell_line']
)
# Creates: 'cis' modality (chr9:123-456) + 'atac' modality (other regions)

# For any custom negbinom modality as primary
model = bayesDREAM(
    meta=meta,
    counts=custom_counts,
    modality_name='my_custom_modality',
    cis_feature='feature_123',
    feature_meta=feature_meta,
    guide_covariates=['cell_line']
)
# Creates: 'cis' modality (feature_123) + 'my_custom_modality' (other features)
```

### API Refactoring (2025-01-07)

**Change**: Cleaned up initialization parameters for clarity and extensibility.

**Removed Parameters**:
- `modalities` - Always start with empty dict, build from counts
- `primary_modality` - Replaced with `modality_name`
- `gene_meta` - Replaced with `feature_meta`

**New/Renamed Parameters**:
- `modality_name` (default='gene') - Name/type of primary modality
- `feature_meta` - General feature-level metadata for any modality
- `guide_covariates` - Now explicitly visible in signature (was implicit)

**Benefits**:
- Clearer intent: `modality_name='atac'` vs `primary_modality='atac'`
- More general: `feature_meta` works for genes, ATAC, transcripts, etc.
- Explicit parameters: `guide_covariates` no longer hidden
- Validation: Primary modality MUST be negbinom (enforced at initialization)

---

## High Priority

### 1. Implement Guide-Prior Infrastructure in fit_cis

**Status**: Infrastructure prepared, not yet integrated into _model_x

**Location**: `bayesDREAM/fitting/cis.py` lines 330-395

**Description**: The `fit_cis()` method accepts `manual_guide_effects` parameter (DataFrame with columns `guide` and `log2FC`) and prepares tensors for use as priors, but these tensors are not yet passed to or used in `_model_x()`.

**What's Done**:
- Parameter parsing and validation in `fit_cis()` (lines 337-368)
- Tensor preparation: `manual_guide_log2fc_tensor` and `manual_guide_mask_tensor`
- Pseudocode documentation explaining intended usage (lines 370-394)

**What's Needed**:
1. Pass `manual_guide_log2fc_tensor`, `manual_guide_mask_tensor`, and `prior_strength` to `_model_x()`
2. Modify `_model_x()` to use manual priors when sampling `mu_x` or `log2_x_eff_g`:
   ```python
   # Inside guides_plate:
   for g in range(G):
       if manual_guide_mask_tensor[g] == 1.0:
           # This guide has a manual prior
           prior_mean = manual_guide_log2fc_tensor[g]
           prior_sd = 1.0 / prior_strength
           # Sample with informative prior
       else:
           # Use standard hierarchical prior
   ```

**Design Decisions to Make**:
1. Should `prior_sd = 1.0 / prior_strength`, or another function?
2. Should manual effects override hierarchical priors completely, or combine them?
3. Should NTC guide always have log2FC=0 enforced, or learned?
4. How to handle cell-line-specific effects (alpha_x) with manual priors?

**Testing**: Once implemented, add test in `tests/test_manual_guide_priors.py`

---

## Medium Priority

### 2. Modality-Specific Cis/Trans Fitting

**Status**: Partially implemented

**Description**: Currently, `fit_cis()` and `fit_trans()` use the primary modality (which can now be any negbinom modality). Support for fitting cis/trans on non-primary modalities is limited.

**What Works**:
- Any negbinom modality can be primary (gene, ATAC, custom)
- `fit_cis()` extracts 'cis' modality and fits on it (works for all primary types)
- `fit_trans()` can fit any modality using `modality_name` parameter

**Current Limitations**:
- No standardized workflow for fitting splicing or other non-negbinom modalities
- No examples showing multi-modality cis/trans workflows

**What's Needed**:
1. Add examples showing how to fit_trans for splicing/custom modalities
2. Document best practices for multi-modal cis/trans workflows
3. Consider whether cis modeling makes sense for multinomial/binomial distributions

---

### 3. Save/Load for Cis Fit

**Status**: Implemented but needs testing

**Location**: `bayesDREAM/io/save.py` and `bayesDREAM/io/load.py`

**Description**: Save and load methods exist for cis fit results, but comprehensive testing is needed.

**What's Needed**:
- Test `save_cis_fit()` and `load_cis_fit()` in `tests/test_cis_save_load.py`
- Verify that loaded x_true and alpha_x_prefit produce identical results
- Test with both 'posterior' and 'point' estimate types

---

## Low Priority / Future Work

### 4. Polynomial Degree Configuration

**Status**: Hardcoded, should be configurable

**Location**: `bayesDREAM/fitting/trans.py` line 344

**Description**: Polynomial degree is currently hardcoded to 6. Should be a user-configurable parameter.

**What's Needed**:
- Expose `polynomial_degree` parameter in `fit_trans()`
- Update documentation
- Add tests for different polynomial degrees

---

### 5. Documentation Updates

**Status**: Mostly complete

**What's Done**:
- CLAUDE.md updated with modular structure and new API
- README.md updated with refactoring note and new API examples
- REFACTORING_PLAN.md marked complete
- API_REFERENCE.md updated with new initialization parameters (2025-01-07)
- QUICKSTART_MULTIMODAL.md updated with new API examples (2025-01-07)
- ARCHITECTURE.md updated with new parameter references (2025-01-07)
- INITIALIZATION.md updated with new API examples (2025-01-07)
- OUTSTANDING_TASKS.md updated with recent changes (2025-01-07)

**What's Needed**:
- Add comprehensive API reference for all public methods
- Create usage examples for each modality type
- Add troubleshooting guide
- Document common error messages and solutions

---

### 6. Example Workflows

**Status**: Basic examples exist

**Location**: `examples/` directory

**What's Needed**:
- Comprehensive multi-modal workflow example
- ATAC modality example
- Splicing modality example with all types (sj, donor, acceptor, exon_skip)
- Custom modality example (SpliZ, SpliZVD)
- Example showing save/load workflow
- Example showing guide-prior usage (after #1 is implemented)

---

### 7. Performance Optimization

**Status**: Not started

**Ideas**:
- Profile code to identify bottlenecks
- Optimize tensor operations in Pyro models
- Consider GPU memory optimization strategies
- Investigate minibatch training for very large datasets

---

### 8. Additional Distributions

**Status**: Core distributions implemented

**Implemented**: negbinom, multinomial, binomial, normal, mvnormal

**Potential Additions**:
- Zero-inflated negative binomial
- Beta distribution (for proportions)
- Gamma distribution (for non-negative continuous data)

---

## Completed Recently ✓

- ✅ Modular refactoring (reduced model.py from 4537 to 311 lines)
- ✅ Delegation pattern for fitters, savers, loaders
- ✅ Multi-modal infrastructure
- ✅ Distribution registry system
- ✅ Per-modality save/load
- ✅ Test cleanup and updates
- ✅ Import fixes in trans.py (find_beta, Hill_based_positive, etc.)
- ✅ Model-level posterior_samples_trans storage fix
- ✅ Multinomial per-group zero-category masking (2025-01-05)
- ✅ Binomial per-group boundary checking (2025-01-05)
- ✅ Empirical Bayes initialization for binomial and multinomial (2025-01-04)
- ✅ Feature_names preservation bug fix in get_cell_subset() (2025-01-07)
- ✅ Plotting double-subsetting bug fix in xy_plots.py (2025-01-07)
- ✅ API refactoring: modality_name, feature_meta, guide_covariates parameters (2025-01-07)
- ✅ Generic negbinom modality creation - ATAC/custom as primary now supported (2025-01-07)
- ✅ Documentation updates for new API across 5 major docs (2025-01-07)

---

## Known Issues

### None currently

If you discover issues, please document them here with:
- Description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Suggested fix (if known)

---

## Notes

- All public API methods should maintain backward compatibility
- Tests should pass before merging new features
- Documentation should be updated alongside code changes
- Use TodoWrite tool for tracking active work
