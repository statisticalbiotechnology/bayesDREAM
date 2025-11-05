# Outstanding Tasks for bayesDREAM

This document tracks remaining implementation tasks and known issues.

Last updated: 2025-01-05

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
# For gene modality
model = bayesDREAM(counts=gene_counts, cis_gene='GFI1B', ...)
# Creates: 'cis' modality (just GFI1B) + 'gene' modality (trans genes)

# For ATAC (not yet implemented as primary, use add_atac_modality with cis_region)
model = bayesDREAM(counts=gene_counts, cis_gene=None)
model.add_atac_modality(atac_counts, region_meta, cis_region='chr9:123-456')
# Creates/updates: 'cis' modality (chr9:123-456)
```

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

**Status**: Not implemented

**Description**: Currently, `fit_cis()` and `fit_trans()` primarily use the primary modality. Should support fitting cis/trans effects for non-primary modalities.

**Current Limitations**:
- `fit_cis()` always uses primary modality (by design - see line 237-244)
- `fit_trans()` can fit any modality but requires manual setup
- No standardized workflow for fitting ATAC or splicing modalities

**What's Needed**:
1. Determine if cis modeling makes sense for non-gene modalities
2. If yes, generalize `fit_cis()` to handle different modality types
3. Add examples showing how to fit_trans for splicing/ATAC/custom modalities

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

**Status**: Partially complete

**What's Done**:
- CLAUDE.md updated with modular structure
- README.md updated with refactoring note
- REFACTORING_PLAN.md marked complete

**What's Needed**:
- Add API reference documentation for all public methods
- Create usage examples for each modality type
- Document distribution-specific requirements
- Add troubleshooting guide

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
