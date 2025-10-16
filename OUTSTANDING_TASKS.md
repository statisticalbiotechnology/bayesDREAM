# Outstanding Tasks for bayesDREAM

Last updated: 2025-10-16

## High Priority

### 1. Per-Modality Fitting Implementation
**Status**: Architecture complete, implementation needed

The infrastructure is ready for per-modality technical and trans fitting:
- ✅ Distribution-specific samplers implemented
- ✅ Filtering at modality creation working
- ✅ Per-modality storage structures in place
- ⏳ Need to implement `fit_technical(modality_name='...')`
- ⏳ Need to implement `fit_trans(modality_name='...')`

**Implementation plan**: See [docs/PER_MODALITY_FITTING_PLAN.md](docs/PER_MODALITY_FITTING_PLAN.md)

### 2. Exon Skipping Edge Cases
**Status**: Core implementation complete, edge cases need testing

Recent improvements:
- ✅ Added informative processing headers
- ✅ Implemented SJ count validation (numerator ≤ denominator)
- ✅ Added aggregation method recovery
- ⏳ Test with real data to identify edge cases
- ⏳ Validate strand-aware vs genomic coordinate detection

### 3. Documentation Polish
**Status**: Recently reorganized, needs final review

Recent changes:
- ✅ Moved docs to `docs/` directory
- ✅ Archived old documentation
- ✅ Created docs/README.md and tests/README.md
- ✅ Updated main README with correct paths
- ⏳ Review all docs for accuracy
- ⏳ Add more usage examples

## Medium Priority

### 4. Testing Suite Expansion
**Status**: Core tests exist, need more coverage

Current tests:
- ✅ Backward compatibility tests
- ✅ Filtering tests
- ✅ Exon skipping aggregation tests
- ⏳ Add integration tests for complete workflows
- ⏳ Add tests for edge cases (empty data, single feature, etc.)
- ⏳ Add performance benchmarks

### 5. Error Handling Improvements
**Status**: Basic error handling exists, needs enhancement

Areas to improve:
- ⏳ Better error messages for data format issues
- ⏳ Validation of user inputs (e.g., check column names early)
- ⏳ Warnings for potential issues (e.g., very few NTC cells)
- ⏳ Recovery suggestions in error messages

### 6. Visualization Tools
**Status**: Not implemented

Potential additions:
- ⏳ Plot dose-response curves
- ⏳ Visualize posterior distributions
- ⏳ QC plots for technical fit
- ⏳ Splicing event browser

## Low Priority

### 7. Performance Optimization
**Status**: Not yet profiled

Potential improvements:
- ⏳ Profile code to identify bottlenecks
- ⏳ Optimize splicing event detection
- ⏳ Consider caching frequently computed values
- ⏳ GPU optimization for large datasets

### 8. Additional Features
**Status**: Ideas for future enhancement

Possibilities:
- ⏳ Support for time-series perturbations
- ⏳ Batch effect correction
- ⏳ Integration with other single-cell tools
- ⏳ Automated hyperparameter tuning

### 9. Package Distribution
**Status**: Development mode only

To do:
- ⏳ Create proper setup.py/pyproject.toml
- ⏳ Publish to PyPI
- ⏳ Create conda package
- ⏳ Add version numbering system
- ⏳ Create CHANGELOG.md

## Completed Recently ✅

- ✅ Exon skipping flexible aggregation (min/mean)
- ✅ Distribution-specific filtering at modality creation
- ✅ Informative splicing processing headers
- ✅ SJ count validation (numerator ≤ denominator check)
- ✅ Documentation reorganization
- ✅ Test suite organization
- ✅ CLAUDE.md architecture documentation
- ✅ API reference documentation
- ✅ Data access guide

## Notes for Future Development

### Design Decisions to Reconsider

1. **Primary modality concept**: Currently only the primary (usually 'gene') modality is used for cis/trans fitting. Future versions should support arbitrary modality selection for fitting.

2. **Sum factor calculation**: Currently gene-specific. May need modality-specific normalization strategies.

3. **Permutation testing**: Currently operates on primary modality only. Should extend to other modalities.

### Technical Debt

1. The base `bayesDREAM` class in `model.py` is large (~2250 lines). Consider refactoring into smaller modules.

2. Some parameter names are inconsistent (e.g., `sum_factor_col` vs `sum_factor`). A breaking change to standardize would be beneficial.

3. The relationship between `MultiModalBayesDREAM` and base `bayesDREAM` could be cleaner - currently uses inheritance which creates some complexity.

## Getting Help

- **Questions**: Open an issue at https://github.com/leahrosen/bayesDREAM/issues
- **Bugs**: Provide reproducible example in a new issue
- **Feature requests**: Describe use case in a new issue
