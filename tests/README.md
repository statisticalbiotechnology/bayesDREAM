# bayesDREAM Tests

This directory contains test scripts for bayesDREAM functionality.

## Current Test Suite

### Core Functionality Tests

- **`test_negbinom_compat.py`** - Backward compatibility test for negative binomial distribution
- **`test_technical_compat.py`** - Backward compatibility test for `fit_technical()` with negbinom
- **`test_multimodal_fitting.py`** - Infrastructure test for multi-modal fitting

### Feature-Specific Tests

- **`test_exon_skip_aggregation.py`** - Tests exon skipping aggregation methods (min vs mean)
- **`test_filtering.py`** - Tests distribution-specific filtering at modality creation
- **`test_filtering_simple.py`** - Simplified filtering tests with clear expected outcomes
- **`test_per_modality_fitting.py`** - Tests per-modality technical fitting with different distributions

## Running Tests

Tests require the `pyroenv` conda environment:

```bash
# Run a specific test
/opt/anaconda3/envs/pyroenv/bin/python tests/test_filtering_simple.py

# Run all tests (if you have pytest)
/opt/anaconda3/envs/pyroenv/bin/python -m pytest tests/
```

## Archived Tests

The `archive/` subdirectory contains older debugging and development test scripts that are kept for reference but are no longer actively maintained.
