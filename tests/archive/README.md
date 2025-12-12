# Archived Test and Diagnostic Scripts

This directory contains diagnostic and test scripts that were used during development but are no longer actively maintained.

## Scripts

### check_defensive_code.py
Checks if defensive NaN/Inf handling is present in distributions.py. Used to verify that the binomial sampler has protective code against non-finite log probability values during Predictive sampling.

**Historical Context**: Part of fixing NaN/Inf errors during fit_technical() with binomial distributions (December 2025).

### diagnose_alpha_y.py
Diagnostic tool for investigating alpha_y (technical correction) parameters. Used to check if technical correction parameters are properly inverted when computing priors.

**Historical Context**: Part of debugging inverted priors issue where fits were consistently too low (November-December 2025).

---

## Note

These scripts are kept for reference but may not work with the current codebase. For active diagnostic tools, see `examples/diagnostics/`.
