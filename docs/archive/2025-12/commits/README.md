# Commit Analysis Documentation (December 2025)

This directory contains documentation analyzing specific commits and code changes made during December 2025.

## Files

### BINOMIAL_CHANGES_SINCE_BCD1C4F.md
Analysis of changes to binomial trans fitting since commit bcd1c4f. Documents:
- Critical bug in alpha_y_full construction (using ones instead of zeros)
- Hill function parameterization changes (Vmax_sum instead of Vmax_a/Vmax_b)
- Removal of [0,1] clamping for binomial predictions
- Impact on statistical power and numerical stability

**Status**: Bug fixed in commit 1db8b2a

### MULTINOMIAL_TECHNICAL_CORRECTION.md
Implementation guide for multinomial technical correction in fit_trans. Documents:
- How to enable technical correction for multinomial (matching binomial)
- Shape handling for 3D ([C,T,K]) and 4D ([S,C,T,K]) alpha_y tensors
- Dirichlet priors for A and upper_limit
- K-1 independent Hill functions with Kth residual
- K independent polynomials in logit space

**Status**: Implemented in commit 59f08d9

### DIFF_ANALYSIS_65a0502_TO_HEAD.md
Comprehensive diff analysis comparing commit 65a0502 to HEAD. Likely covers technical fitting changes and prior computation updates.

---

## Historical Context

These documents capture important debugging and implementation work:
1. Fixed critical alpha_y_full bug that was causing flat fits
2. Enabled multinomial to work like binomial for technical correction
3. Analyzed architectural changes and their impact on power

## Related Documentation

- `../SUBSETTING_NOTES.md` - Notes on subsetting analyses
- `../debug_binomial_fits.md` - Earlier binomial debugging notes
- `../../BINOMIAL_ADDITIVE_HILL_TRACE.md` - Trace of binomial implementation
