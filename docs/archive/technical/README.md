# Technical Documentation Archive

This directory contains historical technical documentation from development and debugging.

## Contents

### BINOMIAL_ADDITIVE_HILL_TRACE.md
**Created**: November 2025
**Purpose**: Detailed execution flow trace for binomial trans fitting with additive Hill functions

Complete walkthrough of the code path when calling `fit_trans()` with binomial distribution. Used during development to understand and debug the binomial fitting pipeline.

### HIGH_MOI_DESIGN.md
**Created**: November 2025
**Purpose**: Design document for high MOI (multiple guides per cell) support

Technical design for additive guide effects in cis modeling. The user-facing guide is now `HIGH_MOI_GUIDE.md` in the main docs/ directory.

### TECHNICAL_CORRECTION_IN_PRIORS.md
**Created**: November 27, 2025
**Status**: Fixed in commit fa959a4
**Purpose**: Documentation of critical bug fix in trans prior computation

Explains how priors were being computed from biased data that included technical batch effects, and documents the inverse correction fix. Important for understanding why the technical correction is applied before computing guide-level statistics for A and Vmax priors.

---

**Note**: These documents are kept for historical reference but may contain outdated implementation details. Always refer to current documentation in `docs/` for up-to-date information.
