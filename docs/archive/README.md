# Documentation Archive

This directory contains historical documentation from various development phases of bayesDREAM.

**Note**: These documents are **archived for reference only**. They may contain outdated information and should not be used as current documentation. Refer to the main docs directory for up-to-date documentation.

---

## Archive Contents

### design/
Design documents and implementation summaries for specific features (October 2025):
- `ATAC_DESIGN.md` - Original ATAC-seq modality design
- `ATAC_IMPLEMENTATION_SUMMARY.md` - ATAC implementation details
- `TECHNICAL_FITTING_CIS_GENE.md` - Technical fitting for cis genes

**Status**: Superseded by current ARCHITECTURE.md and API_REFERENCE.md

### planning/
Planning documents for major refactoring efforts (October 2025):
- `PER_MODALITY_FITTING_PLAN.md` - Original per-modality fitting design
- `REFACTORING_PLAN.md` - Major refactoring plan (completed)
- `REFACTORING_SUMMARY.md` - Refactoring completion summary

**Status**: Refactoring completed. Current architecture documented in ARCHITECTURE.md

### verification/
Verification and testing documentation (October 2025):
- `MODALITY_SAVE_LOAD_VERIFICATION.md` - Save/load functionality verification

**Status**: Save/load functionality documented in SAVE_LOAD_GUIDE.md

### Implementation Summaries
Historical implementation summaries from multi-modal development:
- `MULTIMODAL_IMPLEMENTATION.md` - Original multi-modal implementation notes
- `MULTIMODAL_FITTING_INFRASTRUCTURE.md` - Infrastructure design notes
- `PER_MODALITY_FITTING_COMPLETE.md` - Per-modality fitting completion
- `PER_MODALITY_FITTING_SUMMARY.md` - Per-modality fitting summary
- `DOCUMENTATION_UPDATE_SUMMARY.md` - Documentation update notes

**Status**: Current implementation documented in ARCHITECTURE.md and API_REFERENCE.md

---

## Current Documentation

For current, up-to-date documentation, see:

- **[API_REFERENCE.md](../API_REFERENCE.md)** - Complete API documentation
- **[ARCHITECTURE.md](../ARCHITECTURE.md)** - System architecture
- **[DATA_ACCESS.md](../DATA_ACCESS.md)** - Data access guide
- **[SAVE_LOAD_GUIDE.md](../SAVE_LOAD_GUIDE.md)** - Save/load guide
- **[QUICKSTART_MULTIMODAL.md](../QUICKSTART_MULTIMODAL.md)** - Quick start guide

---

## Archive Policy

Documents are moved to archive when:
1. The feature/refactoring is complete and stable
2. Current documentation covers the topic comprehensively
3. The document is primarily of historical interest

Archived documents are kept for:
- Understanding design decisions
- Historical context
- Tracking evolution of the codebase
