# bayesDREAM Documentation

Comprehensive documentation for the bayesDREAM multi-modal Bayesian framework.

## Getting Started

- **[Main README](../README.md)** - Project overview and installation
- **[QUICKSTART_MULTIMODAL.md](QUICKSTART_MULTIMODAL.md)** - Quick start guide for multi-modal analysis

## Core Documentation

### User Guides

- **[API_REFERENCE.md](API_REFERENCE.md)** - Complete API documentation for all classes and methods
- **[QUICKSTART_MULTIMODAL.md](QUICKSTART_MULTIMODAL.md)** - Quick reference for multi-modal workflows
- **[FIT_TRANS_GUIDE.md](FIT_TRANS_GUIDE.md)** - Trans fitting and function types
- **[HIGH_MOI_GUIDE.md](HIGH_MOI_GUIDE.md)** - High MOI support with additive guide effects
- **[PLOTTING_GUIDE.md](PLOTTING_GUIDE.md)** - Comprehensive guide to visualization functions
- **[DATA_ACCESS.md](DATA_ACCESS.md)** - Guide to accessing fitted parameters and posterior samples
- **[SAVE_LOAD_GUIDE.md](SAVE_LOAD_GUIDE.md)** - Save/load pipeline stages with modality-specific control

### Technical Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture and design patterns
- **[INITIALIZATION.md](INITIALIZATION.md)** - Technical fitting initialization strategies for all distributions
- **[HILL_FUNCTION_PRIORS.md](HILL_FUNCTION_PRIORS.md)** - Hill function prior computation for trans effects
- **[OUTSTANDING_TASKS.md](OUTSTANDING_TASKS.md)** - Current development priorities and known issues

### HPC and Resource Planning

- **[SLURM_JOB_GENERATOR.md](SLURM_JOB_GENERATOR.md)** - Automated SLURM script generation for HPC clusters (Berzelius)
- **[MEMORY_REQUIREMENTS.md](MEMORY_REQUIREMENTS.md)** - Estimating RAM and VRAM requirements for your dataset
- **[memory_calculator.py](memory_calculator.py)** - Interactive memory estimation calculator

### Specialized Guides

- **[CELL_NAMES_GUIDE.md](CELL_NAMES_GUIDE.md)** - Working with numpy arrays and cell names
- **[SUMMARY_EXPORT_GUIDE.md](SUMMARY_EXPORT_GUIDE.md)** - Exporting results to R-friendly CSV files

## Usage Examples

See the `examples/` directory in the repository root for practical usage examples:
- `examples/multimodal_example.py` - Comprehensive multi-modal workflow

## Archive

Historical documentation from previous development phases is available in the **[archive/](archive/)** directory. These documents are kept for reference but may contain outdated information. Always refer to the current documentation above.

## Additional Resources

- Main repository: https://github.com/leahrosen/bayesDREAM
- Issues and support: https://github.com/leahrosen/bayesDREAM/issues
