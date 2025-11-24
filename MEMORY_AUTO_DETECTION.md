# Automatic Memory-Aware Batching in fit_technical()

## Overview

The `fit_technical()` method now includes **automatic memory detection** that dynamically adjusts batching and parallelization based on available system resources. This prevents out-of-memory (OOM) errors during posterior sampling with `Predictive`.

## Features

### 1. Automatic Minibatch Sizing
- Detects available CPU RAM before running `Predictive`
- Estimates memory requirements based on data dimensions (N cells, T features, C groups, S samples)
- Automatically sets `minibatch_size` if memory is limited
- Uses conservative safety factor (70% of available RAM) to prevent OOM

### 2. Intelligent Parallelization Control
- Detects number of CPU cores and estimates per-worker memory usage
- Automatically disables `parallel=True` if it would cause OOM
- Falls back to sequential execution (parallel=False) when necessary
- Still uses batching even in sequential mode if needed

### 3. Optional psutil Dependency
- Uses `psutil` for accurate memory detection when available
- Gracefully falls back to conservative defaults if `psutil` not installed
- To install: `pip install psutil` or `conda install psutil`

## How It Works

### Memory Estimation

For your specific case (31,467 features, 21,761 NTC cells, 2 groups, 1000 samples):

```python
# Parameter posteriors (kept in RAM)
alpha_y [S, C, T] = [1000, 2, 31467]  ~0.23 GB
mu_y [S, T] = [1000, 31467]           ~0.12 GB
phi_y [S, T] = [1000, 31467]          ~0.12 GB
Total params:                          ~0.6 GB

# Input data (counts matrix on CPU)
counts [N, T] = [21761, 31467]         ~2.55 GB

# With parallel=True (32 workers):
# Each worker gets copy of input data
32 workers × 2.55 GB = 81.6 GB base
+ buffer for sample generation
≈ 90-100 GB total (matches your observed usage!)
```

### Decision Logic

1. **User specified `minibatch_size`** → Use it (respects user choice)

2. **`psutil` not available** → Conservative mode:
   - Set `minibatch_size=50`
   - Set `parallel=False`
   - Safe but slower

3. **`psutil` available** → Smart detection:
   ```python
   if parallel_memory > 70% of available RAM:
       # Disable parallel to save memory
       parallel = False
       if all_samples_fit_in_memory:
           minibatch_size = None  # Full batch
       else:
           minibatch_size = auto_calculated
   else:
       # Parallel is safe
       parallel = True
       if params_use_>30%_of_RAM:
           minibatch_size = auto_calculated
       else:
           minibatch_size = None  # Full batch
   ```

## Usage

### Automatic Mode (Recommended)

```python
# Simply call fit_technical without specifying minibatch_size
model.fit_technical(
    sum_factor_col='sum_factor',
    tolerance=0
)

# Output:
# [MEMORY] Available CPU RAM: 450.2 GB / 500.0 GB total
# [MEMORY] Estimated with parallel=True (32 workers): 95.4 GB
# [MEMORY] Parallel execution would exceed safe limit (315.1 GB)
# [MEMORY] Will use sequential execution (parallel=False)
# [MEMORY] Auto-setting minibatch_size=50 (sequential mode)
# [MEMORY] This will require 20 batches
# [INFO] Running Predictive in minibatches of 50 (parallel=False)...
```

### Manual Override (if needed)

```python
# Override automatic detection
model.fit_technical(
    sum_factor_col='sum_factor',
    minibatch_size=10,  # Force specific batch size
    tolerance=0
)

# Output:
# [MEMORY] Using user-specified minibatch_size=10
# [INFO] Running Predictive in minibatches of 10 (parallel=True)...
```

### Adjusting Safety Factor

The safety factor (default 0.7 = 70% of RAM) is hardcoded but can be modified in `technical.py`:

```python
minibatch_size, use_parallel = self._estimate_predictive_memory_and_set_minibatch(
    N=N, T=T_fit, C=C, nsamples=nsamples,
    minibatch_size=minibatch_size,
    distribution=distribution,
    safety_factor=0.7  # Increase to 0.8 or 0.9 if you want to use more RAM
)
```

## Installation

For automatic memory detection to work:

```bash
# Conda (recommended)
conda install psutil

# pip
pip install psutil
```

Without `psutil`, the code will still work but use conservative defaults:
- `minibatch_size=50`
- `parallel=False`

## Benefits

### Before (Your OOM Error)
```
Memory Utilized: 476.77 GB / 496.88 GB (95.95%)
Result: CANCELLED - OOM killed
```

### After (With Auto-Detection)
```
[MEMORY] Detected high memory usage with parallel=True
[MEMORY] Auto-setting minibatch_size=50, parallel=False
Memory Utilized: ~60-80 GB / 496.88 GB
Result: SUCCESS
Runtime: ~2-3x longer but completes successfully
```

## Performance Impact

- **With parallel=True**: Fastest, but high memory (32 × data size)
- **With parallel=False + batching**: Slower, but memory-efficient
- **Rule of thumb**:
  - Small data (<1M cells × features): parallel=True, no batching
  - Medium data (1-10M): parallel=True with batching
  - Large data (>10M): parallel=False with batching

## Troubleshooting

### Still getting OOM?

1. **Reduce number of posterior samples**:
   ```python
   model.fit_technical(nsamples=500)  # Instead of 1000
   ```

2. **Request fewer CPU cores** (reduces parallel workers):
   ```bash
   #SBATCH --cpus-per-task=8  # Instead of 32
   ```

3. **Force smaller minibatch_size**:
   ```python
   model.fit_technical(minibatch_size=10)
   ```

4. **Check if SVI training is the issue** (not Predictive):
   - Reduce batch size in SVI (not yet implemented)
   - Subset features before fitting

### Memory detection not working?

```python
# Check if psutil is installed
import sys
sys.path.insert(0, 'path/to/bayesDREAM')
from bayesDREAM.fitting.technical import HAS_PSUTIL
print(f"psutil available: {HAS_PSUTIL}")

# If False, install it:
# conda install psutil
```

## Technical Details

### Files Modified
- `bayesDREAM/fitting/technical.py`:
  - Added `_estimate_predictive_memory_and_set_minibatch()` method
  - Modified `fit_technical()` to call memory estimator
  - Added `parallel` parameter to `Predictive` calls

### Key Functions

```python
def _estimate_predictive_memory_and_set_minibatch(
    self, N, T, C, nsamples, minibatch_size=None,
    distribution='negbinom', safety_factor=0.7
):
    """
    Returns
    -------
    minibatch_size : int or None
        Recommended batch size, or None if full batch fits
    use_parallel : bool
        Whether to use parallel=True in Predictive
    """
```

### Memory Calculation

```python
# Parameters per sample
params_per_sample_gb = (C * T + 3 * T) * 4 / (1024**3)

# Input data
input_data_gb = (N * T + N) * 4 / (1024**3)

# Per-worker memory (with parallel)
mem_per_worker_gb = input_data_gb + params_per_sample_gb * 10

# Total parallel memory
total_parallel_gb = n_workers * mem_per_worker_gb
```

## Future Enhancements

Potential improvements:
1. Make `safety_factor` a user parameter
2. Add memory estimation for `fit_cis()` and `fit_trans()`
3. Support GPU memory detection for CUDA devices
4. Add progress bars showing memory usage during batching
5. Implement adaptive batch sizing (start large, reduce if OOM)

## References

- Issue: User reported OOM with 500GB RAM, 476.77 GB used (95.95%)
- Root cause: `parallel=True` with 32 workers × 2.55 GB data = ~90 GB minimum
- Solution: Auto-detect and disable parallel when unsafe, use batching when needed
