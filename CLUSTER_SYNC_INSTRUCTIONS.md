# Cluster Sync Instructions

## Problem Summary

Your cluster code is outdated and missing critical fixes for sparse matrix handling. This causes:

1. **IndexError**: Sparse matrices can't use boolean indexing like `sparse[boolean_mask, :]`
2. **AttributeError**: Sparse matrices don't have `.std()` method
3. **Wrong feature counts**: 48 million features excluded instead of expected ~800

## Files That Need to Be Synced

The following files have been fixed locally and need to be copied to the cluster:

### 1. `bayesDREAM/fitting/technical.py`
**Fixes**:
- Lines 489-492: Convert numpy.matrix denominator to numpy.ndarray
- Lines 531-538: Convert numpy.matrix counts to numpy.ndarray
- Lines 732-745: Handle sparse matrices for `.std()` calculation
- Lines 809-837: Use integer indices (not boolean masks) for sparse matrix subsetting

### 2. `bayesDREAM/fitting/cis.py`
**Fix**:
- Lines 299-313: Display actual gene name instead of numeric index

### 3. `bayesDREAM/model.py`
**Fixes**:
- Extract `cell_names` from meta for arrays/sparse matrices (not just DataFrames)
- Modified functions: `_extract_cis_from_gene`, `_extract_cis_generic`, `_create_gene_modality`, `_create_negbinom_modality`

### 4. `environment.yml` (optional but recommended)
**Fix**:
- Now installs PyTorch with CUDA 12.1 support by default (instead of CPU-only)

### 5. `INSTALLATION.md` (optional documentation update)
**Fix**:
- Updated to reflect GPU-enabled default installation

## How to Sync Files to Cluster

### Option 1: rsync from local machine

From your **local machine** (not the cluster), run:

```bash
# Sync the entire bayesDREAM directory
rsync -avz "/Users/lrosen/Library/Mobile Documents/com~apple~CloudDocs/Documents/Postdoc/bayesDREAM code/bayesDREAM_forClaude/bayesDREAM/" \
      <your_username>@<cluster_hostname>:/proj/berzelius-2025-176/users/x_learo/bayesDREAM_forClaude/bayesDREAM/

# Or sync just the modified files
rsync -avz "/Users/lrosen/Library/Mobile Documents/com~apple~CloudDocs/Documents/Postdoc/bayesDREAM code/bayesDREAM_forClaude/bayesDREAM/fitting/technical.py" \
      <your_username>@<cluster_hostname>:/proj/berzelius-2025-176/users/x_learo/bayesDREAM_forClaude/bayesDREAM/fitting/

rsync -avz "/Users/lrosen/Library/Mobile Documents/com~apple~CloudDocs/Documents/Postdoc/bayesDREAM code/bayesDREAM_forClaude/bayesDREAM/fitting/cis.py" \
      <your_username>@<cluster_hostname>:/proj/berzelius-2025-176/users/x_learo/bayesDREAM_forClaude/bayesDREAM/fitting/

rsync -avz "/Users/lrosen/Library/Mobile Documents/com~apple~CloudDocs/Documents/Postdot/bayesDREAM code/bayesDREAM_forClaude/bayesDREAM/model.py" \
      <your_username>@<cluster_hostname>:/proj/berzelius-2025-176/users/x_learo/bayesDREAM_forClaude/bayesDREAM/
```

### Option 2: Manual copy via Jupyter/file browser

1. In Jupyter on the cluster, navigate to `/proj/berzelius-2025-176/users/x_learo/bayesDREAM_forClaude/bayesDREAM/fitting/`
2. Download the local versions of `technical.py`, `cis.py` from your Mac
3. Upload them to the cluster using Jupyter's file browser
4. Repeat for `model.py` in the parent directory

### Option 3: Git (if using version control)

If your repository is under git:

```bash
# On local machine
cd "/Users/lrosen/Library/Mobile Documents/com~apple~CloudDocs/Documents/Postdoc/bayesDREAM code/bayesDREAM_forClaude"
git add bayesDREAM/fitting/technical.py bayesDREAM/fitting/cis.py bayesDREAM/model.py
git commit -m "Fix sparse matrix handling and modality subsetting"
git push

# On cluster
cd /proj/berzelius-2025-176/users/x_learo/bayesDREAM_forClaude
git pull
```

## After Syncing - Verify Fixes Work

After copying the files, restart your Jupyter kernel and run:

```python
# Verify imports work
from bayesDREAM import bayesDREAM
print("✓ Imports successful")

# Check if sparse is handled correctly
from scipy import sparse
import numpy as np
test_sparse = sparse.csr_matrix(np.random.rand(100, 50))
print(f"Sparse matrix test: {sparse.issparse(test_sparse)}")

# Try your model initialization again
# (should now work without errors)
```

## Expected Results After Fix

After syncing the updated code, you should see:

1. **No IndexError**: Boolean indexing converted to integer indices for sparse matrices
2. **Correct feature filtering**: ~800 features excluded (not 48 million)
3. **Cis gene name displayed**: "GFI1B" instead of "43"
4. **Proper cell subsetting**: Modalities subsetted to 31,504 cells
5. **No OOM crashes**: Only NTC cells (21,761) processed during technical fitting

## Fix PyTorch CUDA Support (Separate Issue)

This is a separate problem from the sparse matrix bugs. After syncing code files, also fix PyTorch:

```bash
# On cluster
conda activate bayesdream
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify:
```python
import torch
print('CUDA available:', torch.cuda.is_available())  # Should be True
print('CUDA version:', torch.version.cuda)           # Should be 12.1
```

## Questions?

If you encounter any issues during syncing, check:

1. **Permissions**: Make sure you have write access to `/proj/berzelius-2025-176/users/x_learo/bayesDREAM_forClaude/`
2. **File paths**: Verify the paths match your cluster setup
3. **Kernel restart**: After updating files, restart your Jupyter kernel to load the new code
