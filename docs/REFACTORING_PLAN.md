# bayesDREAM Refactoring Plan

## Current State

**File:** `bayesDREAM/model.py`
- **Lines:** 4,537
- **Classes:** 2 (_BayesDREAMCore, bayesDREAM)
- **Main sections:**
  1. Helper functions (lines ~1-57)
  2. _BayesDREAMCore class (lines 58-3292)
     - Initialization & setup
     - Parameter setters (alpha_x, alpha_y, x_true, etc.)
     - Technical model & fitting
     - Cis model & fitting
     - Trans model & fitting
     - Utility methods
  3. bayesDREAM class (lines 3293-4537)
     - Multi-modal initialization
     - Modality management
     - Save/load methods

## Proposed Refactoring

### New File Structure

```
bayesDREAM/
├── __init__.py                 # Package exports
├── model.py                    # Main bayesDREAM class (lightweight)
├── core.py                     # _BayesDREAMCore base class
├── modality.py                 # ✅ Already exists
├── distributions.py            # ✅ Already exists
├── splicing.py                 # ✅ Already exists
├── utils.py                    # ✅ Already exists
│
├── fitting/                    # NEW: Fitting methods module
│   ├── __init__.py
│   ├── technical.py            # Technical model & fit_technical
│   ├── cis.py                  # Cis model & fit_cis
│   ├── trans.py                # Trans model & fit_trans
│   └── helpers.py              # Shared fitting utilities
│
├── io/                         # NEW: Input/Output module
│   ├── __init__.py
│   ├── save.py                 # Save methods
│   └── load.py                 # Load methods
│
└── modalities/                 # NEW: Modality-specific code
    ├── __init__.py
    ├── transcript.py           # add_transcript_modality
    ├── splicing.py             # add_splicing_modality
    ├── atac.py                 # add_atac_modality
    └── custom.py               # add_custom_modality
```

---

## Detailed Breakdown

### 1. `fitting/technical.py`

**Content:**
- `_model_technical()` (lines 733-915)
- `fit_technical()` (lines 950-1653)
- Helper: `set_technical_groups()` (lines 917-948)

**Approximate lines:** ~900

**Dependencies:**
- PyTorch, Pyro
- distributions.py (for observation samplers)
- Access to model attributes (y_obs, meta, etc.)

**Design:**
```python
# fitting/technical.py
class TechnicalFitter:
    """Handles technical variation fitting."""

    def __init__(self, model):
        self.model = model

    def _model_technical(self, ...):
        """Pyro model for technical variation."""
        ...

    def fit(self, modality_name, sum_factor_col, ...):
        """Fit technical variation model."""
        ...

    def set_technical_groups(self, covariates):
        """Set technical grouping."""
        ...
```

---

### 2. `fitting/cis.py`

**Content:**
- `_model_x()` (lines 1655-1785)
- `fit_cis()` (lines 1787-2279)
- Helper: `cis_init_loc_fn()` (lines 291-353)

**Approximate lines:** ~700

**Dependencies:**
- PyTorch, Pyro
- distributions.py
- Access to model attributes (counts, meta, alpha_x_prefit, etc.)

**Design:**
```python
# fitting/cis.py
class CisFitter:
    """Handles cis gene expression fitting."""

    def __init__(self, model):
        self.model = model

    def _model_x(self, ...):
        """Pyro model for cis effects."""
        ...

    def fit(self, sum_factor_col, cis_feature, ...):
        """Fit cis gene expression."""
        ...

    def cis_init_loc_fn(self, ...):
        """Initialization function for cis model."""
        ...
```

---

### 3. `fitting/trans.py`

**Content:**
- `_model_y()` (lines 2399-2682)
- `fit_trans()` (lines 2700-3285)
- Helper: `refit_sumfactor()` (lines 2281-2397)

**Approximate lines:** ~1,000

**Dependencies:**
- PyTorch, Pyro
- distributions.py
- Access to model attributes (x_true, alpha_y_prefit, etc.)

**Design:**
```python
# fitting/trans.py
class TransFitter:
    """Handles trans effects fitting."""

    def __init__(self, model):
        self.model = model

    def _model_y(self, ...):
        """Pyro model for trans effects."""
        ...

    def fit(self, sum_factor_col, function_type, modality_name, ...):
        """Fit trans effects."""
        ...

    def refit_sumfactor(self, covariates, ...):
        """Refit sum factors based on posterior cis expression."""
        ...
```

---

### 4. `fitting/helpers.py`

**Content:**
- Helper functions currently at top of model.py
- Shared utilities used by multiple fitters

**Functions to move:**
```python
# Lines 1-57 from current model.py
def Hill_function_v1(...)
def Hill_function_v2(...)
def additive_Hill_function(...)
def Polynomial_function(...)
def compute_sparsity_probability(...)
def compute_temp_schedule(...)
def make_linear_increasing_mask(...)
```

**Approximate lines:** ~60

---

### 5. `io/save.py`

**Content:**
- `save_technical_fit()` (lines 4140-4224)
- `save_cis_fit()` (lines 4315-4361)
- `save_trans_fit()` (lines 4407-4476)

**Approximate lines:** ~200

**Design:**
```python
# io/save.py
class ModelSaver:
    """Handles saving fitted parameters."""

    def __init__(self, model):
        self.model = model

    def save_technical_fit(self, output_dir, modalities, save_model_level):
        """Save technical fit results."""
        ...

    def save_cis_fit(self, output_dir):
        """Save cis fit results."""
        ...

    def save_trans_fit(self, output_dir, modalities, save_model_level):
        """Save trans fit results."""
        ...
```

---

### 6. `io/load.py`

**Content:**
- `load_technical_fit()` (lines 4226-4313)
- `load_cis_fit()` (lines 4363-4405)
- `load_trans_fit()` (lines 4478-4531)

**Approximate lines:** ~200

**Design:**
```python
# io/load.py
class ModelLoader:
    """Handles loading fitted parameters."""

    def __init__(self, model):
        self.model = model

    def load_technical_fit(self, input_dir, use_posterior, modalities, load_model_level):
        """Load technical fit results."""
        ...

    def load_cis_fit(self, input_dir, use_posterior):
        """Load cis fit results."""
        ...

    def load_trans_fit(self, input_dir, modalities, load_model_level):
        """Load trans fit results."""
        ...
```

---

### 7. `modalities/transcript.py`

**Content:**
- `add_transcript_modality()` (lines 3552-3733)

**Approximate lines:** ~180

---

### 8. `modalities/splicing.py`

**Content:**
- `add_splicing_modality()` (lines 3735-3802)

**Approximate lines:** ~70

**Note:** Don't confuse with existing `splicing.py` (processing functions)

---

### 9. `modalities/atac.py`

**Content:**
- `add_atac_modality()` (lines 3966-4108)

**Approximate lines:** ~140

---

### 10. `modalities/custom.py`

**Content:**
- `add_custom_modality()` (lines 3804-3964)

**Approximate lines:** ~160

---

### 11. `core.py` (Refactored _BayesDREAMCore)

**Content:**
- `__init__()` (lines 67-289)
- Parameter setters:
  - `set_alpha_x()` (lines 355-388)
  - `set_alpha_y()` (lines 390-428)
  - `set_o_x_grouped()` (lines 430-466)
  - `set_o_x()` (lines 468-496)
  - `set_x_true()` (lines 498-531)
- Utility methods:
  - `adjust_ntc_sum_factor()` (lines 533-648)
  - `permute_genes()` (lines 650-731)

**Methods that delegate to fitters:**
```python
class _BayesDREAMCore:
    def __init__(self, ...):
        # Initialization
        self._technical_fitter = TechnicalFitter(self)
        self._cis_fitter = CisFitter(self)
        self._trans_fitter = TransFitter(self)
        self._saver = ModelSaver(self)
        self._loader = ModelLoader(self)

    def fit_technical(self, *args, **kwargs):
        return self._technical_fitter.fit(*args, **kwargs)

    def fit_cis(self, *args, **kwargs):
        return self._cis_fitter.fit(*args, **kwargs)

    def fit_trans(self, *args, **kwargs):
        return self._trans_fitter.fit(*args, **kwargs)

    def save_technical_fit(self, *args, **kwargs):
        return self._saver.save_technical_fit(*args, **kwargs)

    # ... etc
```

**Approximate lines:** ~600

---

### 12. `model.py` (Refactored bayesDREAM)

**Content:**
- `__init__()` (lines 3315-3495)
- `add_modality()` (lines 3497-3550)
- Delegation to modality adders
- `get_modality()` (lines 4110-4114)
- `list_modalities()` (lines 4116-4134)
- `__repr__()` (lines 4533-4537)

**Design:**
```python
from .core import _BayesDREAMCore
from .modalities.transcript import TranscriptModalityMixin
from .modalities.splicing import SplicingModalityMixin
from .modalities.atac import ATACModalityMixin
from .modalities.custom import CustomModalityMixin

class bayesDREAM(_BayesDREAMCore,
                 TranscriptModalityMixin,
                 SplicingModalityMixin,
                 ATACModalityMixin,
                 CustomModalityMixin):
    """Main bayesDREAM class with multi-modal support."""

    def __init__(self, ...):
        super().__init__(...)
        # Multi-modal specific initialization
```

**Approximate lines:** ~300

---

## Migration Strategy

### Phase 1: Create New Modules (Non-Breaking)

1. Create directory structure
2. Create new files with fitter classes
3. Keep original model.py intact
4. Add tests for new modules

**Files to create:**
- `fitting/__init__.py`
- `fitting/helpers.py`
- `fitting/technical.py`
- `fitting/cis.py`
- `fitting/trans.py`
- `io/__init__.py`
- `io/save.py`
- `io/load.py`
- `modalities/__init__.py`
- `modalities/transcript.py`
- `modalities/splicing_modality.py` (rename to avoid conflict)
- `modalities/atac.py`
- `modalities/custom.py`

### Phase 2: Update Core Classes (Breaking)

1. Create `core.py` with refactored _BayesDREAMCore
2. Update `model.py` to use new structure
3. Update imports in `__init__.py`
4. Run full test suite

### Phase 3: Deprecate (Optional)

1. Keep old model.py as model_legacy.py for one release
2. Add deprecation warnings
3. Update all documentation

---

## Benefits

### 1. **Maintainability**
- Each file < 1,000 lines
- Clear separation of concerns
- Easy to find specific functionality

### 2. **Testability**
- Can test fitters independently
- Mock dependencies easily
- Faster test execution

### 3. **Collaboration**
- Multiple developers can work on different modules
- Reduced merge conflicts
- Clear code ownership

### 4. **Performance**
- Faster import times (lazy loading possible)
- Easier to profile specific components

### 5. **Documentation**
- Each module can have focused documentation
- Clearer API boundaries

---

## File Size Comparison

| File | Current | After Refactoring |
|------|---------|-------------------|
| `model.py` | 4,537 | ~300 |
| `core.py` | - | ~600 |
| `fitting/technical.py` | - | ~900 |
| `fitting/cis.py` | - | ~700 |
| `fitting/trans.py` | - | ~1,000 |
| `fitting/helpers.py` | - | ~60 |
| `io/save.py` | - | ~200 |
| `io/load.py` | - | ~200 |
| `modalities/transcript.py` | - | ~180 |
| `modalities/splicing_modality.py` | - | ~70 |
| `modalities/atac.py` | - | ~140 |
| `modalities/custom.py` | - | ~160 |
| **Total** | **4,537** | **4,510** (similar, but organized) |

---

## Implementation Checklist

### Preparation
- [ ] Create feature branch: `refactor/modularize-model`
- [ ] Backup current working tests
- [ ] Document current test coverage

### Phase 1: Create Modules
- [ ] Create `fitting/` directory and `__init__.py`
- [ ] Create `fitting/helpers.py` with helper functions
- [ ] Create `fitting/technical.py` with TechnicalFitter
- [ ] Create `fitting/cis.py` with CisFitter
- [ ] Create `fitting/trans.py` with TransFitter
- [ ] Create `io/` directory and `__init__.py`
- [ ] Create `io/save.py` with ModelSaver
- [ ] Create `io/load.py` with ModelLoader
- [ ] Create `modalities/` directory and `__init__.py`
- [ ] Create `modalities/transcript.py`
- [ ] Create `modalities/splicing_modality.py`
- [ ] Create `modalities/atac.py`
- [ ] Create `modalities/custom.py`
- [ ] Write unit tests for each new module

### Phase 2: Refactor Core
- [ ] Create `core.py` with _BayesDREAMCore
- [ ] Update `model.py` to use new structure
- [ ] Update `__init__.py` imports
- [ ] Run full test suite
- [ ] Fix any import issues
- [ ] Update CLAUDE.md documentation

### Phase 3: Validation
- [ ] Run all existing tests
- [ ] Run integration tests
- [ ] Benchmark performance (ensure no regression)
- [ ] Update user documentation
- [ ] Create migration guide

### Phase 4: Cleanup
- [ ] Remove commented code
- [ ] Update type hints
- [ ] Run linter
- [ ] Create PR for review

---

## Risks & Mitigation

### Risk 1: Breaking Changes
**Mitigation:**
- Keep public API identical
- All methods still accessible from bayesDREAM class
- Extensive testing before merge

### Risk 2: Import Cycles
**Mitigation:**
- Careful dependency management
- Use dependency injection (pass model to fitters)
- Keep imports at module level

### Risk 3: Performance Regression
**Mitigation:**
- Profile before and after
- Benchmark critical paths
- Lazy loading if needed

### Risk 4: Test Coverage Gaps
**Mitigation:**
- Run coverage report before refactoring
- Ensure same coverage after
- Add tests for new module boundaries

---

## Timeline Estimate

- **Phase 1 (Create Modules):** 2-3 days
- **Phase 2 (Refactor Core):** 1-2 days
- **Phase 3 (Validation):** 1-2 days
- **Phase 4 (Cleanup):** 1 day

**Total:** ~5-8 days

---

## Next Steps

1. **Get approval** for refactoring plan
2. **Create feature branch**
3. **Start with Phase 1** (non-breaking changes)
4. **Iterative testing** after each module
5. **Full integration** in Phase 2

Would you like me to proceed with this refactoring?
