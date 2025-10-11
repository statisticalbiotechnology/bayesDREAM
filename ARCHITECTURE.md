# Multi-Modal bayesDREAM Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Interface Layer                        │
├─────────────────────────────────────────────────────────────────┤
│  MultiModalBayesDREAM                                           │
│  - add_transcript_modality()                                    │
│  - add_splicing_modality()                                      │
│  - add_custom_modality()                                        │
│  - get_modality() / list_modalities()                          │
└────────────┬────────────────────────────────────┬───────────────┘
             │                                    │
             │ inherits                           │ contains
             ↓                                    ↓
┌────────────────────────┐          ┌────────────────────────────┐
│  bayesDREAM (base)     │          │  modalities: Dict[str,     │
│                        │          │                 Modality]  │
│  - fit_technical()     │          │                            │
│  - fit_cis()           │          │  Primary: 'gene'           │
│  - fit_trans()         │          │  Others: 'transcript',     │
│  - permute_genes()     │          │          'splicing_*',     │
│  - refit_sumfactor()   │          │          custom, etc.      │
└────────────────────────┘          └────────────────────────────┘
```

## Modality Class Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                         Modality                                 │
├─────────────────────────────────────────────────────────────────┤
│  Attributes:                                                     │
│    - name: str                                                   │
│    - counts: np.ndarray                                         │
│    - feature_meta: pd.DataFrame                                 │
│    - distribution: str (negbinom|multinomial|binomial|...)      │
│    - denominator: np.ndarray (optional, for binomial)           │
│    - dims: Dict (n_features, n_cells, n_categories?, ...)      │
│                                                                  │
│  Methods:                                                        │
│    - to_tensor() → torch.Tensor                                 │
│    - get_feature_subset(indices) → Modality                     │
│    - get_cell_subset(indices) → Modality                        │
└─────────────────────────────────────────────────────────────────┘
                            ↓ created by
    ┌───────────────────────┴───────────────────────────────┐
    │                                                         │
    ↓                                                         ↓
┌─────────────────────┐                    ┌─────────────────────────┐
│  User-created       │                    │  Auto-created via       │
│                     │                    │  helper functions       │
│  Modality(          │                    │                         │
│    name='custom',   │                    │  - add_transcript_      │
│    counts=data,     │                    │    modality()           │
│    feature_meta=... │                    │  - add_splicing_        │
│    distribution=... │                    │    modality()           │
│  )                  │                    │  - create_splicing_     │
└─────────────────────┘                    │    modality()           │
                                            └─────────────────────────┘
```

## Distribution to Data Structure Mapping

```
Distribution Type         Data Shape              Example
─────────────────────────────────────────────────────────────────────
negbinom                  2D: (F, C)             Gene counts
                                                 Transcript counts
                          ┌───┬───┬───┬───┐
                          │ C1│ C2│ C3│...│
                          ├───┼───┼───┼───┤
                        F1│ 5 │ 3 │ 7 │...│
                        F2│ 2 │ 9 │ 1 │...│
                        F3│ 8 │ 0 │ 4 │...│
                          └───┴───┴───┴───┘

multinomial               3D: (F, C, K)          Donor usage
                                                 Isoform usage
                          Cell 1    Cell 2
                          ┌───────┬───────┐
                        F1│[3,2,1]│[4,1,2]│
                        F2│[5,0,3]│[2,3,4]│
                          └───────┴───────┘
                          K=3 categories per feature

binomial                  2D: (F, C) + denom     Exon skipping PSI

                          Inclusion:  Total:
                          ┌───┬───┐  ┌────┬────┐
                        F1│ 5 │ 3 │  │ 10 │  8 │
                        F2│ 2 │ 9 │  │  5 │ 15 │
                          └───┴───┘  └────┴────┘
                          PSI = inclusion / total

normal                    2D: (F, C)             SpliZ scores

                          ┌──────┬──────┬──────┐
                        F1│ 0.5  │-0.2  │ 1.3  │
                        F2│-1.1  │ 0.8  │-0.4  │
                          └──────┴──────┴──────┘

mvnormal                  3D: (F, C, D)          SpliZVD
                                                 D=3 (z0, z1, z2)
                          Cell 1         Cell 2
                          ┌────────────┬────────────┐
                        F1│[0.5,1.2,-3]│[0.8,-1,2.1]│
                        F2│[1.1,0.3,4] │[-2,0.9,1.5]│
                          └────────────┴────────────┘
```

## Splicing Data Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input: Splice Junction Data                   │
│                                                                  │
│  sj_counts (junctions × cells)   sj_meta (junction metadata)    │
│  ┌───────────────┬─────┐        ┌──────────┬──────┬────────┐  │
│  │  coord.intron │ ... │        │coord.int │chrom │ start  │  │
│  ├───────────────┼─────┤        ├──────────┼──────┼────────┤  │
│  │chr1:100:200:+ │  5  │        │chr1:100..│ chr1 │  100   │  │
│  │chr1:100:300:+ │  3  │        │chr1:100..│ chr1 │  100   │  │
│  └───────────────┴─────┘        └──────────┴──────┴────────┘  │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                  splicing.py (Python wrapper)                    │
│                                                                  │
│  run_r_splicing_function():                                     │
│    1. Save counts & metadata to temp CSV files                  │
│    2. Generate R script calling CodeDump.R functions            │
│    3. Execute: Rscript temp_script.R                            │
│    4. Read results from temp CSV                                │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                   CodeDump.R (R functions)                       │
│                                                                  │
│  psi_donor_usage_strand()                                       │
│    → Groups junctions by donor site                             │
│    → Returns: (coord.intron, cell.id, PSI, donor, acceptor)    │
│                                                                  │
│  psi_acceptor_usage_strand()                                    │
│    → Groups junctions by acceptor site                          │
│    → Returns: (coord.intron, cell.id, PSI, donor, acceptor)    │
│                                                                  │
│  psi_exon_skipping_strand()                                     │
│    → Finds cassette exon triplets (inc1, inc2, skip)           │
│    → Returns: (trip_id, cell.id, inc, skip, tot, ...)          │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│              splicing.py (Reshape to Modality)                   │
│                                                                  │
│  process_donor_usage():                                         │
│    → 3D array (donors, cells, acceptors)                        │
│    → metadata: donor coords, list of acceptors                  │
│                                                                  │
│  process_acceptor_usage():                                      │
│    → 3D array (acceptors, cells, donors)                        │
│    → metadata: acceptor coords, list of donors                  │
│                                                                  │
│  process_exon_skipping():                                       │
│    → 2D inclusion + 2D total arrays                             │
│    → metadata: triplet coordinates                              │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                  Output: Modality Objects                        │
│                                                                  │
│  Modality(name='splicing_donor', distribution='multinomial')    │
│  Modality(name='splicing_acceptor', distribution='multinomial') │
│  Modality(name='splicing_exon_skip', distribution='binomial')   │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow in MultiModalBayesDREAM

```
┌─────────────────────────────────────────────────────────────────┐
│                          User Code                               │
│                                                                  │
│  model = MultiModalBayesDREAM(meta, counts, cis_gene='GFI1B')  │
│  model.add_splicing_modality(sj_counts, sj_meta, [...])        │
│  model.add_custom_modality('spliz', scores, meta, 'normal')     │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                  MultiModalBayesDREAM.__init__                   │
│                                                                  │
│  1. Create 'gene' Modality from counts DataFrame                │
│  2. Store in self.modalities['gene']                            │
│  3. Pass gene counts to super().__init__() → bayesDREAM         │
│  4. Filter cells based on meta (ntc + cis_gene)                 │
│  5. Subset all modalities to match filtered cells               │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│              self.modalities (runtime state)                     │
│                                                                  │
│  {                                                               │
│    'gene': Modality(negbinom, 1000 features, 500 cells),       │
│    'splicing_donor': Modality(multinomial, 50 features, ...),  │
│    'splicing_acceptor': Modality(multinomial, 45 features,...),│
│    'splicing_exon_skip': Modality(binomial, 20 features, ...), │
│    'spliz': Modality(normal, 1000 features, 500 cells),        │
│    'custom_mod': Modality(...),                                 │
│  }                                                               │
│                                                                  │
│  self.primary_modality = 'gene'                                 │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Modeling Pipeline                             │
│                                                                  │
│  model.fit_technical(covariates=['cell_line'])                  │
│    → Uses self.counts from primary modality                     │
│    → Fits _model_technical (NegBinom)                           │
│    → Saves self.alpha_y_prefit                                  │
│                                                                  │
│  model.fit_cis(sum_factor_col='sum_factor')                     │
│    → Uses self.counts from primary modality                     │
│    → Fits _model_x (cis effects)                                │
│    → Saves self.x_true, self.posterior_samples_cis              │
│                                                                  │
│  model.fit_trans(function_type='additive_hill')                 │
│    → Uses self.counts from primary modality                     │
│    → Fits _model_y (trans effects as f(x_true))                 │
│    → Saves self.posterior_samples_trans                         │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                   Downstream Analysis                            │
│                                                                  │
│  # Access gene modality (used in modeling)                      │
│  gene_mod = model.get_modality('gene')                          │
│  gene_posteriors = model.posterior_samples_cis                  │
│                                                                  │
│  # Access other modalities (for manual analysis)                │
│  donor_mod = model.get_modality('splicing_donor')               │
│  donor_counts = donor_mod.counts  # 3D array                    │
│  donor_meta = donor_mod.feature_meta                            │
│                                                                  │
│  # Future: fit modality-specific models                         │
│  # model.fit_splicing_trans(modality='splicing_donor', ...)    │
└─────────────────────────────────────────────────────────────────┘
```

## Class Relationships

```
┌─────────────────────────────────────────────────────────────────┐
│                       bayesDREAM (base)                          │
│  Unchanged from original implementation                          │
├─────────────────────────────────────────────────────────────────┤
│  Attributes:                                                     │
│    - meta: pd.DataFrame (cell metadata)                         │
│    - counts: pd.DataFrame (gene counts, features × cells)       │
│    - cis_gene: str                                              │
│    - trans_genes: List[str]                                     │
│    - alpha_y_prefit, alpha_x_prefit: tensors                    │
│    - x_true: posterior cis expression                           │
│    - posterior_samples_cis, posterior_samples_trans: dicts      │
│                                                                  │
│  Methods:                                                        │
│    - fit_technical(covariates, ...)                             │
│    - fit_cis(sum_factor_col, ...)                               │
│    - fit_trans(function_type, ...)                              │
│    - set_alpha_x(), set_alpha_y(), set_x_true()                 │
│    - adjust_ntc_sum_factor(), refit_sumfactor()                 │
│    - permute_genes()                                             │
│    - _model_technical(), _model_x(), _model_y()                 │
└────────────────────────────┬────────────────────────────────────┘
                             │ inherits
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│              MultiModalBayesDREAM (extension)                    │
│  Adds multi-modal support while preserving base functionality   │
├─────────────────────────────────────────────────────────────────┤
│  Additional Attributes:                                          │
│    - modalities: Dict[str, Modality]                            │
│    - primary_modality: str (default: 'gene')                    │
│                                                                  │
│  Additional Methods:                                             │
│    - add_modality(name, modality, overwrite)                    │
│    - add_transcript_modality(counts, meta, use_isoform_usage)   │
│    - add_splicing_modality(sj_counts, sj_meta, types, ...)     │
│    - add_custom_modality(name, counts, feature_meta, dist)      │
│    - get_modality(name) → Modality                              │
│    - list_modalities() → pd.DataFrame                           │
│    - fit_modality_technical(modality_name, ...) [placeholder]   │
│                                                                  │
│  Inherited Methods (work on primary modality):                  │
│    - fit_technical(), fit_cis(), fit_trans() [from bayesDREAM] │
└─────────────────────────────────────────────────────────────────┘
```

## Future Architecture (Modality-Specific Models)

```
Current:
  model.fit_trans() → fits _model_y for primary modality only

Future:
┌─────────────────────────────────────────────────────────────────┐
│  model.fit_trans(modality='gene', ...)                          │
│    → _model_y_negbinom(x_true, y_counts, alpha_y, ...)         │
│                                                                  │
│  model.fit_trans(modality='splicing_donor', ...)                │
│    → _model_y_multinomial(x_true, donor_usage, alpha_d, ...)   │
│                                                                  │
│  model.fit_trans(modality='spliz', ...)                         │
│    → _model_y_normal(x_true, spliz_scores, sigma, ...)         │
│                                                                  │
│  model.fit_cross_modality(['gene', 'splicing_donor'])           │
│    → Joint model: gene expression + donor usage                 │
└─────────────────────────────────────────────────────────────────┘
```

## Module Dependencies

```
model.py (original)
  ├── pandas, numpy
  ├── torch, pyro
  └── scipy, sklearn

modality.py (new)
  ├── pandas, numpy
  └── torch

splicing.py (new)
  ├── pandas, numpy
  ├── subprocess (for R)
  └── modality.py

multimodal.py (new)
  ├── pandas, numpy
  ├── torch
  ├── model.py
  ├── modality.py
  └── splicing.py

__init__.py (updated)
  ├── model.py
  ├── modality.py
  ├── multimodal.py
  └── splicing.py
```
