"""
bayesDREAM: Bayesian Dosage Response Effects Across Modalities

Core implementation for bayesDREAM - a three-step Bayesian framework for modeling
perturbation effects across multiple molecular modalities:

1. fit_technical() - Model technical variation in non-targeting controls
2. fit_cis() - Model direct effects on targeted genes
3. fit_trans() - Model downstream effects as dose-response functions

The bayesDREAM class combines:
- _BayesDREAMCore: Base fitting infrastructure
- Modality mixins: Methods for adding different data types (transcripts, splicing, ATAC, custom)
"""

import warnings
from typing import Dict, Optional
import numpy as np
import pandas as pd

from .core import _BayesDREAMCore
from .modality import Modality
from .modalities import (
    TranscriptModalityMixin,
    SplicingModalityMixin,
    ATACModalityMixin,
    CustomModalityMixin,
    PlottingMixin
)

warnings.simplefilter(action="ignore", category=FutureWarning)


class bayesDREAM(
    PlottingMixin,
    TranscriptModalityMixin,
    SplicingModalityMixin,
    ATACModalityMixin,
    CustomModalityMixin,
    _BayesDREAMCore
):
    """
    bayesDREAM: Bayesian Dosage Response Effects Across Modalities

    A three-step Bayesian framework for modeling perturbation effects across
    multiple molecular modalities including genes, transcripts, splicing, and
    custom measurements.

    Supports:
    - Gene counts (negbinom) - the default/primary modality
    - Transcript counts (negbinom or multinomial for isoform usage)
    - Splice junction counts with donor/acceptor/exon skipping (multinomial/binomial)
    - ATAC-seq peaks (negbinom or binomial)
    - Custom modalities with user-specified distributions

    Attributes
    ----------
    modalities : Dict[str, Modality]
        Dictionary of all modalities. 'gene' is the primary modality.
    primary_modality : str
        Name of primary modality used for cis/trans analysis (default: 'gene')
    """

    def __init__(
        self,
        meta: pd.DataFrame,
        counts: pd.DataFrame = None,
        modality_name: str = 'gene',
        feature_meta: pd.DataFrame = None,
        cis_gene: str = None,
        cis_feature: str = None,
        guide_covariates: list = None,
        guide_covariates_ntc: list = None,
        sum_factor_col: str = 'sum_factor',
        output_dir: str = "./model_out",
        label: str = None,
        device: str = None,
        random_seed: int = 2402,
        cores: int = 1
    ):
        """
        Initialize bayesDREAM.

        Parameters
        ----------
        meta : pd.DataFrame
            Cell-level metadata with columns: cell, guide, target, sum_factor, etc.
        counts : pd.DataFrame, optional
            Count matrix for the primary modality (features × cells). This will be used
            for trans modeling and must represent negbinom count data.

            The modality_name parameter determines how this data is interpreted:
            - 'gene': Gene expression counts (most common)
            - 'atac': ATAC-seq peak counts
            - Custom name: User-defined count modality

            Additional modalities can be added after initialization using add_*_modality() methods.
        modality_name : str, default='gene'
            Name/type of the primary modality provided in counts parameter.

            Pre-set types:
            - 'gene': Gene expression (default, use with feature_meta for gene annotations)
            - 'atac': ATAC-seq peaks

            Custom types:
            - Any string: Creates a custom negbinom modality with that name

            The primary modality MUST be negative binomial (count data) for cis/trans modeling.
        feature_meta : pd.DataFrame, optional
            Feature-level metadata for the primary modality.

            For modality_name='gene':
            - Recommended columns: 'gene', 'gene_name', 'gene_id'
            - If not provided, minimal metadata created from counts.index

            For other modalities:
            - Should contain relevant feature annotations
            - If not provided, minimal metadata created from counts.index
        cis_gene : str, optional
            Feature to extract as 'cis' modality. When modality_name='gene', this is
            a gene name (e.g., 'GFI1B'). For other modality types, use cis_feature instead.

            The cis feature will be:
            1. Extracted as a separate 'cis' modality (1 feature)
            2. Removed from the primary modality (which becomes trans-only)
            3. Used for cis effect modeling in fit_cis()
        cis_feature : str, optional
            Alternative to cis_gene for non-gene modalities. Specifies which feature
            to extract as the 'cis' modality.
        guide_covariates : list, optional
            Covariates for guide grouping (e.g., ['cell_line', 'batch'])
        guide_covariates_ntc : list, optional
            Covariates for NTC guide grouping (if different from guide_covariates)
        sum_factor_col : str, default='sum_factor'
            Column name in meta containing size factors
        output_dir : str, default="./model_out"
            Output directory for saving results
        label : str, optional
            Run label for organizing outputs
        device : str, optional
            'cpu' or 'cuda'. If None, auto-detects.
        random_seed : int, default=2402
            Random seed for reproducibility
        cores : int, default=1
            Number of CPU cores for parallel operations

        Raises
        ------
        ValueError
            - If cis_feature is specified but counts is None
            - If cis_feature not found in counts
            - If cis_feature has zero variance
            - If primary modality would be empty after filtering

        Examples
        --------
        Basic gene expression analysis:
        >>> model = bayesDREAM(
        ...     meta=cell_metadata,
        ...     counts=gene_counts,
        ...     cis_gene='GFI1B'
        ... )

        With gene metadata:
        >>> model = bayesDREAM(
        ...     meta=cell_metadata,
        ...     counts=gene_counts,
        ...     feature_meta=gene_metadata,
        ...     cis_gene='GFI1B',
        ...     guide_covariates=['cell_line']
        ... )

        ATAC-seq analysis:
        >>> model = bayesDREAM(
        ...     meta=cell_metadata,
        ...     counts=atac_counts,
        ...     modality_name='atac',
        ...     feature_meta=peak_metadata,
        ...     cis_feature='chr9:132283881-132284881'
        ... )

        Custom modality:
        >>> model = bayesDREAM(
        ...     meta=cell_metadata,
        ...     counts=my_counts,
        ...     modality_name='my_custom_modality',
        ...     cis_feature='feature_123'
        ... )

        Notes
        -----
        - The 'cis' modality is ONLY extracted during initialization from the primary modality
        - When you call add_*_modality() methods later, NO cis extraction occurs
        - The primary modality MUST be negbinom (count data) for cis/trans modeling
        - Additional modalities (transcripts, splicing, etc.) are added via add_*_modality() methods
        """
        # Initialize modalities dict (always start empty, build from counts)
        self.modalities = {}

        # Resolve cis_feature: cis_gene is an alias for cis_feature when modality_name='gene'
        if cis_gene is not None and cis_feature is not None:
            raise ValueError("Provide either cis_gene or cis_feature, not both")
        if cis_gene is not None:
            if modality_name != 'gene':
                warnings.warn(
                    f"cis_gene parameter is intended for modality_name='gene'. "
                    f"You have modality_name='{modality_name}'. Use cis_feature instead.",
                    UserWarning
                )
            cis_feature = cis_gene

        # Store original counts for base class initialization
        counts_for_base = counts

        # Validate that counts is provided if cis_feature is specified
        if cis_feature is not None and counts is None:
            raise ValueError("Cannot specify cis_feature without providing counts")

        # Extract 'cis' modality from primary modality (if both counts and cis_feature provided)
        if counts is not None and cis_feature is not None:
            if modality_name == 'gene':
                self._extract_cis_from_gene(counts, cis_feature)
            else:
                # Generic cis extraction for any negbinom modality
                self._extract_cis_generic(counts, cis_feature, modality_name)

        # Create primary modality
        if counts is not None:
            if modality_name == 'gene':
                # Use gene-specific creation (with gene_meta handling)
                self._create_gene_modality(counts, cis_feature, gene_meta=feature_meta)
            else:
                # Generic negbinom modality creation
                self._create_negbinom_modality(counts, modality_name, cis_feature, feature_meta)

        # Store primary modality name
        self.primary_modality = modality_name

        # ========================================================================
        # CRITICAL VALIDATION: Primary modality MUST be negative binomial
        # ========================================================================
        if modality_name in self.modalities:
            primary_distribution = self.modalities[modality_name].distribution
            if primary_distribution != 'negbinom':
                raise ValueError(
                    f"Primary modality '{modality_name}' has distribution '{primary_distribution}', "
                    f"but bayesDREAM requires primary modality to be 'negbinom' for cis/trans modeling. "
                    f"The primary modality must represent count data that follows a negative binomial distribution."
                )
            if cis_feature is not None:
                print(f"[VALIDATION] Primary modality '{modality_name}' is 'negbinom' - cis modeling is valid")
        elif counts is None:
            # No counts provided - user will add modalities later
            warnings.warn(
                f"No counts provided during initialization. Primary modality '{modality_name}' must be added "
                f"via add_custom_modality() with distribution='negbinom' before fitting.",
                UserWarning
            )

        # Get counts for base class initialization
        # Use original counts (with cis feature) if provided, otherwise get from primary modality
        if counts_for_base is None:
            if modality_name in self.modalities:
                primary_counts = self.modalities[modality_name].count_df
                if primary_counts is None:
                    # Convert array to DataFrame
                    mod = self.modalities[modality_name]
                    if mod.cells_axis == 1:
                        primary_counts = pd.DataFrame(
                            mod.counts,
                            index=mod.feature_meta.index,
                            columns=mod.cell_names if mod.cell_names else range(mod.dims['n_cells'])
                        )
                    else:
                        primary_counts = pd.DataFrame(
                            mod.counts.T,
                            index=mod.feature_meta.index,
                            columns=mod.cell_names if mod.cell_names else range(mod.dims['n_cells'])
                        )
            else:
                # No counts and no primary modality - create placeholder
                # Will be replaced when modality is added
                print("[INFO] Creating placeholder counts - will be replaced when modality is added")
                # Use meta['cell'] as columns to pass validation
                # Use cis_feature as index if provided, otherwise use placeholder
                placeholder_index = [cis_feature] if cis_feature is not None else ['_placeholder_']
                primary_counts = pd.DataFrame(
                    np.ones((1, len(meta))),  # Use 1s instead of 0s to avoid zero-count error
                    index=placeholder_index,
                    columns=meta['cell'].values
                )
        else:
            # Use original counts (includes cis feature for cis modeling)
            primary_counts = counts_for_base

        # Prepare gene_meta for base class
        # Only pass feature_meta as gene_meta if modality_name is 'gene', otherwise None
        gene_meta_for_base = feature_meta if modality_name == 'gene' else None

        # Initialize base bayesDREAM with original counts (including cis feature)
        super().__init__(
            meta=meta,
            counts=primary_counts,
            gene_meta=gene_meta_for_base,
            cis_gene=cis_feature,  # Pass resolved cis_feature as cis_gene
            guide_covariates=guide_covariates,
            guide_covariates_ntc=guide_covariates_ntc,
            sum_factor_col=sum_factor_col,
            output_dir=output_dir,
            label=label,
            device=device,
            random_seed=random_seed,
            cores=cores
        )

        # Subset all modalities to match filtered cells from base class
        # Base class (super().__init__) has filtered self.meta to valid cells
        valid_cells = self.meta['cell'].tolist()
        print(f"[INFO] Subsetting modalities to {len(valid_cells)} cells from filtered meta")

        for mod_name in list(self.modalities.keys()):
            mod = self.modalities[mod_name]
            if mod.cell_names is not None:
                # Find indices of valid cells in this modality
                cell_indices = [i for i, c in enumerate(mod.cell_names)
                              if c in valid_cells]
                if len(cell_indices) < len(mod.cell_names):
                    print(f"[INFO] Subsetting modality '{mod_name}' from {len(mod.cell_names)} to {len(cell_indices)} cells")
                    self.modalities[mod_name] = mod.get_cell_subset(cell_indices)
                elif len(cell_indices) == len(mod.cell_names):
                    print(f"[INFO] Modality '{mod_name}' already has {len(cell_indices)} cells - no subsetting needed")
                else:
                    # This shouldn't happen - means we have fewer cells in modality than in meta
                    print(f"[WARNING] Modality '{mod_name}' has {len(mod.cell_names)} cells but filtered meta has {len(valid_cells)}")

        print(f"[INIT] bayesDREAM: {len(self.modalities)} modalities loaded")
        for name, mod in self.modalities.items():
            print(f"  - {name}: {mod}")

    def _extract_cis_from_gene(self, counts: pd.DataFrame, cis_gene: str):
        """
        Extract 'cis' modality from gene counts.

        Parameters
        ----------
        counts : pd.DataFrame
            Gene counts (genes × cells)
        cis_gene : str
            Gene name to extract as cis modality
        """
        if cis_gene not in counts.index:
            raise ValueError(
                f"cis_gene '{cis_gene}' not found in counts.\n"
                f"Available genes: {counts.index[:10].tolist()}..."
            )

        # Check if cis gene has zero variance (critical for cis modeling)
        cis_gene_std = counts.loc[cis_gene].std()
        if cis_gene_std == 0:
            raise ValueError(
                f"Cis gene '{cis_gene}' has zero standard deviation across all cells. "
                f"Cannot model cis effects for a gene with constant expression."
            )

        print(f"[INFO] Extracting 'cis' modality from gene '{cis_gene}'")
        cis_feature_meta = pd.DataFrame({
            'gene': [cis_gene]
        }, index=[cis_gene])
        self.modalities['cis'] = Modality(
            name='cis',
            counts=counts.loc[[cis_gene]],
            feature_meta=cis_feature_meta,
            distribution='negbinom',
            cells_axis=1
        )

    def _extract_cis_generic(self, counts: pd.DataFrame, cis_feature: str, modality_name: str):
        """
        Extract 'cis' modality from a generic negbinom modality.

        Parameters
        ----------
        counts : pd.DataFrame
            Feature counts (features × cells)
        cis_feature : str
            Feature name to extract as cis modality
        modality_name : str
            Name of the modality type (for error messages)
        """
        if cis_feature not in counts.index:
            raise ValueError(
                f"cis_feature '{cis_feature}' not found in counts.\n"
                f"Available features: {counts.index[:10].tolist()}..."
            )

        # Check if cis feature has zero variance (critical for cis modeling)
        cis_feature_std = counts.loc[cis_feature].std()
        if cis_feature_std == 0:
            raise ValueError(
                f"Cis feature '{cis_feature}' has zero standard deviation across all cells. "
                f"Cannot model cis effects for a feature with constant values."
            )

        print(f"[INFO] Extracting 'cis' modality from feature '{cis_feature}' ({modality_name})")
        cis_feature_meta = pd.DataFrame({
            'feature': [cis_feature],
            'modality_type': [modality_name]
        }, index=[cis_feature])
        self.modalities['cis'] = Modality(
            name='cis',
            counts=counts.loc[[cis_feature]],
            feature_meta=cis_feature_meta,
            distribution='negbinom',
            cells_axis=1
        )

    def _create_negbinom_modality(
        self,
        counts: pd.DataFrame,
        modality_name: str,
        cis_feature: Optional[str] = None,
        feature_meta: Optional[pd.DataFrame] = None
    ):
        """
        Create a generic negbinom modality (excluding cis feature if specified).

        Parameters
        ----------
        counts : pd.DataFrame
            Feature counts (features × cells)
        modality_name : str
            Name for this modality
        cis_feature : str, optional
            If provided, this feature will be excluded from the modality
        feature_meta : pd.DataFrame, optional
            Feature metadata. If None, creates minimal metadata from counts.index
        """
        if modality_name in self.modalities:
            warnings.warn(f"Modality '{modality_name}' already exists. Overwriting.")

        # Exclude cis feature from modality (trans features only)
        if cis_feature is not None and cis_feature in counts.index:
            print(f"[INFO] Creating '{modality_name}' modality with trans features (excluding '{cis_feature}')")
            counts_trans = counts.drop(index=cis_feature)
        else:
            counts_trans = counts

        # Check for and remove duplicate indices in counts_trans
        if counts_trans.index.duplicated().any():
            n_dups = counts_trans.index.duplicated().sum()
            dup_features = counts_trans.index[counts_trans.index.duplicated()].unique().tolist()
            warnings.warn(
                f"[WARNING] counts has {n_dups} duplicate feature(s). "
                f"Keeping first occurrence for each duplicate. "
                f"Duplicates: {dup_features[:10]}",
                UserWarning
            )
            # Remove duplicates, keeping first occurrence
            counts_trans = counts_trans[~counts_trans.index.duplicated(keep='first')]

        # Filter features with zero standard deviation across ALL cells
        feature_stds = counts_trans.std(axis=1)
        zero_std_mask = feature_stds == 0
        num_zero_std = zero_std_mask.sum()

        if num_zero_std > 0:
            print(f"[INFO] Filtering {num_zero_std} feature(s) with zero standard deviation across all cells")
            counts_trans = counts_trans.loc[~zero_std_mask]

        if len(counts_trans) == 0:
            raise ValueError(f"No features left in '{modality_name}' after filtering features with zero standard deviation!")

        # Create or use provided feature metadata
        if feature_meta is None:
            mod_feature_meta = pd.DataFrame({
                'feature': counts_trans.index.tolist()
            }, index=counts_trans.index)
        else:
            # Check for duplicate indices in feature_meta
            if feature_meta.index.duplicated().any():
                n_dups = feature_meta.index.duplicated().sum()
                dup_features = feature_meta.index[feature_meta.index.duplicated()].unique().tolist()
                warnings.warn(
                    f"[WARNING] feature_meta has {n_dups} duplicate feature(s). "
                    f"Keeping first occurrence for each duplicate. "
                    f"Duplicates: {dup_features[:10]}",
                    UserWarning
                )
                # Remove duplicates, keeping first occurrence
                feature_meta = feature_meta[~feature_meta.index.duplicated(keep='first')]

            # Use provided metadata, but subset to remaining features
            # Use reindex instead of loc to guarantee exact match (no duplicates after deduplication)
            mod_feature_meta = feature_meta.reindex(counts_trans.index).copy()

            # Check for any missing features in feature_meta
            if mod_feature_meta.isnull().any().any():
                missing_features = counts_trans.index[mod_feature_meta.isnull().any(axis=1)]
                warnings.warn(
                    f"[WARNING] {len(missing_features)} feature(s) in counts not found in feature_meta. "
                    f"Creating minimal metadata for these features. First few: {missing_features[:5].tolist()}",
                    UserWarning
                )
                # Fill missing features with minimal metadata
                for feature in missing_features:
                    mod_feature_meta.loc[feature, 'feature'] = feature

        self.modalities[modality_name] = Modality(
            name=modality_name,
            counts=counts_trans,
            feature_meta=mod_feature_meta,
            distribution='negbinom',
            cells_axis=1
        )

    def _create_gene_modality(
        self,
        counts: pd.DataFrame,
        cis_gene: Optional[str] = None,
        gene_meta: Optional[pd.DataFrame] = None
    ):
        """
        Create 'gene' modality from gene counts (excluding cis gene if specified).

        Parameters
        ----------
        counts : pd.DataFrame
            Gene counts (genes × cells)
        cis_gene : str, optional
            If provided, this gene will be excluded from the gene modality
        gene_meta : pd.DataFrame, optional
            Gene metadata. If None, creates minimal metadata from counts.index
        """
        if 'gene' in self.modalities:
            warnings.warn("Gene modality already exists. Overwriting.")

        # Exclude cis gene from gene modality (trans genes only)
        if cis_gene is not None and cis_gene in counts.index:
            print(f"[INFO] Creating 'gene' modality with trans genes (excluding '{cis_gene}')")
            counts_trans = counts.drop(index=cis_gene)
        else:
            counts_trans = counts

        # Check for and remove duplicate indices in counts_trans
        if counts_trans.index.duplicated().any():
            n_dups = counts_trans.index.duplicated().sum()
            dup_genes = counts_trans.index[counts_trans.index.duplicated()].unique().tolist()
            warnings.warn(
                f"[WARNING] counts has {n_dups} duplicate gene(s). "
                f"Keeping first occurrence for each duplicate. "
                f"Duplicates: {dup_genes[:10]}",
                UserWarning
            )
            # Remove duplicates, keeping first occurrence
            counts_trans = counts_trans[~counts_trans.index.duplicated(keep='first')]

        # Filter genes with zero standard deviation across ALL cells
        gene_stds = counts_trans.std(axis=1)
        zero_std_mask = gene_stds == 0
        num_zero_std = zero_std_mask.sum()

        if num_zero_std > 0:
            print(f"[INFO] Filtering {num_zero_std} gene(s) with zero standard deviation across all cells")
            counts_trans = counts_trans.loc[~zero_std_mask]

        if len(counts_trans) == 0:
            raise ValueError("No genes left after filtering genes with zero standard deviation!")

        # Create or use provided gene metadata
        if gene_meta is None:
            gene_feature_meta = pd.DataFrame({
                'gene': counts_trans.index.tolist()
            }, index=counts_trans.index)
        else:
            # Check for duplicate indices in gene_meta
            if gene_meta.index.duplicated().any():
                n_dups = gene_meta.index.duplicated().sum()
                dup_genes = gene_meta.index[gene_meta.index.duplicated()].unique().tolist()
                warnings.warn(
                    f"[WARNING] gene_meta has {n_dups} duplicate gene(s). "
                    f"Keeping first occurrence for each duplicate. "
                    f"Duplicates: {dup_genes[:10]}",
                    UserWarning
                )
                # Remove duplicates, keeping first occurrence
                gene_meta = gene_meta[~gene_meta.index.duplicated(keep='first')]

            # Use provided gene_meta, but subset to remaining genes
            # Use reindex instead of loc to guarantee exact match (no duplicates after deduplication)
            gene_feature_meta = gene_meta.reindex(counts_trans.index).copy()

            # Check for any missing genes in gene_meta
            if gene_feature_meta.isnull().any().any():
                missing_genes = counts_trans.index[gene_feature_meta.isnull().any(axis=1)]
                warnings.warn(
                    f"[WARNING] {len(missing_genes)} gene(s) in counts not found in gene_meta. "
                    f"Creating minimal metadata for these genes. First few: {missing_genes[:5].tolist()}",
                    UserWarning
                )
                # Fill missing genes with minimal metadata
                for gene in missing_genes:
                    gene_feature_meta.loc[gene, 'gene'] = gene

        self.modalities['gene'] = Modality(
            name='gene',
            counts=counts_trans,
            feature_meta=gene_feature_meta,
            distribution='negbinom',
            cells_axis=1
        )

    def add_modality(
        self,
        name: str,
        modality: Modality,
        overwrite: bool = False
    ):
        """
        Add a new modality.

        Parameters
        ----------
        name : str
            Modality name
        modality : Modality
            Modality object
        overwrite : bool
            Whether to overwrite existing modality with same name
        """
        if name in self.modalities and not overwrite:
            raise ValueError(f"Modality '{name}' already exists. Use overwrite=True to replace it.")

        self.modalities[name] = modality
        print(f"[INFO] Added modality '{name}': {modality}")

    def get_modality(self, name: str) -> Modality:
        """Get a modality by name."""
        if name not in self.modalities:
            raise KeyError(f"Modality '{name}' not found. Available: {list(self.modalities.keys())}")
        return self.modalities[name]

    def list_modalities(self) -> pd.DataFrame:
        """
        List all modalities.

        Returns
        -------
        pd.DataFrame
            Summary of modalities with columns: name, distribution, n_features, n_cells
        """
        if not self.modalities:
            return pd.DataFrame(columns=['name', 'distribution', 'n_features', 'n_cells'])

        rows = []
        for name, mod in self.modalities.items():
            rows.append({
                'name': name,
                'distribution': mod.distribution,
                'n_features': mod.dims['n_features'],
                'n_cells': mod.dims['n_cells']
            })
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        """String representation."""
        n_cells = len(self.meta)
        n_modalities = len(self.modalities)
        modality_names = ', '.join(self.modalities.keys())

        return (
            f"bayesDREAM(\n"
            f"  cis_gene='{self.cis_gene}',\n"
            f"  n_cells={n_cells},\n"
            f"  n_modalities={n_modalities},\n"
            f"  modalities=[{modality_names}],\n"
            f"  primary_modality='{self.primary_modality}',\n"
            f"  device={self.device}\n"
            f")"
        )
