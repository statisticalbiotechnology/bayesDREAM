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
    CustomModalityMixin
)

warnings.simplefilter(action="ignore", category=FutureWarning)


class bayesDREAM(
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
        gene_meta: pd.DataFrame = None,
        modalities: Dict[str, Modality] = None,
        cis_gene: str = None,
        cis_feature: str = None,
        primary_modality: str = 'gene',
        guide_covariates: list = ["cell_line"],
        guide_covariates_ntc: list = None,
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
            Count matrix for primary modality. Shape and interpretation depend on primary_modality:
            - For 'gene': genes × cells (negbinom)
            - For 'atac': regions × cells (negbinom)
            - For other modalities: see add_*_modality() methods
        gene_meta : pd.DataFrame, optional
            Gene metadata DataFrame (only used if primary_modality='gene').
            Recommended columns: 'gene', 'gene_name', 'gene_id'.
            If not provided, minimal metadata will be created from counts.index.
        modalities : Dict[str, Modality], optional
            Pre-constructed modalities. Not commonly used.
        cis_gene : str, optional
            For gene modality: gene name to extract as 'cis' modality (e.g., 'GFI1B')
            Alias for cis_feature when primary_modality='gene'
        cis_feature : str, optional
            Feature identifier to extract as 'cis' modality from primary modality.
            - For 'gene': gene name (same as cis_gene)
            - For 'atac': region ID (e.g., 'chr9:132283881-132284881')
            - For other modalities: feature identifier from that modality
        primary_modality : str
            Which modality to use for trans effects (default: 'gene')
            The 'cis' modality will be extracted from this modality during initialization.
        guide_covariates : list
            Covariates for guide grouping
        guide_covariates_ntc : list, optional
            Covariates for NTC guide grouping
        output_dir : str
            Output directory
        label : str
            Run label
        device : str
            'cpu' or 'cuda'
        random_seed : int
            Random seed
        cores : int
            Number of CPU cores

        Notes
        -----
        The 'cis' modality is ONLY extracted during initialization from the primary modality.
        When you call add_*_modality() methods later, NO cis extraction occurs.
        """
        # Initialize modalities dict
        self.modalities = modalities if modalities is not None else {}

        # Resolve cis_feature: cis_gene is an alias for cis_feature when primary_modality='gene'
        if cis_gene is not None and cis_feature is not None:
            raise ValueError("Provide either cis_gene or cis_feature, not both")
        if cis_gene is not None:
            if primary_modality != 'gene':
                warnings.warn(
                    f"cis_gene parameter is intended for primary_modality='gene'. "
                    f"You have primary_modality='{primary_modality}'. Use cis_feature instead.",
                    UserWarning
                )
            cis_feature = cis_gene

        # Store original counts for base class initialization
        counts_for_base = counts

        # Extract 'cis' modality from primary modality (if provided)
        if counts is not None and cis_feature is not None:
            if primary_modality == 'gene':
                self._extract_cis_from_gene(counts, cis_feature)
            elif primary_modality == 'atac':
                raise NotImplementedError(
                    "ATAC as primary modality during initialization is not yet implemented.\n"
                    "Use primary_modality='gene' and then call model.add_atac_modality(..., cis_region='...')"
                )
            else:
                raise NotImplementedError(
                    f"Cis extraction from primary_modality='{primary_modality}' is not yet implemented.\n"
                    f"Currently supported: 'gene'"
                )

        # Create primary modality
        if counts is not None:
            if primary_modality == 'gene':
                self._create_gene_modality(counts, cis_feature)
            elif primary_modality == 'atac':
                raise NotImplementedError(
                    "ATAC as primary modality during initialization is not yet implemented.\n"
                    "Use primary_modality='gene' and then call model.add_atac_modality()"
                )
            else:
                raise NotImplementedError(
                    f"Primary modality '{primary_modality}' is not yet implemented.\n"
                    f"Currently supported: 'gene'"
                )

        # Validate primary modality exists (or will be added later)
        if primary_modality not in self.modalities:
            if counts_for_base is not None:
                # counts provided but primary_modality missing - this is an error
                raise ValueError(f"primary_modality '{primary_modality}' not found in modalities. "
                               f"Available: {list(self.modalities.keys())}")
            else:
                # No counts and no primary modality yet - will be added via add_*_modality()
                print(f"[INFO] Primary modality '{primary_modality}' not yet present. "
                      f"Add it via add_*_modality() before fitting.")
                primary_counts = None  # Will be set when modality is added

        self.primary_modality = primary_modality

        # ========================================================================
        # CRITICAL VALIDATION: Primary modality MUST be negative binomial when cis is present
        # ========================================================================
        if cis_feature is not None:
            # Check if primary modality exists and validate its distribution
            if primary_modality in self.modalities:
                primary_distribution = self.modalities[primary_modality].distribution
                if primary_distribution != 'negbinom':
                    raise ValueError(
                        f"Primary modality '{primary_modality}' has distribution '{primary_distribution}', "
                        f"but cis modeling requires primary modality to be 'negbinom'. "
                        f"The cis feature/gene must represent count data that follows a negative binomial distribution."
                    )
                print(f"[VALIDATION] Primary modality '{primary_modality}' is 'negbinom' - cis modeling is valid")
            elif counts_for_base is not None:
                # Counts were provided, so we're creating the primary modality now
                # For 'gene' primary modality, distribution is always 'negbinom', so we're good
                # For other modalities, they haven't been implemented yet anyway
                if primary_modality == 'gene':
                    # Gene modality is always negbinom, validation passes
                    print(f"[VALIDATION] Creating primary modality '{primary_modality}' as 'negbinom' for cis modeling")
                else:
                    # Other primary modalities are not implemented for cis extraction yet
                    # This should be caught earlier, but add a check just in case
                    warnings.warn(
                        f"Primary modality '{primary_modality}' is not 'gene'. "
                        f"Ensure this modality uses 'negbinom' distribution for cis modeling.",
                        UserWarning
                    )
            else:
                # No counts and no primary modality yet - will be validated when modality is added
                warnings.warn(
                    f"Cannot validate primary modality distribution yet (modality not loaded). "
                    f"Ensure primary modality '{primary_modality}' uses 'negbinom' distribution for cis modeling.",
                    UserWarning
                )

        # Get counts for base class initialization
        # Use original counts (with cis gene) if provided, otherwise get from primary modality
        if counts_for_base is None:
            if primary_modality in self.modalities:
                primary_counts = self.modalities[primary_modality].count_df
                if primary_counts is None:
                    # Convert array to DataFrame
                    mod = self.modalities[primary_modality]
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
                # Use cis_gene as index if provided, otherwise use placeholder
                placeholder_index = [cis_gene] if cis_gene is not None else ['_placeholder_']
                primary_counts = pd.DataFrame(
                    np.ones((1, len(meta))),  # Use 1s instead of 0s to avoid zero-count error
                    index=placeholder_index,
                    columns=meta['cell'].values
                )
        else:
            # Use original counts (includes cis gene for cis modeling)
            primary_counts = counts_for_base

        # Initialize base bayesDREAM with original counts (including cis gene)
        super().__init__(
            meta=meta,
            counts=primary_counts,
            gene_meta=gene_meta,
            cis_gene=cis_gene,
            guide_covariates=guide_covariates,
            guide_covariates_ntc=guide_covariates_ntc,
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

    def _create_gene_modality(self, counts: pd.DataFrame, cis_gene: Optional[str] = None):
        """
        Create 'gene' modality from gene counts (excluding cis gene if specified).

        Parameters
        ----------
        counts : pd.DataFrame
            Gene counts (genes × cells)
        cis_gene : str, optional
            If provided, this gene will be excluded from the gene modality
        """
        if 'gene' in self.modalities:
            warnings.warn("Gene modality already exists. Overwriting.")

        # Exclude cis gene from gene modality (trans genes only)
        if cis_gene is not None and cis_gene in counts.index:
            print(f"[INFO] Creating 'gene' modality with trans genes (excluding '{cis_gene}')")
            counts_trans = counts.drop(index=cis_gene)
        else:
            counts_trans = counts

        # Filter genes with zero standard deviation across ALL cells
        gene_stds = counts_trans.std(axis=1)
        zero_std_mask = gene_stds == 0
        num_zero_std = zero_std_mask.sum()

        if num_zero_std > 0:
            print(f"[INFO] Filtering {num_zero_std} gene(s) with zero standard deviation across all cells")
            counts_trans = counts_trans.loc[~zero_std_mask]

        if len(counts_trans) == 0:
            raise ValueError("No genes left after filtering genes with zero standard deviation!")

        # Create gene modality (trans genes only)
        gene_feature_meta = pd.DataFrame({
            'gene': counts_trans.index.tolist()
        }, index=counts_trans.index)
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
