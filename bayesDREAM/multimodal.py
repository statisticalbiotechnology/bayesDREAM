"""
Multi-modal extension for bayesDREAM.

This module provides a MultiModalBayesDREAM class that extends the base bayesDREAM
class to support multiple molecular modalities (genes, transcripts, splicing, etc.).
"""

import warnings
from typing import Dict, Optional, List, Union
import numpy as np
import pandas as pd
import torch

from .model import bayesDREAM
from .modality import Modality
from .splicing import create_splicing_modality


class MultiModalBayesDREAM(bayesDREAM):
    """
    Multi-modal extension of bayesDREAM supporting genes, transcripts, splicing, and custom modalities.

    This class wraps the standard bayesDREAM model and adds support for:
    - Gene counts (negbinom) - the default/primary modality
    - Transcript counts (negbinom or multinomial for isoform usage)
    - Splice junction counts with donor/acceptor/exon skipping (multinomial/binomial)
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
        modalities: Dict[str, Modality] = None,
        cis_gene: str = None,
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
        Initialize multi-modal bayesDREAM.

        Parameters
        ----------
        meta : pd.DataFrame
            Cell-level metadata with columns: cell, guide, target, sum_factor, etc.
        counts : pd.DataFrame, optional
            Gene counts (genes × cells). If provided, becomes 'gene' modality.
            Either counts or modalities (with 'gene' key) must be provided.
        modalities : Dict[str, Modality], optional
            Pre-constructed modalities. If 'gene' not present and counts is provided,
            a gene modality will be created from counts.
        cis_gene : str
            Gene targeted in cis
        primary_modality : str
            Which modality to use for cis effects (default: 'gene')
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
        """
        # Initialize modalities dict
        self.modalities = modalities if modalities is not None else {}

        # Store original counts for base class initialization
        counts_for_base = counts

        # Handle gene counts
        if counts is not None:
            if 'gene' in self.modalities:
                warnings.warn("Both counts and modalities['gene'] provided. Using counts.")

            # Exclude cis gene from gene count modality features
            # (The base class still gets the full counts with cis gene for cis modeling)
            if cis_gene is not None and cis_gene in counts.index:
                print(f"[INFO] Excluding cis gene '{cis_gene}' from gene count modality features")
                counts_trans = counts.drop(index=cis_gene)
            else:
                counts_trans = counts

            # Create gene modality (trans genes only, without cis gene)
            gene_feature_meta = pd.DataFrame({
                'gene': counts_trans.index.tolist()
            })
            self.modalities['gene'] = Modality(
                name='gene',
                counts=counts_trans,
                feature_meta=gene_feature_meta,
                distribution='negbinom',
                cells_axis=1
            )

        # Validate primary modality exists
        if primary_modality not in self.modalities:
            raise ValueError(f"primary_modality '{primary_modality}' not found in modalities. "
                           f"Available: {list(self.modalities.keys())}")

        self.primary_modality = primary_modality

        # Get counts for base class initialization
        # Use original counts (with cis gene) if provided, otherwise get from primary modality
        if counts_for_base is None:
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
            # Use original counts (includes cis gene for cis modeling)
            primary_counts = counts_for_base

        # Initialize base bayesDREAM with original counts (including cis gene)
        super().__init__(
            meta=meta,
            counts=primary_counts,
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
        valid_cells = self.meta['cell'].tolist()
        for mod_name in list(self.modalities.keys()):
            if self.modalities[mod_name].cell_names is not None:
                # Find indices of valid cells
                cell_indices = [i for i, c in enumerate(self.modalities[mod_name].cell_names)
                              if c in valid_cells]
                self.modalities[mod_name] = self.modalities[mod_name].get_cell_subset(cell_indices)

        print(f"[INIT] MultiModalBayesDREAM: {len(self.modalities)} modalities loaded")
        for name, mod in self.modalities.items():
            print(f"  - {name}: {mod}")

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
            raise ValueError(f"Modality '{name}' already exists. Set overwrite=True to replace.")

        # Validate cell alignment with primary modality
        valid_meta_cells = set(self.meta['cell'].tolist())

        if modality.cell_names is not None:
            modality_cells = set(modality.cell_names)

            # Check for cells in modality but not in meta (will be discarded)
            extra_cells = modality_cells - valid_meta_cells
            if extra_cells:
                n_extra = len(extra_cells)
                warnings.warn(
                    f"[{name}] {n_extra} cell(s) in modality are not in the primary gene counts and will be discarded. "
                    f"Cells without cis gene expression cannot be used in modeling.",
                    UserWarning
                )

            # Check for cells in meta but not in modality (informational)
            missing_cells = valid_meta_cells - modality_cells
            if missing_cells:
                n_missing = len(missing_cells)
                print(f"[INFO] [{name}] {n_missing} cell(s) from primary modality are not in this modality (this is OK).")

            # Subset to common cells
            common_cells = [c for c in modality.cell_names if c in valid_meta_cells]
            if len(common_cells) == 0:
                raise ValueError(f"No overlapping cells between modality '{name}' and primary modality!")

            if len(common_cells) < len(modality.cell_names):
                modality = modality.get_cell_subset(common_cells)
                print(f"[INFO] [{name}] Subsetted to {len(common_cells)} common cells.")

        self.modalities[name] = modality
        print(f"[INFO] Added modality: {modality}")

    def add_transcript_modality(
        self,
        transcript_counts: pd.DataFrame,
        transcript_meta: pd.DataFrame,
        use_isoform_usage: bool = False,
        name: str = 'transcript'
    ):
        """
        Add transcript-level data.

        Parameters
        ----------
        transcript_counts : pd.DataFrame
            Transcript counts (transcripts × cells)
        transcript_meta : pd.DataFrame
            Transcript metadata with at least 'transcript_id' and 'gene' columns
        use_isoform_usage : bool
            If True, model as multinomial (isoform usage within gene).
            If False, model as independent negative binomial.
        name : str
            Modality name
        """
        if 'transcript_id' not in transcript_meta.columns or 'gene' not in transcript_meta.columns:
            raise ValueError("transcript_meta must have 'transcript_id' and 'gene' columns")

        if use_isoform_usage:
            # Group transcripts by gene and create 3D array
            # Shape: (genes, cells, max_transcripts_per_gene)
            genes = sorted(transcript_meta['gene'].unique())
            cells = transcript_counts.columns.tolist()

            gene_to_transcripts = transcript_meta.groupby('gene')['transcript_id'].apply(list).to_dict()
            max_transcripts = max(len(t) for t in gene_to_transcripts.values())

            counts_3d = np.zeros((len(genes), len(cells), max_transcripts))

            gene_meta_rows = []
            for gene_idx, gene in enumerate(genes):
                transcripts = gene_to_transcripts[gene]
                gene_meta_rows.append({
                    'gene': gene,
                    'transcripts': transcripts,
                    'n_transcripts': len(transcripts)
                })

                for tx_idx, tx in enumerate(transcripts):
                    if tx in transcript_counts.index:
                        counts_3d[gene_idx, :, tx_idx] = transcript_counts.loc[tx].values

            gene_meta_df = pd.DataFrame(gene_meta_rows)

            modality = Modality(
                name=name,
                counts=counts_3d,
                feature_meta=gene_meta_df,
                distribution='multinomial',
                cells_axis=1
            )
        else:
            # Treat as independent transcripts with negbinom
            modality = Modality(
                name=name,
                counts=transcript_counts,
                feature_meta=transcript_meta,
                distribution='negbinom',
                cells_axis=1
            )

        self.add_modality(name, modality)

    def add_splicing_modality(
        self,
        sj_counts: pd.DataFrame,
        sj_meta: pd.DataFrame,
        splicing_types: Union[str, List[str]] = ['donor', 'acceptor', 'exon_skip'],
        gene_counts: Optional[pd.DataFrame] = None,
        min_cell_total: int = 1,
        min_total_exon: int = 2
    ):
        """
        Add splicing modalities (raw SJ counts, donor usage, acceptor usage, exon skipping).

        Parameters
        ----------
        sj_counts : pd.DataFrame
            Splice junction counts (junctions × cells)
        sj_meta : pd.DataFrame
            Junction metadata with required columns: coord.intron, chrom, intron_start,
            intron_end, strand, gene_name_start, gene_name_end
            Optional columns: gene_id_start, gene_id_end (for Ensembl ID support)
        splicing_types : str or list
            Which splicing metrics to compute: 'sj', 'donor', 'acceptor', 'exon_skip', or list
        gene_counts : pd.DataFrame, optional
            Gene-level counts for SJ denominator (genes × cells).
            If not provided, will use primary gene counts from model (self.counts).
        min_cell_total : int
            Minimum reads for donor/acceptor
        min_total_exon : int
            Minimum reads for exon skipping
        """
        if isinstance(splicing_types, str):
            splicing_types = [splicing_types]

        # Get gene_counts (use model's counts if not provided)
        if gene_counts is None:
            if hasattr(self, 'counts') and self.counts is not None:
                gene_counts_to_use = self.counts
            else:
                raise ValueError("gene_counts must be provided or model must have been initialized with counts")
        else:
            gene_counts_to_use = gene_counts

        # Subset sj_counts to valid cells (cells in model.meta)
        # Allow sj_counts to have extra cells (will be discarded)
        # Also allow sj_counts to be missing some cells (that's OK)
        valid_cells = self.meta['cell'].tolist()
        sj_cells = sj_counts.columns.tolist()
        common_sj_cells = [c for c in sj_cells if c in valid_cells]

        if len(common_sj_cells) == 0:
            raise ValueError("No overlapping cells between sj_counts and model cells")

        if len(common_sj_cells) < len(sj_cells):
            print(f"[INFO] Subsetting sj_counts from {len(sj_cells)} to {len(common_sj_cells)} cells to match model")
            sj_counts_subset = sj_counts[common_sj_cells].copy()
        else:
            sj_counts_subset = sj_counts

        for stype in splicing_types:
            modality = create_splicing_modality(
                sj_counts=sj_counts_subset,
                sj_meta=sj_meta,
                splicing_type=stype,
                gene_counts=gene_counts_to_use,
                min_cell_total=min_cell_total,
                min_total_exon=min_total_exon
            )
            self.add_modality(f'splicing_{stype}', modality)

    def add_custom_modality(
        self,
        name: str,
        counts: Union[np.ndarray, pd.DataFrame],
        feature_meta: pd.DataFrame,
        distribution: str,
        denominator: Optional[np.ndarray] = None
    ):
        """
        Add a custom user-defined modality.

        Parameters
        ----------
        name : str
            Modality name
        counts : array or DataFrame
            Measurement data
        feature_meta : pd.DataFrame
            Feature metadata
        distribution : str
            'negbinom', 'multinomial', 'binomial', 'normal', or 'mvnormal'
        denominator : array, optional
            For binomial: denominator counts
        """
        modality = Modality(
            name=name,
            counts=counts,
            feature_meta=feature_meta,
            distribution=distribution,
            denominator=denominator,
            cells_axis=1 if isinstance(counts, pd.DataFrame) else 1
        )
        self.add_modality(name, modality)

    def get_modality(self, name: str) -> Modality:
        """Get a modality by name."""
        if name not in self.modalities:
            raise KeyError(f"Modality '{name}' not found. Available: {list(self.modalities.keys())}")
        return self.modalities[name]

    def list_modalities(self) -> pd.DataFrame:
        """
        Get summary of all modalities.

        Returns
        -------
        pd.DataFrame
            Summary table with modality names, distributions, and dimensions
        """
        rows = []
        for name, mod in self.modalities.items():
            row = {
                'modality': name,
                'distribution': mod.distribution,
                **mod.dims,
                'is_primary': name == self.primary_modality
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def fit_modality_technical(
        self,
        modality_name: str,
        covariates: list,
        sum_factor_col: str = 'sum_factor',
        **kwargs
    ):
        """
        Fit technical model for a specific modality.

        This is a placeholder for future implementation of modality-specific
        technical fits. Currently only supports the primary gene modality.

        Parameters
        ----------
        modality_name : str
            Which modality to fit
        covariates : list
            Covariates for technical variation
        sum_factor_col : str
            Sum factor column
        **kwargs
            Additional arguments passed to fit_technical
        """
        if modality_name != self.primary_modality:
            raise NotImplementedError(
                f"Technical fitting for non-primary modalities not yet implemented. "
                f"Primary modality is '{self.primary_modality}'"
            )

        return self.fit_technical(covariates=covariates, sum_factor_col=sum_factor_col, **kwargs)

    def __repr__(self) -> str:
        mod_list = ', '.join(self.modalities.keys())
        return (f"MultiModalBayesDREAM(label='{self.label}', "
                f"primary='{self.primary_modality}', "
                f"modalities=[{mod_list}])")
