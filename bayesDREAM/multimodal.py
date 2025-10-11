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

        # Handle gene counts
        if counts is not None:
            if 'gene' in self.modalities:
                warnings.warn("Both counts and modalities['gene'] provided. Using counts.")

            # Create gene modality
            gene_feature_meta = pd.DataFrame({
                'gene': counts.index.tolist()
            })
            self.modalities['gene'] = Modality(
                name='gene',
                counts=counts,
                feature_meta=gene_feature_meta,
                distribution='negbinom',
                cells_axis=1
            )

        # Validate primary modality exists
        if primary_modality not in self.modalities:
            raise ValueError(f"primary_modality '{primary_modality}' not found in modalities. "
                           f"Available: {list(self.modalities.keys())}")

        self.primary_modality = primary_modality

        # Get counts from primary modality for base class
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

        # Initialize base bayesDREAM with primary modality
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

        # Validate cell alignment
        if modality.cell_names is not None:
            if not set(modality.cell_names).issubset(set(self.meta['cell'].tolist())):
                warnings.warn("Some cells in new modality are not in meta. Subsetting to common cells.")
                valid_cells = [c for c in modality.cell_names if c in self.meta['cell'].tolist()]
                modality = modality.get_cell_subset(valid_cells)

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
        gene_of_interest: Optional[str] = None,
        min_cell_total: int = 1,
        min_total_exon: int = 2,
        r_code_path: Optional[str] = None
    ):
        """
        Add splicing modalities (donor usage, acceptor usage, exon skipping).

        Parameters
        ----------
        sj_counts : pd.DataFrame
            Splice junction counts (junctions × cells)
        sj_meta : pd.DataFrame
            Junction metadata with: coord.intron, chrom, intron_start, intron_end, strand
        splicing_types : str or list
            Which splicing metrics to compute: 'donor', 'acceptor', 'exon_skip', or list
        gene_of_interest : str, optional
            Filter to specific gene
        min_cell_total : int
            Minimum reads for donor/acceptor
        min_total_exon : int
            Minimum reads for exon skipping
        r_code_path : str, optional
            Path to CodeDump.R
        """
        if isinstance(splicing_types, str):
            splicing_types = [splicing_types]

        for stype in splicing_types:
            modality = create_splicing_modality(
                sj_counts=sj_counts,
                sj_meta=sj_meta,
                splicing_type=stype,
                gene_of_interest=gene_of_interest,
                min_cell_total=min_cell_total,
                min_total_exon=min_total_exon,
                r_code_path=r_code_path
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
