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
from .distributions import get_observation_sampler, requires_denominator, is_3d_distribution


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

            # Check if cis gene has zero variance (critical for cis modeling)
            if cis_gene is not None and cis_gene in counts.index:
                cis_gene_std = counts.loc[cis_gene].std()
                if cis_gene_std == 0:
                    raise ValueError(
                        f"Cis gene '{cis_gene}' has zero standard deviation across all cells. "
                        f"Cannot model cis effects for a gene with constant expression."
                    )

            # Exclude cis gene from gene count modality features
            # (The base class still gets the full counts with cis gene for cis modeling)
            if cis_gene is not None and cis_gene in counts.index:
                print(f"[INFO] Excluding cis gene '{cis_gene}' from gene count modality features")
                counts_trans = counts.drop(index=cis_gene)
            else:
                counts_trans = counts

            # Filter genes with zero standard deviation across ALL cells
            # (genes that are constant across all cells can't be modeled)
            gene_stds = counts_trans.std(axis=1)
            zero_std_mask = gene_stds == 0
            num_zero_std = zero_std_mask.sum()

            if num_zero_std > 0:
                print(f"[INFO] Filtering {num_zero_std} gene(s) with zero standard deviation across all cells")
                counts_trans = counts_trans.loc[~zero_std_mask]

            if len(counts_trans) == 0:
                raise ValueError("No genes left after filtering genes with zero standard deviation!")

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
        modality_types: Union[str, List[str]] = 'counts',
        counts_name: str = 'transcript_counts',
        usage_name: str = 'transcript_usage'
    ):
        """
        Add transcript-level data as counts and/or isoform usage.

        Parameters
        ----------
        transcript_counts : pd.DataFrame
            Transcript counts (transcripts × cells)
        transcript_meta : pd.DataFrame
            Transcript metadata with required columns: 'transcript_id' and either
            'gene' or 'gene_name' or 'gene_id' (gene symbol or Ensembl ID)
        modality_types : str or list of str
            Which modalities to add: 'counts', 'usage', or both.
            Options: 'counts' (negbinom), 'usage' (multinomial), or ['counts', 'usage']
        counts_name : str
            Name for transcript counts modality (default: 'transcript_counts')
        usage_name : str
            Name for isoform usage modality (default: 'transcript_usage')
        """
        # Validate required columns
        if 'transcript_id' not in transcript_meta.columns:
            raise ValueError("transcript_meta must have 'transcript_id' column")

        # Flexible gene column detection (gene, gene_name, gene_id)
        gene_col = None
        for col in ['gene', 'gene_name', 'gene_id']:
            if col in transcript_meta.columns:
                gene_col = col
                break

        if gene_col is None:
            raise ValueError("transcript_meta must have one of: 'gene', 'gene_name', or 'gene_id' column")

        print(f"[INFO] Using '{gene_col}' column for gene identifiers")

        # Standardize to 'gene' column for internal processing
        if gene_col != 'gene':
            transcript_meta = transcript_meta.copy()
            transcript_meta['gene'] = transcript_meta[gene_col]

        # Validate transcript_id values are in transcript_counts
        meta_transcripts = set(transcript_meta['transcript_id'])
        count_transcripts = set(transcript_counts.index)
        missing_in_counts = meta_transcripts - count_transcripts

        if missing_in_counts:
            print(f"[INFO] {len(missing_in_counts)} transcript(s) in metadata not found in counts (will be skipped)")

        # Subset transcript_counts to valid cells
        valid_cells = self.meta['cell'].tolist()
        tx_cells = transcript_counts.columns.tolist()
        common_tx_cells = [c for c in tx_cells if c in valid_cells]

        if len(common_tx_cells) == 0:
            raise ValueError("No overlapping cells between transcript_counts and model cells")

        if len(common_tx_cells) < len(tx_cells):
            print(f"[INFO] Subsetting transcript_counts from {len(tx_cells)} to {len(common_tx_cells)} cells to match model")
            transcript_counts_subset = transcript_counts[common_tx_cells].copy()
        else:
            transcript_counts_subset = transcript_counts

        # Normalize modality_types to list
        if isinstance(modality_types, str):
            modality_types = [modality_types]

        # Validate modality types
        valid_types = {'counts', 'usage'}
        invalid = set(modality_types) - valid_types
        if invalid:
            raise ValueError(f"Invalid modality_types: {invalid}. Must be 'counts', 'usage', or both.")

        # Add counts modality
        if 'counts' in modality_types:
            # Subset metadata to transcripts present in counts
            valid_tx = [tx for tx in transcript_meta['transcript_id'] if tx in transcript_counts_subset.index]
            tx_meta_subset = transcript_meta[transcript_meta['transcript_id'].isin(valid_tx)].copy()

            if len(valid_tx) == 0:
                warnings.warn(f"No transcripts found in counts. Skipping '{counts_name}' modality.")
            else:
                # Ensure counts are in same order as metadata
                transcript_counts_ordered = transcript_counts_subset.loc[valid_tx]

                # Filter transcripts with zero standard deviation across ALL cells
                # (transcripts that are constant across all cells can't be modeled)
                tx_stds = transcript_counts_ordered.std(axis=1)
                zero_std_mask = tx_stds == 0
                num_zero_std = zero_std_mask.sum()

                if num_zero_std > 0:
                    print(f"[INFO] Filtering {num_zero_std} transcript(s) with zero standard deviation across all cells")
                    transcript_counts_ordered = transcript_counts_ordered.loc[~zero_std_mask]
                    tx_meta_subset = tx_meta_subset[tx_meta_subset['transcript_id'].isin(transcript_counts_ordered.index)].copy()

                if len(transcript_counts_ordered) == 0:
                    warnings.warn(f"No transcripts left after filtering zero-variance transcripts. Skipping '{counts_name}' modality.")
                else:
                    modality = Modality(
                        name=counts_name,
                        counts=transcript_counts_ordered,
                        feature_meta=tx_meta_subset.reset_index(drop=True),
                        distribution='negbinom',
                        cells_axis=1
                    )
                    self.add_modality(counts_name, modality)

        # Add usage modality
        if 'usage' in modality_types:
            # Group transcripts by gene and create 3D array
            # Shape: (genes, cells, max_transcripts_per_gene)
            gene_to_transcripts = transcript_meta.groupby('gene')['transcript_id'].apply(list).to_dict()

            # Filter to transcripts present in counts
            gene_to_transcripts_filtered = {
                gene: [tx for tx in txs if tx in transcript_counts_subset.index]
                for gene, txs in gene_to_transcripts.items()
            }

            # Remove genes with no transcripts
            gene_to_transcripts_filtered = {
                gene: txs for gene, txs in gene_to_transcripts_filtered.items() if len(txs) > 0
            }

            if len(gene_to_transcripts_filtered) == 0:
                warnings.warn(f"No genes with transcripts found in counts. Skipping '{usage_name}' modality.")
            else:
                genes = sorted(gene_to_transcripts_filtered.keys())
                n_cells = len(common_tx_cells)
                max_transcripts = max(len(txs) for txs in gene_to_transcripts_filtered.values())

                counts_3d = np.zeros((len(genes), n_cells, max_transcripts))

                gene_meta_rows = []
                n_genes_dropped = 0

                for gene_idx, gene in enumerate(genes):
                    transcripts = gene_to_transcripts_filtered[gene]

                    # Skip genes with only one transcript (no isoform variation)
                    if len(transcripts) < 2:
                        n_genes_dropped += 1
                        continue

                    gene_meta_rows.append({
                        'gene': gene,
                        'transcripts': transcripts,
                        'n_transcripts': len(transcripts)
                    })

                    for tx_idx, tx in enumerate(transcripts):
                        if tx in transcript_counts_subset.index:
                            counts_3d[gene_idx, :, tx_idx] = transcript_counts_subset.loc[tx].values

                if n_genes_dropped > 0:
                    print(f"[INFO] Dropped {n_genes_dropped} gene(s) with only 1 transcript (no isoform usage to model)")

                if len(gene_meta_rows) == 0:
                    warnings.warn(f"No genes with multiple transcripts found. Skipping '{usage_name}' modality.")
                else:
                    # Filter counts_3d to genes with multiple transcripts
                    valid_gene_indices = [i for i, gene in enumerate(genes)
                                         if len(gene_to_transcripts_filtered[gene]) >= 2]
                    counts_3d = counts_3d[valid_gene_indices, :, :]

                    gene_meta_df = pd.DataFrame(gene_meta_rows)

                    modality = Modality(
                        name=usage_name,
                        counts=counts_3d,
                        feature_meta=gene_meta_df,
                        distribution='multinomial',
                        cells_axis=1
                    )
                    self.add_modality(usage_name, modality)

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
        Add a custom user-defined modality with distribution-specific filtering.

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
        # Convert counts to ndarray for consistent filtering
        if isinstance(counts, pd.DataFrame):
            counts_array = counts.values
            is_dataframe = True
            counts_index = counts.index
        else:
            counts_array = np.asarray(counts)
            is_dataframe = False
            counts_index = None

        # Apply distribution-specific filtering
        valid_features = None

        if distribution in ['negbinom', 'normal']:
            # Filter features with zero standard deviation
            if counts_array.ndim == 2:
                feature_stds = counts_array.std(axis=1)
                valid_features = feature_stds != 0
                num_filtered = (~valid_features).sum()
                if num_filtered > 0:
                    print(f"[INFO] Filtering {num_filtered} feature(s) with zero std in '{name}' modality ({distribution})")

        elif distribution == 'binomial':
            # Filter features with zero variance in numerator/denominator ratio
            if denominator is None:
                raise ValueError(f"denominator required for binomial distribution in '{name}' modality")

            if isinstance(denominator, pd.DataFrame):
                denom_array = denominator.values
            else:
                denom_array = np.asarray(denominator)

            if counts_array.shape != denom_array.shape:
                raise ValueError(f"counts and denominator must have same shape for binomial in '{name}' modality")

            if counts_array.ndim == 2:
                n_features = counts_array.shape[0]
                valid_features = np.ones(n_features, dtype=bool)

                for i in range(n_features):
                    numer = counts_array[i, :]
                    denom = denom_array[i, :]

                    # Compute ratios, excluding cells where denominator is 0
                    valid_mask = denom > 0
                    if valid_mask.sum() == 0:
                        valid_features[i] = False
                        continue

                    ratios = numer[valid_mask] / denom[valid_mask]
                    if ratios.std() == 0:
                        valid_features[i] = False

                num_filtered = (~valid_features).sum()
                if num_filtered > 0:
                    print(f"[INFO] Filtering {num_filtered} feature(s) with zero ratio variance in '{name}' modality (binomial)")

        elif distribution == 'multinomial':
            # Filter features where ALL category ratios have zero variance
            if counts_array.ndim != 3:
                raise ValueError(f"multinomial requires 3D counts (features, cells, categories) in '{name}' modality")

            n_features = counts_array.shape[0]
            valid_features = np.ones(n_features, dtype=bool)

            for i in range(n_features):
                feature_counts = counts_array[i, :, :]  # (cells, categories)
                totals = feature_counts.sum(axis=1, keepdims=True)  # (cells, 1)

                with np.errstate(divide='ignore', invalid='ignore'):
                    ratios = np.where(totals > 0, feature_counts / totals, 0)  # (cells, categories)

                # Check if ALL ratios have zero std across cells
                ratio_stds = ratios.std(axis=0)  # std for each category across cells
                if np.all(ratio_stds == 0):
                    valid_features[i] = False

            num_filtered = (~valid_features).sum()
            if num_filtered > 0:
                print(f"[INFO] Filtering {num_filtered} feature(s) with zero variance in ALL category ratios in '{name}' modality (multinomial)")

        elif distribution == 'mvnormal':
            # Filter features where ALL dimensions have zero variance
            if counts_array.ndim != 3:
                raise ValueError(f"mvnormal requires 3D counts (features, cells, dimensions) in '{name}' modality")

            n_features = counts_array.shape[0]
            valid_features = np.ones(n_features, dtype=bool)

            for i in range(n_features):
                feature_data = counts_array[i, :, :]  # (cells, dimensions)
                dim_stds = feature_data.std(axis=0)  # std for each dimension
                if np.all(dim_stds == 0):
                    valid_features[i] = False

            num_filtered = (~valid_features).sum()
            if num_filtered > 0:
                print(f"[INFO] Filtering {num_filtered} feature(s) with zero variance in ALL dimensions in '{name}' modality (mvnormal)")

        # Apply filtering if necessary
        if valid_features is not None:
            if not np.any(valid_features):
                raise ValueError(f"No features left after filtering zero-variance features in '{name}' modality!")

            if not np.all(valid_features):
                # Apply mask
                counts_array = counts_array[valid_features]
                feature_meta = feature_meta.iloc[valid_features].copy()
                if denominator is not None:
                    if isinstance(denominator, pd.DataFrame):
                        denominator = denominator.iloc[valid_features]
                    else:
                        denominator = denom_array[valid_features]

        # Convert back to DataFrame if original was DataFrame
        if is_dataframe:
            if counts_array.ndim == 2:
                counts_final = pd.DataFrame(
                    counts_array,
                    index=feature_meta.index,
                    columns=counts_index if counts_index is not None else range(counts_array.shape[1])
                )
            else:
                # For 3D data, keep as ndarray
                counts_final = counts_array
        else:
            counts_final = counts_array

        modality = Modality(
            name=name,
            counts=counts_final,
            feature_meta=feature_meta,
            distribution=distribution,
            denominator=denominator,
            cells_axis=1
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

    def __repr__(self) -> str:
        mod_list = ', '.join(self.modalities.keys())
        return (f"MultiModalBayesDREAM(label='{self.label}', "
                f"primary='{self.primary_modality}', "
                f"modalities=[{mod_list}])")
