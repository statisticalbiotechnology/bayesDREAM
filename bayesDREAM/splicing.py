"""
Splicing data processing for bayesDREAM.

Pure Python implementation of donor/acceptor usage and exon skipping metrics.
Previously relied on R functions from CodeDump.R, now fully in Python.
"""

import warnings
from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd

from .modality import Modality


def _normalize_strand(strand_values):
    """
    Normalize strand notation to '+'/'-'.

    Accepts: 1/2 (integers), '+'/'-', 'plus'/'minus'
    Returns: '+' or '-' or None for invalid
    """
    def norm_single(x):
        if pd.isna(x):
            return None
        if isinstance(x, (int, np.integer)):
            return '+' if x == 1 else ('-' if x == 2 else None)
        s = str(x).lower()
        if s in ['+', '1', 'plus']:
            return '+'
        if s in ['-', '2', 'minus']:
            return '-'
        return None

    if isinstance(strand_values, (list, pd.Series, np.ndarray)):
        return [norm_single(x) for x in strand_values]
    else:
        return norm_single(strand_values)


def _build_sj_index(sj_counts: pd.DataFrame,
                    sj_meta: pd.DataFrame,
                    gene_of_interest: Optional[str] = None) -> pd.DataFrame:
    """
    Build splice junction index with donor/acceptor annotations.

    Parameters
    ----------
    sj_counts : pd.DataFrame
        Splice junction counts (junctions × cells)
    sj_meta : pd.DataFrame
        Junction metadata with: coord.intron, chrom, intron_start, intron_end, strand
    gene_of_interest : str, optional
        Filter to specific gene

    Returns
    -------
    pd.DataFrame
        Indexed SJ metadata with donor/acceptor positions
    """
    # Validate required columns
    required = ['coord.intron', 'chrom', 'intron_start', 'intron_end', 'strand']
    missing = [c for c in required if c not in sj_meta.columns]
    if missing:
        raise ValueError(f"sj_meta missing required columns: {missing}")

    # Copy and prepare
    idx = sj_meta.copy()
    idx['strand'] = _normalize_strand(idx['strand'].values)
    idx['start'] = idx['intron_start'].astype(int)
    idx['end'] = idx['intron_end'].astype(int)

    # Define donor (5'SS) and acceptor (3'SS) based on strand
    idx['donor'] = np.where(idx['strand'] == '+', idx['start'], idx['end'])
    idx['acceptor'] = np.where(idx['strand'] == '+', idx['end'], idx['start'])

    # Genomic coordinates (for exon skipping)
    idx['left'] = np.minimum(idx['start'], idx['end'])
    idx['right'] = np.maximum(idx['start'], idx['end'])

    # Filter to gene of interest if specified
    if gene_of_interest is not None:
        gene_cols = [c for c in idx.columns if 'gene' in c.lower()]
        if gene_cols:
            # Check if any gene column contains the target gene
            mask = pd.Series([False] * len(idx))
            for col in gene_cols:
                mask |= (idx[col] == gene_of_interest)
            idx = idx[mask].copy()
            if len(idx) == 0:
                raise ValueError(f"No splice junctions found for gene: {gene_of_interest}")
        else:
            warnings.warn(f"gene_of_interest='{gene_of_interest}' specified but no gene columns in sj_meta")

    # Keep only junctions present in counts matrix
    present = sj_counts.index.tolist()
    idx = idx[idx['coord.intron'].isin(present)].copy()

    if len(idx) == 0:
        raise ValueError("No splice junctions overlap between sj_counts and sj_meta")

    return idx


def process_donor_usage(sj_counts: pd.DataFrame,
                       sj_meta: pd.DataFrame,
                       gene_of_interest: Optional[str] = None,
                       min_cell_total: int = 1) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """
    Compute donor usage (which acceptor is used for each donor site).

    For each donor site (5' splice site), count how many reads go to each
    possible acceptor (3' splice site). Returns multinomial counts.

    Parameters
    ----------
    sj_counts : pd.DataFrame
        Splice junction counts (junctions × cells)
    sj_meta : pd.DataFrame
        Junction metadata
    gene_of_interest : str, optional
        Filter to specific gene
    min_cell_total : int
        Minimum total reads per donor per cell to include

    Returns
    -------
    counts_3d : np.ndarray
        Shape: (n_donors, n_cells, max_acceptors_per_donor)
    feature_meta : pd.DataFrame
        Donor site metadata with columns: chrom, strand, donor, acceptors (list), n_acceptors
    cell_names : list
        Cell identifiers
    """
    idx = _build_sj_index(sj_counts, sj_meta, gene_of_interest)

    # Group by (chrom, strand, donor)
    donor_groups = idx.groupby(['chrom', 'strand', 'donor'], sort=True)

    # Get cell names
    cell_names = sj_counts.columns.tolist()
    n_cells = len(cell_names)

    # Build feature metadata and 3D array
    feature_rows = []
    all_counts = []

    for (chrom, strand, donor), group in donor_groups:
        # Get all acceptors for this donor
        acceptors = sorted(group['acceptor'].unique())
        n_acceptors = len(acceptors)

        # Build mapping from acceptor to index
        acceptor_to_idx = {acc: i for i, acc in enumerate(acceptors)}

        # Initialize counts for this donor: (n_cells, n_acceptors)
        donor_counts = np.zeros((n_cells, n_acceptors), dtype=float)

        # Fill in counts for each junction
        for _, row in group.iterrows():
            coord = row['coord.intron']
            if coord not in sj_counts.index:
                continue
            acceptor = row['acceptor']
            acc_idx = acceptor_to_idx[acceptor]

            # Add counts for all cells
            donor_counts[:, acc_idx] += sj_counts.loc[coord].values

        # Apply min_cell_total filter
        if min_cell_total > 0:
            cell_totals = donor_counts.sum(axis=1)
            donor_counts[cell_totals < min_cell_total, :] = 0

        all_counts.append(donor_counts)

        feature_rows.append({
            'chrom': chrom,
            'strand': strand,
            'donor': donor,
            'acceptors': acceptors,
            'n_acceptors': n_acceptors
        })

    # Stack into 3D array: (n_donors, n_cells, max_acceptors)
    max_acceptors = max(row['n_acceptors'] for row in feature_rows)
    n_donors = len(feature_rows)

    counts_3d = np.zeros((n_donors, n_cells, max_acceptors), dtype=float)
    for i, donor_counts in enumerate(all_counts):
        n_acc = donor_counts.shape[1]
        counts_3d[i, :, :n_acc] = donor_counts

    feature_meta = pd.DataFrame(feature_rows)

    return counts_3d, feature_meta, cell_names


def process_acceptor_usage(sj_counts: pd.DataFrame,
                           sj_meta: pd.DataFrame,
                           gene_of_interest: Optional[str] = None,
                           min_cell_total: int = 1) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """
    Compute acceptor usage (which donor is used for each acceptor site).

    For each acceptor site (3' splice site), count how many reads come from each
    possible donor (5' splice site). Returns multinomial counts.

    Parameters
    ----------
    sj_counts : pd.DataFrame
        Splice junction counts (junctions × cells)
    sj_meta : pd.DataFrame
        Junction metadata
    gene_of_interest : str, optional
        Filter to specific gene
    min_cell_total : int
        Minimum total reads per acceptor per cell to include

    Returns
    -------
    counts_3d : np.ndarray
        Shape: (n_acceptors, n_cells, max_donors_per_acceptor)
    feature_meta : pd.DataFrame
        Acceptor site metadata with columns: chrom, strand, acceptor, donors (list), n_donors
    cell_names : list
        Cell identifiers
    """
    idx = _build_sj_index(sj_counts, sj_meta, gene_of_interest)

    # Group by (chrom, strand, acceptor)
    acceptor_groups = idx.groupby(['chrom', 'strand', 'acceptor'], sort=True)

    # Get cell names
    cell_names = sj_counts.columns.tolist()
    n_cells = len(cell_names)

    # Build feature metadata and 3D array
    feature_rows = []
    all_counts = []

    for (chrom, strand, acceptor), group in acceptor_groups:
        # Get all donors for this acceptor
        donors = sorted(group['donor'].unique())
        n_donors = len(donors)

        # Build mapping from donor to index
        donor_to_idx = {don: i for i, don in enumerate(donors)}

        # Initialize counts for this acceptor: (n_cells, n_donors)
        acceptor_counts = np.zeros((n_cells, n_donors), dtype=float)

        # Fill in counts for each junction
        for _, row in group.iterrows():
            coord = row['coord.intron']
            if coord not in sj_counts.index:
                continue
            donor = row['donor']
            don_idx = donor_to_idx[donor]

            # Add counts for all cells
            acceptor_counts[:, don_idx] += sj_counts.loc[coord].values

        # Apply min_cell_total filter
        if min_cell_total > 0:
            cell_totals = acceptor_counts.sum(axis=1)
            acceptor_counts[cell_totals < min_cell_total, :] = 0

        all_counts.append(acceptor_counts)

        feature_rows.append({
            'chrom': chrom,
            'strand': strand,
            'acceptor': acceptor,
            'donors': donors,
            'n_donors': n_donors
        })

    # Stack into 3D array: (n_acceptors, n_cells, max_donors)
    max_donors = max(row['n_donors'] for row in feature_rows)
    n_acceptors = len(feature_rows)

    counts_3d = np.zeros((n_acceptors, n_cells, max_donors), dtype=float)
    for i, acceptor_counts in enumerate(all_counts):
        n_don = acceptor_counts.shape[1]
        counts_3d[i, :, :n_don] = acceptor_counts

    feature_meta = pd.DataFrame(feature_rows)

    return counts_3d, feature_meta, cell_names


def _find_cassette_triplets_strand(sj_counts: pd.DataFrame,
                                   sj_meta: pd.DataFrame,
                                   gene_of_interest: Optional[str] = None) -> pd.DataFrame:
    """
    Find cassette exon triplets using strand-aware coordinates.

    A cassette exon event consists of:
    - sj_skip: Junction that skips the exon (donor d1 -> acceptor a3)
    - sj_inc1: Junction including the exon's 5' end (donor d1 -> acceptor a2)
    - sj_inc2: Junction including the exon's 3' end (donor d2 -> acceptor a3)

    Strand-specific ordering:
    - Plus strand: d1 < a2 < d2 < a3
    - Minus strand: d1 > a2 > d2 > a3

    Returns
    -------
    pd.DataFrame
        Columns: trip_id, chrom, strand, d1, a2, d2, a3, sj_inc1, sj_inc2, sj_skip
    """
    idx = _build_sj_index(sj_counts, sj_meta, gene_of_interest)

    # Keep only clean junctions
    idx = idx[idx['strand'].notna() & idx['donor'].notna() & idx['acceptor'].notna()].copy()
    idx = idx.drop_duplicates(subset=['chrom', 'strand', 'donor', 'acceptor', 'coord.intron'])

    if len(idx) == 0:
        return pd.DataFrame(columns=['trip_id', 'chrom', 'strand', 'd1', 'a2', 'd2', 'a3',
                                    'sj_inc1', 'sj_inc2', 'sj_skip'])

    # Build efficient lookup structures
    idx_by_donor = idx.groupby(['chrom', 'strand', 'donor'])
    idx_by_acceptor = idx.groupby(['chrom', 'strand', 'acceptor'])
    idx_by_pair = idx.set_index(['chrom', 'strand', 'donor', 'acceptor'])['coord.intron'].to_dict()

    triplets = []

    # For each potential skip junction
    for _, row in idx.iterrows():
        chrom = row['chrom']
        strand = row['strand']
        d1 = row['donor']
        a3 = row['acceptor']
        sj_skip = row['coord.intron']

        # Find all acceptors from d1
        try:
            group_d1 = idx_by_donor.get_group((chrom, strand, d1))
            a2_candidates = group_d1['acceptor'].unique()
        except KeyError:
            continue

        # Find all donors to a3
        try:
            group_a3 = idx_by_acceptor.get_group((chrom, strand, a3))
            d2_candidates = group_a3['donor'].unique()
        except KeyError:
            continue

        # Check all combinations
        for a2 in a2_candidates:
            if pd.isna(a2):
                continue
            for d2 in d2_candidates:
                if pd.isna(d2):
                    continue

                # Verify strand-specific ordering
                if strand == '+':
                    if not (d1 < a2 < d2 < a3):
                        continue
                else:  # strand == '-'
                    if not (d1 > a2 > d2 > a3):
                        continue

                # Check if inclusion junctions exist
                sj_inc1 = idx_by_pair.get((chrom, strand, d1, a2))
                sj_inc2 = idx_by_pair.get((chrom, strand, d2, a3))

                if sj_inc1 is None or sj_inc2 is None:
                    continue

                triplets.append({
                    'chrom': chrom,
                    'strand': strand,
                    'd1': d1,
                    'a2': a2,
                    'd2': d2,
                    'a3': a3,
                    'sj_inc1': sj_inc1,
                    'sj_inc2': sj_inc2,
                    'sj_skip': sj_skip
                })

    if not triplets:
        return pd.DataFrame(columns=['trip_id', 'chrom', 'strand', 'd1', 'a2', 'd2', 'a3',
                                    'sj_inc1', 'sj_inc2', 'sj_skip'])

    trips_df = pd.DataFrame(triplets)
    trips_df = trips_df.drop_duplicates()
    trips_df['trip_id'] = range(len(trips_df))

    return trips_df


def _find_cassette_triplets_genomic(sj_counts: pd.DataFrame,
                                    sj_meta: pd.DataFrame,
                                    gene_of_interest: Optional[str] = None) -> pd.DataFrame:
    """
    Find cassette exon triplets using genomic coordinates (fallback when strand info is poor).

    Uses left/right genomic positions instead of strand-aware donor/acceptor.
    Pattern: L1 < R2 < L2 < R3

    Returns
    -------
    pd.DataFrame
        Columns: trip_id, chrom, strand, d1, a2, d2, a3, sj_inc1, sj_inc2, sj_skip
    """
    idx = _build_sj_index(sj_counts, sj_meta, gene_of_interest)

    # Keep only clean junctions
    idx = idx[idx['left'].notna() & idx['right'].notna()].copy()

    if len(idx) == 0:
        return pd.DataFrame(columns=['trip_id', 'chrom', 'strand', 'd1', 'a2', 'd2', 'a3',
                                    'sj_inc1', 'sj_inc2', 'sj_skip'])

    # Build lookup structures
    idx_by_left = idx.groupby(['chrom', 'left'])
    idx_by_right = idx.groupby(['chrom', 'right'])
    idx_by_coords = idx.set_index(['chrom', 'left', 'right'])['coord.intron'].to_dict()

    triplets = []

    for _, row in idx.iterrows():
        chrom = row['chrom']
        L1 = row['left']
        R3 = row['right']
        sj_skip = row['coord.intron']
        strand = row.get('strand', None)

        # Find junctions with left=L1 and right between L1 and R3
        try:
            group_L1 = idx_by_left.get_group((chrom, L1))
            R2_candidates = group_L1[(group_L1['right'] > L1) & (group_L1['right'] < R3)]['right'].unique()
        except KeyError:
            continue

        # Find junctions with right=R3 and left between L1 and R3
        try:
            group_R3 = idx_by_right.get_group((chrom, R3))
            L2_candidates = group_R3[(group_R3['left'] > L1) & (group_R3['left'] < R3)]['left'].unique()
        except KeyError:
            continue

        for R2 in R2_candidates:
            sj_inc1 = idx_by_coords.get((chrom, L1, R2))
            if sj_inc1 is None:
                continue

            for L2 in L2_candidates:
                sj_inc2 = idx_by_coords.get((chrom, L2, R3))
                if sj_inc2 is None:
                    continue

                triplets.append({
                    'chrom': chrom,
                    'strand': strand if pd.notna(strand) else None,
                    'd1': L1,
                    'a2': R2,
                    'd2': L2,
                    'a3': R3,
                    'sj_inc1': sj_inc1,
                    'sj_inc2': sj_inc2,
                    'sj_skip': sj_skip
                })

    if not triplets:
        return pd.DataFrame(columns=['trip_id', 'chrom', 'strand', 'd1', 'a2', 'd2', 'a3',
                                    'sj_inc1', 'sj_inc2', 'sj_skip'])

    trips_df = pd.DataFrame(triplets)
    trips_df = trips_df.drop_duplicates()
    trips_df['trip_id'] = range(len(trips_df))

    return trips_df


def process_exon_skipping(sj_counts: pd.DataFrame,
                         sj_meta: pd.DataFrame,
                         gene_of_interest: Optional[str] = None,
                         min_total_exon: int = 2,
                         method: str = 'min',
                         fallback_genomic: bool = True) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    """
    Compute exon skipping (cassette exon) inclusion counts.

    For each cassette exon event, compute:
    - inc: Inclusion reads (min or mean of inc1 and inc2)
    - tot: Total reads (inc + skip)

    Returns binomial data (inclusion count, total count).

    Parameters
    ----------
    sj_counts : pd.DataFrame
        Splice junction counts (junctions × cells)
    sj_meta : pd.DataFrame
        Junction metadata
    gene_of_interest : str, optional
        Filter to specific gene
    min_total_exon : int
        Minimum total reads per event per cell
    method : str
        'min' (default) or 'mean' for computing inclusion from inc1 and inc2
    fallback_genomic : bool
        If True, use genomic coordinates when strand-aware search finds no events

    Returns
    -------
    inclusion_counts : np.ndarray
        Shape: (n_events, n_cells)
    total_counts : np.ndarray
        Shape: (n_events, n_cells)
    feature_meta : pd.DataFrame
        Event metadata with columns: trip_id, chrom, strand, d1, a2, d2, a3, sj_inc1, sj_inc2, sj_skip
    cell_names : list
        Cell identifiers
    """
    # Find cassette triplets
    trips = _find_cassette_triplets_strand(sj_counts, sj_meta, gene_of_interest)

    if len(trips) == 0 and fallback_genomic:
        print("[INFO] No strand-aware cassette exons found, trying genomic coordinates...")
        trips = _find_cassette_triplets_genomic(sj_counts, sj_meta, gene_of_interest)

    if len(trips) == 0:
        # Return empty arrays
        cell_names = sj_counts.columns.tolist()
        return (np.zeros((0, len(cell_names))),
                np.zeros((0, len(cell_names))),
                pd.DataFrame(columns=['trip_id', 'chrom', 'strand', 'd1', 'a2', 'd2', 'a3',
                                     'sj_inc1', 'sj_inc2', 'sj_skip']),
                cell_names)

    cell_names = sj_counts.columns.tolist()
    n_cells = len(cell_names)
    n_events = len(trips)

    # Initialize arrays
    inclusion_counts = np.zeros((n_events, n_cells), dtype=float)
    total_counts = np.zeros((n_events, n_cells), dtype=float)

    # Get counts for all needed junctions
    needed_sjs = pd.concat([trips['sj_inc1'], trips['sj_inc2'], trips['sj_skip']]).unique()
    needed_sjs = [sj for sj in needed_sjs if sj in sj_counts.index]
    sj_data = sj_counts.loc[needed_sjs]

    for i, row in trips.iterrows():
        sj_inc1 = row['sj_inc1']
        sj_inc2 = row['sj_inc2']
        sj_skip = row['sj_skip']

        # Get counts
        inc1 = sj_data.loc[sj_inc1].values if sj_inc1 in sj_data.index else np.zeros(n_cells)
        inc2 = sj_data.loc[sj_inc2].values if sj_inc2 in sj_data.index else np.zeros(n_cells)
        skip = sj_data.loc[sj_skip].values if sj_skip in sj_data.index else np.zeros(n_cells)

        # Compute inclusion
        if method == 'min':
            inc = np.minimum(inc1, inc2)
        elif method == 'mean':
            inc = (inc1 + inc2) / 2.0
        else:
            raise ValueError(f"method must be 'min' or 'mean', got: {method}")

        # Total = inclusion + skipping
        tot = inc + skip

        # Apply minimum total filter
        if min_total_exon > 0:
            mask = tot < min_total_exon
            inc[mask] = 0
            tot[mask] = 0

        inclusion_counts[i, :] = inc
        total_counts[i, :] = tot

    return inclusion_counts, total_counts, trips, cell_names


def create_splicing_modality(sj_counts: pd.DataFrame,
                             sj_meta: pd.DataFrame,
                             splicing_type: str,
                             gene_of_interest: Optional[str] = None,
                             min_cell_total: int = 1,
                             min_total_exon: int = 2,
                             **kwargs) -> Modality:
    """
    Create a Modality object for splicing data.

    Parameters
    ----------
    sj_counts : pd.DataFrame
        Splice junction counts (junctions × cells)
    sj_meta : pd.DataFrame
        Junction metadata with: coord.intron, chrom, intron_start, intron_end, strand
    splicing_type : str
        Type of splicing metric: 'donor', 'acceptor', or 'exon_skip'
    gene_of_interest : str, optional
        Filter to specific gene
    min_cell_total : int
        Minimum reads for donor/acceptor usage
    min_total_exon : int
        Minimum reads for exon skipping
    **kwargs
        Additional arguments (e.g., method, fallback_genomic for exon_skip)

    Returns
    -------
    Modality
        Modality object with appropriate distribution and metadata
    """
    if splicing_type == 'donor':
        counts_3d, feature_meta, cell_names = process_donor_usage(
            sj_counts, sj_meta, gene_of_interest, min_cell_total
        )
        return Modality(
            name='splicing_donor',
            counts=counts_3d,
            feature_meta=feature_meta,
            distribution='multinomial',
            cells_axis=1,
            cell_names=cell_names
        )

    elif splicing_type == 'acceptor':
        counts_3d, feature_meta, cell_names = process_acceptor_usage(
            sj_counts, sj_meta, gene_of_interest, min_cell_total
        )
        return Modality(
            name='splicing_acceptor',
            counts=counts_3d,
            feature_meta=feature_meta,
            distribution='multinomial',
            cells_axis=1,
            cell_names=cell_names
        )

    elif splicing_type == 'exon_skip':
        method = kwargs.get('method', 'min')
        fallback_genomic = kwargs.get('fallback_genomic', True)

        inc_counts, tot_counts, feature_meta, cell_names = process_exon_skipping(
            sj_counts, sj_meta, gene_of_interest, min_total_exon, method, fallback_genomic
        )
        return Modality(
            name='splicing_exon_skip',
            counts=inc_counts,
            feature_meta=feature_meta,
            distribution='binomial',
            denominator=tot_counts,
            cells_axis=1,
            cell_names=cell_names
        )

    else:
        raise ValueError(f"splicing_type must be 'donor', 'acceptor', or 'exon_skip', got: {splicing_type}")
