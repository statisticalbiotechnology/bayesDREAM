"""
Splicing data processing for bayesDREAM.

This module provides functions to compute donor usage, acceptor usage,
and exon skipping metrics from splice junction counts, wrapping R functions
from CodeDump.R.
"""

import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
from typing import Optional, Literal, Tuple
from .modality import Modality


def run_r_splicing_function(
    r_function_name: str,
    sj_counts: pd.DataFrame,
    sj_meta: pd.DataFrame,
    gene_of_interest: Optional[str] = None,
    r_code_path: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Run an R splicing function from CodeDump.R.

    Parameters
    ----------
    r_function_name : str
        Name of R function ('psi_donor_usage_strand', 'psi_acceptor_usage_strand', 'psi_exon_skipping_strand')
    sj_counts : pd.DataFrame
        Splice junction counts (junctions × cells)
    sj_meta : pd.DataFrame
        Junction metadata with columns: coord.intron, chrom, intron_start, intron_end, strand
    gene_of_interest : str, optional
        Filter to specific gene
    r_code_path : str, optional
        Path to CodeDump.R. If None, searches in splicing code/
    **kwargs
        Additional arguments passed to R function (min_cell_total, etc.)

    Returns
    -------
    pd.DataFrame
        Long-format results with columns depending on function
    """
    # Find CodeDump.R
    if r_code_path is None:
        # Try to find it relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        r_code_path = os.path.join(os.path.dirname(current_dir), 'splicing code', 'CodeDump.R')

    if not os.path.exists(r_code_path):
        raise FileNotFoundError(f"R code not found at {r_code_path}")

    # Create temp directory for data exchange
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save inputs
        counts_file = os.path.join(tmpdir, 'sj_counts.csv')
        meta_file = os.path.join(tmpdir, 'sj_meta.csv')
        output_file = os.path.join(tmpdir, 'output.csv')

        sj_counts.to_csv(counts_file)
        sj_meta.to_csv(meta_file, index=False)

        # Build R script
        r_script = f"""
library(data.table)
source("{r_code_path}")

# Load data
sj_counts <- read.csv("{counts_file}", row.names=1, check.names=FALSE)
sj_meta <- read.csv("{meta_file}")

# Create MarvelObject-like structure
MarvelObject <- list(
    sj.count.matrix = as.matrix(sj_counts),
    sj.metadata = sj_meta
)

# Run function
gene_of_interest <- {f'"{gene_of_interest}"' if gene_of_interest else 'NULL'}
min_cell_total <- {kwargs.get('min_cell_total', 0)}
min_total <- {kwargs.get('min_total_exon', 0)}

result <- {r_function_name}(
    MarvelObject = MarvelObject,
    feature_dt = sj_meta,
    gene_of_interest = gene_of_interest,
    min_cell_total = min_cell_total
)

# Save result
write.csv(result, "{output_file}", row.names=FALSE)
"""

        # Handle exon skipping parameters
        if r_function_name == 'psi_exon_skipping_strand':
            method = kwargs.get('method', 'min')
            r_script = r_script.replace('min_cell_total = min_cell_total',
                                       f'min_total = min_total, method = "{method}"')

        # Write and run R script
        script_file = os.path.join(tmpdir, 'run_splicing.R')
        with open(script_file, 'w') as f:
            f.write(r_script)

        result = subprocess.run(['Rscript', script_file], capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"R script failed:\n{result.stderr}")

        # Read results
        output_df = pd.read_csv(output_file)
        return output_df


def process_donor_usage(
    sj_counts: pd.DataFrame,
    sj_meta: pd.DataFrame,
    gene_of_interest: Optional[str] = None,
    min_cell_total: int = 1,
    r_code_path: Optional[str] = None
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Compute donor usage PSI values and return in modality-ready format.

    Parameters
    ----------
    sj_counts : pd.DataFrame
        Splice junction counts (junctions × cells)
    sj_meta : pd.DataFrame
        Junction metadata
    gene_of_interest : str, optional
        Filter to specific gene
    min_cell_total : int
        Minimum total reads at donor site per cell
    r_code_path : str, optional
        Path to CodeDump.R

    Returns
    -------
    counts_array : np.ndarray
        3D array (donors, cells, acceptors) of junction counts
    donor_meta : pd.DataFrame
        Donor site metadata (chrom, strand, donor position, list of acceptors)
    cell_order : pd.DataFrame
        Cell ordering with cell names
    """
    # Run R function
    psi_long = run_r_splicing_function(
        'psi_donor_usage_strand',
        sj_counts=sj_counts,
        sj_meta=sj_meta,
        gene_of_interest=gene_of_interest,
        min_cell_total=min_cell_total,
        r_code_path=r_code_path
    )

    # Convert to wide format grouped by donor
    # Group by (chrom, strand, donor) to get donor-level features
    donors = psi_long[['chrom', 'strand', 'donor']].drop_duplicates()
    donors['donor_id'] = range(len(donors))

    # For each donor, get all acceptors
    donor_to_acceptors = psi_long.groupby(['chrom', 'strand', 'donor'])['acceptor'].apply(
        lambda x: sorted(x.unique())
    ).to_dict()

    # Create 3D array: (donors, cells, acceptors)
    cells = sorted(psi_long['cell.id'].unique())
    max_acceptors = max(len(acc) for acc in donor_to_acceptors.values())

    counts_array = np.zeros((len(donors), len(cells), max_acceptors))

    for _, donor_row in donors.iterrows():
        donor_idx = donor_row['donor_id']
        donor_key = (donor_row['chrom'], donor_row['strand'], donor_row['donor'])
        acceptors_list = donor_to_acceptors[donor_key]

        for acc_idx, acceptor in enumerate(acceptors_list):
            # Get counts for this donor-acceptor pair
            subset = psi_long[
                (psi_long['chrom'] == donor_row['chrom']) &
                (psi_long['strand'] == donor_row['strand']) &
                (psi_long['donor'] == donor_row['donor']) &
                (psi_long['acceptor'] == acceptor)
            ]

            for _, row in subset.iterrows():
                cell_idx = cells.index(row['cell.id'])
                counts_array[donor_idx, cell_idx, acc_idx] = row['sj.count']

    # Create donor metadata
    donors['acceptors'] = donors.apply(
        lambda row: donor_to_acceptors[(row['chrom'], row['strand'], row['donor'])],
        axis=1
    )
    donors['n_acceptors'] = donors['acceptors'].apply(len)

    cell_df = pd.DataFrame({'cell': cells})

    return counts_array, donors, cell_df


def process_acceptor_usage(
    sj_counts: pd.DataFrame,
    sj_meta: pd.DataFrame,
    gene_of_interest: Optional[str] = None,
    min_cell_total: int = 1,
    r_code_path: Optional[str] = None
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Compute acceptor usage PSI values and return in modality-ready format.

    Similar to process_donor_usage but groups by acceptor site.

    Returns
    -------
    counts_array : np.ndarray
        3D array (acceptors, cells, donors) of junction counts
    acceptor_meta : pd.DataFrame
        Acceptor site metadata
    cell_order : pd.DataFrame
        Cell ordering
    """
    # Run R function
    psi_long = run_r_splicing_function(
        'psi_acceptor_usage_strand',
        sj_counts=sj_counts,
        sj_meta=sj_meta,
        gene_of_interest=gene_of_interest,
        min_cell_total=min_cell_total,
        r_code_path=r_code_path
    )

    # Group by acceptor
    acceptors = psi_long[['chrom', 'strand', 'acceptor']].drop_duplicates()
    acceptors['acceptor_id'] = range(len(acceptors))

    # For each acceptor, get all donors
    acceptor_to_donors = psi_long.groupby(['chrom', 'strand', 'acceptor'])['donor'].apply(
        lambda x: sorted(x.unique())
    ).to_dict()

    # Create 3D array: (acceptors, cells, donors)
    cells = sorted(psi_long['cell.id'].unique())
    max_donors = max(len(don) for don in acceptor_to_donors.values())

    counts_array = np.zeros((len(acceptors), len(cells), max_donors))

    for _, acc_row in acceptors.iterrows():
        acc_idx = acc_row['acceptor_id']
        acc_key = (acc_row['chrom'], acc_row['strand'], acc_row['acceptor'])
        donors_list = acceptor_to_donors[acc_key]

        for don_idx, donor in enumerate(donors_list):
            # Get counts for this acceptor-donor pair
            subset = psi_long[
                (psi_long['chrom'] == acc_row['chrom']) &
                (psi_long['strand'] == acc_row['strand']) &
                (psi_long['acceptor'] == acc_row['acceptor']) &
                (psi_long['donor'] == donor)
            ]

            for _, row in subset.iterrows():
                cell_idx = cells.index(row['cell.id'])
                counts_array[acc_idx, cell_idx, don_idx] = row['sj.count']

    # Create acceptor metadata
    acceptors['donors'] = acceptors.apply(
        lambda row: acceptor_to_donors[(row['chrom'], row['strand'], row['acceptor'])],
        axis=1
    )
    acceptors['n_donors'] = acceptors['donors'].apply(len)

    cell_df = pd.DataFrame({'cell': cells})

    return counts_array, acceptors, cell_df


def process_exon_skipping(
    sj_counts: pd.DataFrame,
    sj_meta: pd.DataFrame,
    gene_of_interest: Optional[str] = None,
    min_total: int = 2,
    method: Literal['min', 'mean'] = 'min',
    r_code_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Compute exon skipping events and return in binomial-ready format.

    Parameters
    ----------
    sj_counts : pd.DataFrame
        Splice junction counts (junctions × cells)
    sj_meta : pd.DataFrame
        Junction metadata
    gene_of_interest : str, optional
        Filter to specific gene
    min_total : int
        Minimum total reads (inclusion + skipping) per cell
    method : str
        How to combine inc1 and inc2 counts ('min' or 'mean')
    r_code_path : str, optional
        Path to CodeDump.R

    Returns
    -------
    inclusion_counts : np.ndarray
        2D array (events, cells) of inclusion counts
    total_counts : np.ndarray
        2D array (events, cells) of total counts (inclusion + skipping)
    event_meta : pd.DataFrame
        Exon skipping event metadata
    cell_order : pd.DataFrame
        Cell ordering
    """
    # Run R function
    result = run_r_splicing_function(
        'psi_exon_skipping_strand',
        sj_counts=sj_counts,
        sj_meta=sj_meta,
        gene_of_interest=gene_of_interest,
        min_total_exon=min_total,
        method=method,
        r_code_path=r_code_path
    )

    # Get unique events and cells
    events = result[['trip_id', 'chrom', 'strand', 'd1', 'a2', 'd2', 'a3',
                    'sj_inc1', 'sj_inc2', 'sj_skip']].drop_duplicates()
    events = events.sort_values('trip_id').reset_index(drop=True)

    cells = sorted(result['cell.id'].unique())

    # Create 2D arrays
    inclusion_counts = np.zeros((len(events), len(cells)))
    total_counts = np.zeros((len(events), len(cells)))

    for _, row in result.iterrows():
        event_idx = events[events['trip_id'] == row['trip_id']].index[0]
        cell_idx = cells.index(row['cell.id'])

        inclusion_counts[event_idx, cell_idx] = row['inc'] if not pd.isna(row['inc']) else 0
        total_counts[event_idx, cell_idx] = row['tot'] if not pd.isna(row['tot']) else 0

    cell_df = pd.DataFrame({'cell': cells})

    return inclusion_counts, total_counts, events, cell_df


def create_splicing_modality(
    sj_counts: pd.DataFrame,
    sj_meta: pd.DataFrame,
    splicing_type: Literal['donor', 'acceptor', 'exon_skip'],
    gene_of_interest: Optional[str] = None,
    min_cell_total: int = 1,
    min_total_exon: int = 2,
    r_code_path: Optional[str] = None
) -> Modality:
    """
    Create a Modality object for splicing data.

    Parameters
    ----------
    sj_counts : pd.DataFrame
        Splice junction counts (junctions × cells)
    sj_meta : pd.DataFrame
        Junction metadata with columns: coord.intron, chrom, intron_start, intron_end, strand
    splicing_type : str
        Type of splicing metric: 'donor', 'acceptor', or 'exon_skip'
    gene_of_interest : str, optional
        Filter to specific gene
    min_cell_total : int
        Minimum reads for donor/acceptor usage
    min_total_exon : int
        Minimum reads for exon skipping
    r_code_path : str, optional
        Path to CodeDump.R

    Returns
    -------
    Modality
        Splicing modality with appropriate distribution
    """
    if splicing_type == 'donor':
        counts_array, feature_meta, _ = process_donor_usage(
            sj_counts, sj_meta, gene_of_interest, min_cell_total, r_code_path
        )
        return Modality(
            name='splicing_donor',
            counts=counts_array,
            feature_meta=feature_meta,
            distribution='multinomial',
            cells_axis=1
        )

    elif splicing_type == 'acceptor':
        counts_array, feature_meta, _ = process_acceptor_usage(
            sj_counts, sj_meta, gene_of_interest, min_cell_total, r_code_path
        )
        return Modality(
            name='splicing_acceptor',
            counts=counts_array,
            feature_meta=feature_meta,
            distribution='multinomial',
            cells_axis=1
        )

    elif splicing_type == 'exon_skip':
        inc_counts, total_counts, feature_meta, _ = process_exon_skipping(
            sj_counts, sj_meta, gene_of_interest, min_total_exon, 'min', r_code_path
        )
        return Modality(
            name='splicing_exon_skip',
            counts=inc_counts,
            feature_meta=feature_meta,
            distribution='binomial',
            denominator=total_counts,
            cells_axis=1
        )

    else:
        raise ValueError(f"Unknown splicing_type: {splicing_type}")
