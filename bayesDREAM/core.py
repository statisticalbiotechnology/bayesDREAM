"""
Core bayesDREAM implementation.

This module contains the _BayesDREAMCore base class with delegation to
specialized fitters for technical, cis, and trans modeling.
"""

import os
import subprocess
import warnings
from typing import Dict, Optional, List, Union
import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.transforms import iterated, affine_autoregressive
import pyro.optim as optim
import pyro.infer as infer
import h5py
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
import gc

# Import utility functions and modules
from .utils import (
    set_max_threads,
    find_beta,
    calculate_mu_x_guide,
    Hill_based_positive,
    Hill_based_negative,
    Hill_based_piecewise,
    Polynomial_function,
    cutoff_sigmoid,
    sample_or_use_point,
    check_tensor
)
from .modality import Modality
from .splicing import create_splicing_modality
from .distributions import get_observation_sampler, requires_denominator, is_3d_distribution

# Import fitters
from .fitting import TechnicalFitter, CisFitter, TransFitter
from .io import ModelSaver, ModelLoader

warnings.simplefilter(action="ignore", category=FutureWarning)


class _BayesDREAMCore:
    """
    Internal core class for the three-step Bayesian Dosage Response Effects Across Modalities framework.

    This class provides the foundational infrastructure and delegates to specialized fitters.
    """

    def __init__(
        self,
        meta: pd.DataFrame,
        counts: pd.DataFrame,
        gene_meta: pd.DataFrame = None,
        cis_gene: str = None,
        guide_covariates: list[str] = ["cell_line"],
        guide_covariates_ntc: list[str] = None,
        output_dir: str = "./model_out",
        label: str = None,
        device: str = None,
        random_seed: int = 2402,
        cores: int = 1
    ):
        """
        Initialize the model with the metadata and count matrices.

        Parameters
        ----------
        meta : pd.DataFrame
            Cell metadata DataFrame (includes columns: cell, guide, target, cell_line, sum_factor, etc.)
        counts : pd.DataFrame
            Counts DataFrame (genes as rows, cell barcodes as columns)
        gene_meta : pd.DataFrame, optional
            Gene metadata DataFrame with genes as rows. Required to have at least one identifier column.
            Recommended columns: 'gene' (or use index), 'gene_name', 'gene_id'
            If not provided, will create minimal metadata from counts.index
        cis_gene : str
            The 'X' gene for cis modeling
        guide_covariates : list of str
            List of columns used to construct guide_used for non-NTC guides.
        guide_covariates_ntc : list of str or None
            List of columns used to construct guide_used for NTC guides.
        output_dir : str
            Where to save results
        label : str
            A label to prefix output files
        device : str or None
            "cuda" or "cpu" or None. If None, auto-detect.
        random_seed : int
            Random seed for reproducibility
        cores : int
            Number of CPU cores for Pyro to use
        """
        
        if label is None and cis_gene is not None:
            label = cis_gene
        elif label is None:
            label = ""
            
        # Basic assignments
        self.meta = meta.copy()
        self.counts = counts.copy()
        self.cis_gene = cis_gene
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.label = label

        # Handle gene metadata
        if gene_meta is None:
            # Create minimal gene metadata from counts index
            print("[INFO] No gene_meta provided - creating minimal metadata from counts.index")
            self.gene_meta = pd.DataFrame({
                'gene': counts.index.tolist()
            }, index=counts.index)
        else:
            # Validate and process provided gene_meta
            gene_meta = gene_meta.copy()

            # Check that gene_meta index or has a gene identifier column
            has_gene_col = 'gene' in gene_meta.columns
            has_gene_name = 'gene_name' in gene_meta.columns
            has_gene_id = 'gene_id' in gene_meta.columns

            if not has_gene_col and gene_meta.index.name is None and not has_gene_name and not has_gene_id:
                raise ValueError(
                    "gene_meta must have at least one gene identifier: "
                    "'gene' column, named index, 'gene_name' column, or 'gene_id' column"
                )

            # Ensure we have a 'gene' column for internal use
            if not has_gene_col:
                if gene_meta.index.name is not None:
                    gene_meta['gene'] = gene_meta.index
                    print(f"[INFO] Using gene_meta index ('{gene_meta.index.name}') as 'gene' column")
                elif has_gene_name:
                    gene_meta['gene'] = gene_meta['gene_name']
                    print("[INFO] Using 'gene_name' column as 'gene' identifier")
                elif has_gene_id:
                    gene_meta['gene'] = gene_meta['gene_id']
                    print("[INFO] Using 'gene_id' column as 'gene' identifier")

            # Set index to match counts if needed
            if not gene_meta.index.equals(counts.index):
                # Try to match by gene identifiers
                matched_genes = []
                unmatched_counts = []

                for gene in counts.index:
                    # Try to find this gene in gene_meta by any identifier
                    found = False
                    for col in ['gene', 'gene_name', 'gene_id']:
                        if col in gene_meta.columns:
                            if gene in gene_meta[col].values:
                                matched_genes.append(gene_meta[gene_meta[col] == gene].index[0])
                                found = True
                                break

                    if not found:
                        unmatched_counts.append(gene)

                if unmatched_counts:
                    warnings.warn(
                        f"[WARNING] {len(unmatched_counts)} gene(s) in counts not found in gene_meta. "
                        f"These genes will have minimal metadata. First few: {unmatched_counts[:5]}",
                        UserWarning
                    )
                    # Add missing genes with minimal metadata
                    missing_meta = pd.DataFrame({
                        'gene': unmatched_counts
                    }, index=unmatched_counts)
                    gene_meta = pd.concat([gene_meta, missing_meta])

                # Reindex to match counts
                gene_meta = gene_meta.loc[counts.index] if set(counts.index).issubset(gene_meta.index) else gene_meta

            self.gene_meta = gene_meta

            # Print summary
            meta_cols = [c for c in ['gene', 'gene_name', 'gene_id'] if c in self.gene_meta.columns]
            print(f"[INFO] Gene metadata loaded with {len(self.gene_meta)} genes and columns: {meta_cols}")

        # Ensure guide_covariates and guide_covariates_ntc are always lists
        if guide_covariates is None:
            guide_covariates = []

        if guide_covariates_ntc is None:
            guide_covariates_ntc = []

        # Input checks
        required_cols = {"target", "cell", "sum_factor", "guide"} | set(guide_covariates) | set(guide_covariates_ntc)
        missing_cols = required_cols - set(self.meta.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in meta: {missing_cols}")

        if "ntc" not in self.meta["target"].values:
            raise ValueError("The 'target' column in meta must contain 'ntc'.")

        if not set(self.meta["cell"]).issubset(set(self.counts.columns)):
            raise ValueError("The 'cell' column in meta must correspond 1:1 with the column names of counts.")

        if (self.meta["sum_factor"] <= 0).any():
            raise ValueError("All values in 'sum_factor' column must be strictly greater than 0.")

        # Subset meta and counts to relevant cells
        valid_cells = self.meta[self.meta["target"].isin(["ntc", self.cis_gene])]["cell"].unique()
        if len(valid_cells) < len(self.meta["cell"].unique()):
            warnings.warn(
                f"Subsetting reduced the number of cells in the metadata from {len(self.meta['cell'].unique())} to {len(valid_cells)}. "
                "This may impact downstream analysis.",
                UserWarning
            )
        self.meta = self.meta[self.meta["cell"].isin(valid_cells)].copy()
        self.counts = self.counts[valid_cells].copy()

        # Remove genes with zero total counts
        gene_sums = pd.Series(self.counts.values.sum(axis=1), index=self.counts.index)
        detected_mask = gene_sums > 0
        num_removed = (~detected_mask).sum()

        # Raise error if cis gene is undetected
        if self.cis_gene is not None:
            if not detected_mask.get(self.cis_gene, False):
                raise ValueError(f"[ERROR] The cis gene '{self.cis_gene}' has zero counts after subsetting!")

        # Subset counts to detected genes only
        self.counts = self.counts.loc[detected_mask]

        # Subset gene_meta to match counts (keep only detected genes)
        self.gene_meta = self.gene_meta.loc[self.counts.index]

        # Set trans genes to all detected genes except cis
        self.trans_genes = [g for g in self.counts.index if g != self.cis_gene]

        if num_removed > 0:
            warnings.warn(
                f"[WARNING] {num_removed} gene(s) had zero counts after subsetting and were removed from the counts matrix.",
                UserWarning
            )
        
        # Ensure same order of meta and counts
        self.meta = self.meta.set_index("cell", drop=False).loc[self.counts.columns]
        
        # Construct guide_used column
        self.meta["guide_used"] = self.meta.apply(
            lambda row: f"{row['guide']}_{'_'.join(str(row[cov]) for cov in (guide_covariates_ntc if row['target'] == 'ntc' else set(guide_covariates_ntc + guide_covariates)))}",
            axis=1
        )
        
        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Set random seeds & threads
        pyro.set_rng_seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        set_max_threads(cores)

        # one-hot encode things
        self.meta['guide_code'] = pd.Categorical(self.meta['guide_used']).codes

        # Bookkeeping for results
        self.alpha_x_prefit = None    # from step1
        self.alpha_x_type = None    # from step1
        self.alpha_y_prefit = None    # from step1
        self.alpha_y_type = None    # from step1
        self.trace_cellline = None    # from step1
        self.trace_x = None          # from step2
        self.trace_y = None          # from step3

        print(f"[INIT] bayesDREAM core: label={self.label}, device={self.device}")


    # Initialize fitters
        self._technical_fitter = TechnicalFitter(self)
        self._cis_fitter = CisFitter(self)
        self._trans_fitter = TransFitter(self)
        self._saver = ModelSaver(self)
        self._loader = ModelLoader(self)

    def cis_init_loc_fn(
        self,
        mu_init: torch.Tensor,
        sigma_init: torch.Tensor,
        beta_o_init: float,
        o_x_init: float,
        sigma_eff_init: float,
        sigma_eff_alpha_init: float,
        sigma_eff_beta_init: float,
        alpha_x_init: torch.Tensor = None,
    ):
        """
        Returns a function that uses custom data-driven initial values for certain sites:
            - 'mu' initialized to `mu_init`
            - 'sigma' initialized to `sigma_init`
            - 'alpha_x' (if present) initialized to `alpha_x_init`
            - 'beta_o' initialized to `beta_o_init`
            - 'o_x' initialized to `o_x_init`
            - 'sigma_eff' (the gamma shape or scale for per-guide sigmas) initialized to `sigma_eff_init`
            - 'sigma_eff_alpha' (the gamma shape or scale for per-guide sigmas) initialized to `sigma_eff_alpha_init`
            - 'sigma_eff_beta' (the gamma shape or scale for per-guide sigmas) initialized to `sigma_eff_beta_init`
        Fallback to `init_to_median` for all other sites.
        """

        def to_tensor(value, device):
            """Helper function to ensure value is a tensor."""
            if isinstance(value, torch.Tensor):
                return value.detach().clone().to(device)
            return torch.tensor(value, dtype=torch.float32, device=device)
    
        def _init_fun(site):
            name = site["name"]
            device = self.device
    
            if name == "mu":
                return to_tensor(mu_init, device)
    
            elif name == "sigma":
                return to_tensor(sigma_init, device)
    
            elif name == "alpha_x" and alpha_x_init is not None:
                return to_tensor(alpha_x_init, device)
    
            elif name == "beta_o":
                return to_tensor(beta_o_init, device)
    
            elif name == "o_x":
                return to_tensor(o_x_init, device)
    
            elif name == "sigma_eff_alpha":
                return to_tensor(sigma_eff_alpha_init, device)
    
            elif name == "sigma_eff_beta":
                return to_tensor(sigma_eff_beta_init, device)
    
            elif name == "sigma_eff":
                shape = site["fn"].sample().shape  # Get shape from distribution
                return to_tensor(sigma_eff_init, device).expand(shape)
    
            # Otherwise, fall back to a default
            return pyro.infer.autoguide.initialization.init_to_sample(site)
    
        return _init_fun

    def set_alpha_x(
        self,
        alpha_x: float, # expected to be C x 1 point estimate or S x C x 1 posterior
        is_posterior: bool,
        covariates: list[str] = None # ["cell_line"] NOT empty. The point is to fit to the covariates. Lane is typically not included as this tends to be corrected by sum factor adjustment alone
    ):
        """
        Sets alpha_x either as a point estimate (float/tensor) or as a Pyro posterior.
        """
        if covariates:
            if "technical_group_code" in self.meta.columns:
                warnings.warn("technical_group already set. Overwriting.")
            self.meta["technical_group_code"] = self.meta.groupby(covariates).ngroup()
        elif not "technical_group_code" in self.meta.columns:
            raise ValueError(f"No column 'technical_group_code' found in meta, and no covariates provided.")
        else:
            warnings.warn("technical_group previously set. Assuming alpha_x corresponds.")
        self.alpha_x_prefit = sample_or_use_point("alpha_x_posterior", alpha_x, self.device)
        
        if is_posterior:
            if not (self.alpha_x_prefit.ndim == 2 or (self.alpha_x_prefit.ndim == 3 and self.alpha_x_prefit.shape[2] == 1)):
                raise ValueError(
                    f"when it is a posterior, alpha_x_prefit is expected to be of shape S x C or S x C x 1, "
                    f"but got {self.alpha_x_prefit.shape}."
                )
            self.alpha_x_type = 'posterior'
        else:
            if not (self.alpha_x_prefit.ndim == 0 or  # Scalar case
                    (self.alpha_x_prefit.ndim == 2 and self.alpha_x_prefit.shape[1] == 1)):  # C x 1 case
                raise ValueError(
                    f"when it is a point estimate, alpha_x_prefit is expected to be a scalar or of shape C x 1, "
                    f"but got {self.alpha_x_prefit.shape}."
                )
            self.alpha_x_type = 'point'

    def set_alpha_y(
        self,
        alpha_y,
        is_posterior: bool,
        covariates: list[str] = None # ["cell_line"] NOT empty. The point is to fit to the covariates. Lane is typically not included as this tends to be corrected by sum factor adjustment alone
    ):
        if covariates:
            if "technical_group_code" in self.meta.columns:
                warnings.warn("technical_group already set. Overwriting.")
            self.meta["technical_group_code"] = self.meta.groupby(covariates).ngroup()
        elif not "technical_group_code" in self.meta.columns:
            raise ValueError(f"No column 'technical_group_code' found in meta, and no covariates provided.")
        else:
            warnings.warn("technical_group previously set. Assuming alpha_xy corresponds.")

        # Determine T (number of response variables)
        if self.cis_gene is not None:
            T = self.counts.drop([self.cis_gene]).shape[0]
        else:
            T = self.counts.shape[0]
        C = self.meta["technical_group_code"].nunique() - 1
    
        # Convert alpha_y to tensor
        alpha_y = sample_or_use_point("alpha_y_posterior", alpha_y, self.device)
    
        if is_posterior:
            if not (alpha_y.ndim == 3 and alpha_y.shape[1] == C and alpha_y.shape[2] == T):
                raise ValueError(
                    f"When it is a posterior, alpha_y is expected to have shape S x C-1 x T, but got {alpha_y.shape}."
                )
            self.alpha_y_type = 'posterior'
        else:
            if not (alpha_y.ndim == 2 and alpha_y.shape[0] == C and alpha_y.shape[1] == T):
                raise ValueError(
                    f"When it is a point estimate, alpha_y must have shape C-1 x T, but got {alpha_y.shape}."
                )
            self.alpha_y_type = 'point'
    
        self.alpha_y_prefit = alpha_y  # Store validated tensor

    def set_o_x_grouped(
        self,
        o_x,  # Expected to be C x 1 point estimate or S x C x 1 posterior
        is_posterior: bool,
        covariates: list[str] = None,  # ["cell_line"] NOT empty if using
    ):
        """
        Sets o_x either as a point estimate (float/tensor) or as a Pyro posterior.
        """
        if covariates:
            if "technical_group_code" in self.meta.columns:
                warnings.warn("technical_group already set. Overwriting.")
            self.meta["technical_group_code"] = self.meta.groupby(covariates).ngroup()
        elif "technical_group_code" not in self.meta.columns:
            raise ValueError("No column 'technical_group_code' found in meta, and no covariates provided.")
        else:
            warnings.warn("technical_group previously set. Assuming o_x corresponds.")
    
        self.o_x_prefit = sample_or_use_point("o_x_posterior", o_x, self.device)
    
        if is_posterior:
            if not (self.o_x_prefit.ndim == 2 or
                    (self.o_x_prefit.ndim == 3 and self.o_x_prefit.shape[2] == 1)):
                raise ValueError(
                    f"When it is a posterior, o_x_prefit must be S x C or S x C x 1, "
                    f"but got {self.o_x_prefit.shape}."
                )
            self.o_x_type = 'posterior'
        else:
            if not (self.o_x_prefit.ndim == 0 or
                    (self.o_x_prefit.ndim == 2 and self.o_x_prefit.shape[1] == 1)):  # C x 1 case
                raise ValueError(
                    f"When it is a point estimate, o_x_prefit must be a scalar or of shape C x 1, "
                    f"but got {self.o_x_prefit.shape}."
                )
            self.o_x_type = 'point'


    def set_o_x(
        self,
        o_x,
        is_posterior: bool
    ):
        """
        Sets o_x either as a point estimate (float/tensor) or as a posterior.
        
        - If posterior: o_x should have shape S, S x 1, or S x 1 x 1.
        - If point estimate: o_x should be a scalar or [scalar] (1-element tensor).
        """
        self.o_x = sample_or_use_point("o_x_posterior", o_x, self.device)
    
        if is_posterior:
            if not (
                (self.o_x.ndim == 1) or  # S
                (self.o_x.ndim == 2 and self.o_x.shape[1] == 1) or  # S x 1
                (self.o_x.ndim == 3 and self.o_x.shape[1] == 1 and self.o_x.shape[2] == 1)  # S x 1 x 1
            ):
                raise ValueError(
                    f"When it is a posterior, o_x must have shape S, S x 1, or S x 1 x 1, but got {self.o_x.shape}."
                )
            self.o_x_type = "posterior"
        else:
            if not (self.o_x.ndim == 0 or (self.o_x.ndim == 1 and self.o_x.numel() == 1)):
                raise ValueError(
                    f"When it is a point estimate, o_x must be a scalar or a single-element tensor, but got {self.o_x.shape}."
                )
            self.o_x_type = "point"

    def set_x_true(
        self,
        x_true,
        is_posterior: bool
    ):

        """
        Sets x_true either as a point estimate or as a posterior.
        
        - If posterior: x_true should have shape S x N or S x 1 x N.
        - If point estimate: x_true should have shape N.
        """
        # Determine N (number of observations)
        N = self.counts.shape[1]
    
        # Convert x_true to tensor
        x_true = sample_or_use_point("x_true_posterior", x_true, self.device)
    
        if is_posterior:
            if not ((x_true.ndim == 2 and x_true.shape[1] == N) or
                    (x_true.ndim == 3 and x_true.shape[1] == 1 and x_true.shape[2] == N)):
                raise ValueError(
                    f"When it is a posterior, x_true must have shape S x N or S x 1 x N, but got {x_true.shape}."
                )
            self.x_true_type = "posterior"
        else:
            if not (x_true.ndim == 1 and x_true.shape[0] == N):
                raise ValueError(
                    f"When it is a point estimate, x_true must have shape N ({N},), but got {x_true.shape}."
                )
            self.x_true_type = "point"
    
        self.x_true = x_true  # Store validated tensor


    def adjust_ntc_sum_factor(
        self,
        sum_factor_col_old: str = "sum_factor",
        sum_factor_col_adj: str = "sum_factor_adj",
        covariates: list[str] = None # ["lane", "cell_line"] or could be empty
    ):
        """
        Step 1 of sum factor adjustment: Normalize guides to NTC controls.

        Use BEFORE fit_cis() to account for guide-level technical variation.
        Computes adjustment factor = mean_ntc_sum_factor / mean_guide_sum_factor
        within covariate groups (e.g., cell_line, lane).

        Typical workflow:
            1. adjust_ntc_sum_factor() -> creates 'sum_factor_adj'
            2. fit_cis(sum_factor_col='sum_factor_adj')
            3. refit_sumfactor() -> creates 'sum_factor_refit'
            4. fit_trans(sum_factor_col='sum_factor_refit')

        Parameters
        ----------
        sum_factor_col_old : str
            Name of existing sum factor column in meta (default: 'sum_factor')
        sum_factor_col_adj : str
            Name for adjusted sum factor column to create (default: 'sum_factor_adj')
        covariates : list of str, optional
            Columns to group by for adjustment (e.g., ['cell_line', 'lane']).
            If None or empty, uses global mean across all cells.

        Returns
        -------
        None
            Creates new column sum_factor_col_adj in self.meta

        Notes
        -----
        For each guide within each covariate group:
        - Compute mean NTC sum factor: mean_ntc
        - Compute mean guide sum factor: mean_guide
        - Adjustment factor = mean_ntc / mean_guide
        - Adjusted sum factor = sum_factor * adjustment_factor
        """
        meta_out = self.meta.copy()
        meta_out["original_index"] = np.arange(len(meta_out))

        if sum_factor_col_old not in meta_out.columns:
            raise ValueError(f"No column '{sum_factor_col_old}' found in meta. Provide a precomputed sum_factor.")

        # Drop existing adjustment_factor column if it exists (prevents merge conflicts)
        if "adjustment_factor" in meta_out.columns:
            meta_out = meta_out.drop(columns=["adjustment_factor"])

        # Make sure all covariates are actually in meta_out
        if covariates:
            missing_cols = [c for c in covariates if c not in meta_out.columns]
            if missing_cols:
                raise ValueError(
                    f"Missing covariate columns: {missing_cols}. "
                    f"Available columns are: {list(meta_out.columns)}"
            )
        
        if not covariates:
            # (1) Mean sum_factor among NTC rows, grouped by covariates (e.g. lane, cell_line)
            mean_ntc_value = meta_out.loc[meta_out[target_col] == gene_ntc, sum_factor_col_old].mean()
    
            # (2) Mean sum_factor among *all* guides, grouped by covariates + [guide_col]
            df_guide = (
                meta_out.groupby(["guide_used"])[sum_factor_col_old]
                .mean()
                .reset_index(name="mean_SumFacs_guide")
            )
    
            # (3) Merge them and compute ratio = mean_NTC / mean_guide
            df_guide["adjustment_factor"] = (
                mean_ntc_value / (df_guide["mean_SumFacs_guide"])
            )
    
            # (4) Merge that ratio back onto meta_out
            meta_out = pd.merge(
                meta_out,
                df_guide[["guide_used", "adjustment_factor"]],
                on="guide_used",
                how="left"
            )

        else:
            # (1) Mean sum_factor among NTC rows, grouped by covariates (e.g. lane, cell_line)
            df_ntc = (
                meta_out.loc[meta_out["target"] == "ntc"]
                .groupby(covariates)[sum_factor_col_old]
                .mean()
                .reset_index(name="mean_SumFacs_ntc")
            )
    
            # (2) Mean sum_factor among *all* guides, grouped by covariates + [guide_col]
            df_guide = (
                meta_out.groupby(covariates + ["guide_used"])[sum_factor_col_old]
                .mean()
                .reset_index(name="mean_SumFacs_guide")
            )
    
            # (3) Merge them and compute ratio = mean_NTC / mean_guide
            merged = pd.merge(df_guide, df_ntc, on=covariates, how="left")
            merged["adjustment_factor"] = merged["mean_SumFacs_ntc"] / merged["mean_SumFacs_guide"]
    
            # (4) Merge that ratio back onto meta_out
            merge_cols = covariates + ["guide_used", "adjustment_factor"]
            meta_out = pd.merge(meta_out, merged[merge_cols], on=covariates + ["guide_used"], how="left")
            
        # (5) Multiply original sum_factor by ratio
        meta_out[sum_factor_col_adj] = meta_out[sum_factor_col_old] * meta_out["adjustment_factor"]

        meta_out.sort_values("original_index", inplace=True)
        meta_out.drop(columns="original_index", inplace=True)
        self.meta = meta_out
        print(f"[INFO] Created '{sum_factor_col_adj}' in meta with NTC-based guide-level adjustment.")

    def permute_genes(
        self,
        genes2permute: list[str] = None,
        covariates: list[str] = ["cell_line", "lane"],
        sum_factor_col: str = 'sum_factor_adj',
        permute_ntc_x: bool = True
    ):
        """
        Permute specified genes within technical covariates while ensuring consistency with NTC cells.
        After permutation, the sum factors should be adjusted.

        Parameters
        ----------
        genes2permute : list of str
            List of gene names to permute. If 'All', all genes except the cis gene are permuted.
        covariates : list of str
            Covariates used to group cells for permutation (e.g., ['cell_line', 'lane']).
        """

        if sum_factor_col not in self.meta.columns:
            raise ValueError(f"No column '{sum_factor_col}' found in meta. Provide a precomputed sum_factor.")
            
        print("Running gene permutation...")

        # If genes2permute is 'All', include all genes except the cis_gene
        if genes2permute == 'All' or genes2permute == ['All']:
            genes2permute = list(self.counts.index.values[self.counts.index.values != self.cis_gene])

        if isinstance(genes2permute, str):
            genes2permute = [genes2permute]

        meta_sub = self.meta.copy()
        counts_sub = self.counts.copy()

        for gene in genes2permute:
            if gene in counts_sub.index:  # Ensure gene exists in the count matrix
                for cov_values, group in meta_sub.groupby(covariates):
                    mycells = group.loc[group["target"] != "ntc", "cell"]
                    my_ntc_cells = group.loc[group["target"] == "ntc", "cell"]
                    
                    if len(mycells) > 0 and len(my_ntc_cells) > 0:
                        sampled_values = np.random.choice(
                            counts_sub.loc[gene, my_ntc_cells] /
                            meta_sub.set_index("cell").loc[my_ntc_cells, sum_factor_col],
                            size=len(mycells),
                            replace=True
                        ) * meta_sub.set_index("cell").loc[mycells, sum_factor_col]
                        counts_sub.loc[gene, mycells] = np.round(sampled_values)
        if permute_ntc_x:
            for cov_values, group in meta_sub.groupby(covariates):
                my_ntc_cells = group.loc[group["target"] == "ntc", "cell"]
                
                if len(my_ntc_cells) > 0:
                    # Set up normalized cis-gene counts for these NTC cells
                    ntc_sum_factor = meta_sub.set_index("cell").loc[my_ntc_cells, sum_factor_col].values
                    ntc_expr = counts_sub.loc[self.cis_gene, my_ntc_cells].values
                    ntc_expr_norm = ntc_expr / ntc_sum_factor
        
                    # Sample indices with replacement
                    permuted_indices = np.random.choice(len(my_ntc_cells), size=len(my_ntc_cells), replace=True)
        
                    # Permute counts
                    new_counts = ntc_expr_norm[permuted_indices] * ntc_sum_factor
                    counts_sub.loc[self.cis_gene, my_ntc_cells] = np.round(new_counts)
        
                    # Permute x_true in the same order
                    ntc_idx = meta_sub.index[meta_sub["cell"].isin(my_ntc_cells)].tolist()
                    assert len(ntc_idx) == len(my_ntc_cells)  # Sanity check
        
                    # Apply permutation to x_true
                    if isinstance(self.x_true, torch.Tensor):
                        x_true_np = self.x_true.detach().cpu().numpy()
                    else:
                        x_true_np = np.array(self.x_true)
        
                    x_true_ntc = x_true_np[ntc_idx]
                    x_true_np[ntc_idx] = x_true_ntc[permuted_indices]
        
                    # Assign back
                    self.x_true = torch.tensor(x_true_np, dtype=self.x_true.dtype, device=self.x_true.device)

        self.counts = counts_sub



    # ========================================================================
    # Delegation methods to fitters
    # ========================================================================

    def _model_technical(self, *args, **kwargs):
        """Delegate to TechnicalFitter."""
        return self._technical_fitter._model_technical(*args, **kwargs)

    def set_technical_groups(self, *args, **kwargs):
        """Delegate to TechnicalFitter."""
        return self._technical_fitter.set_technical_groups(*args, **kwargs)

    def fit_technical(self, *args, **kwargs):
        """Delegate to TechnicalFitter."""
        return self._technical_fitter.fit_technical(*args, **kwargs)

    def _model_x(self, *args, **kwargs):
        """Delegate to CisFitter."""
        return self._cis_fitter._model_x(*args, **kwargs)

    def cis_init_loc_fn(self, *args, **kwargs):
        """Delegate to CisFitter."""
        return self._cis_fitter.cis_init_loc_fn(*args, **kwargs)

    def fit_cis(self, *args, **kwargs):
        """Delegate to CisFitter."""
        return self._cis_fitter.fit_cis(*args, **kwargs)

    def refit_sumfactor(self, *args, **kwargs):
        """Delegate to CisFitter."""
        return self._cis_fitter.refit_sumfactor(*args, **kwargs)

    def _model_y(self, *args, **kwargs):
        """Delegate to TransFitter."""
        return self._trans_fitter._model_y(*args, **kwargs)

    def fit_trans(self, *args, **kwargs):
        """Delegate to TransFitter."""
        return self._trans_fitter.fit_trans(*args, **kwargs)

    def save_technical_fit(self, *args, **kwargs):
        """Delegate to ModelSaver."""
        return self._saver.save_technical_fit(*args, **kwargs)

    def save_cis_fit(self, *args, **kwargs):
        """Delegate to ModelSaver."""
        return self._saver.save_cis_fit(*args, **kwargs)

    def save_trans_fit(self, *args, **kwargs):
        """Delegate to ModelSaver."""
        return self._saver.save_trans_fit(*args, **kwargs)

    def load_technical_fit(self, *args, **kwargs):
        """Delegate to ModelLoader."""
        return self._loader.load_technical_fit(*args, **kwargs)

    def load_cis_fit(self, *args, **kwargs):
        """Delegate to ModelLoader."""
        return self._loader.load_cis_fit(*args, **kwargs)

    def load_trans_fit(self, *args, **kwargs):
        """Delegate to ModelLoader."""
        return self._loader.load_trans_fit(*args, **kwargs)
