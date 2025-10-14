"""
bayesDREAM: Bayesian Dosage Response Effects Across Modalities

Core bayesDREAM class implementing three-step Bayesian CRISPR perturbation modeling:
1. fit_technical() - Model technical variation in non-targeting controls
2. fit_cis() - Model direct effects on targeted genes
3. fit_trans() - Model downstream effects as dose-response functions

For multi-modal support, use MultiModalBayesDREAM from multimodal.py.
"""

import os
import subprocess
import warnings
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

# Import utility functions
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

warnings.simplefilter(action="ignore", category=FutureWarning)


########################################
# bayesDREAM Class
########################################

class bayesDREAM:
    """
    An object-oriented wrapper for the three-step Bayesian Doseage Responce Effects Across Modalities model:
    
    1) Optional cell-line prefit (modeling alpha_y for NTC),
    2) Fitting cis effects (model_x),
    3) Fitting trans effects (model_y).
    """

    def __init__(
        self,
        meta: pd.DataFrame,
        counts: pd.DataFrame,
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
            Metadata DataFrame (includes columns: L_cell_barcode, guide, cell_line, etc.)
        counts : pd.DataFrame
            Counts DataFrame (genes as rows, cell barcodes as columns)
        guide_covariates : list of str
            List of columns used to construct guide_used for non-NTC guides.
        guide_covariates_ntc : list of str or None
            List of columns used to construct guide_used for NTC guides.
        cis_gene : str
            The 'X' gene for cis modeling
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

        print(f"[INIT] CRISPRBayesianModel: label={self.label}, device={self.device}")

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

    def _model_technical(
        self,
        N,
        T,
        C,
        groups_ntc_tensor,
        y_obs_ntc_tensor,
        sum_factor_ntc_tensor,
        beta_o_alpha_tensor,
        beta_o_beta_tensor,
        alpha_alpha_mu_tensor,
        mu_x_mean_tensor,
        mu_x_sd_tensor,
        epsilon_tensor,
        distribution='negbinom',
        denominator_ntc_tensor=None,
        K=None,
        D=None,
    ):
    
        # Global parameters (no plate needed here)
    
        # Define the c_plate ONCE and store it
        c_plate = pyro.plate("c_plate", C - 1, dim=-2)  # <-- Store it as a variableit
        cfull_plate = pyro.plate("cfull_plate", C, dim=-2)  # <-- Store it as a variable
    
        # Sample alpha_alpha and alpha_mu using the stored c_plate
        #with cfull_plate:
        #    beta_o = pyro.sample("beta_o", dist.Gamma(beta_o_alpha_tensor, beta_o_beta_tensor))
        beta_o = pyro.sample("beta_o", dist.Gamma(beta_o_alpha_tensor, beta_o_beta_tensor))
    
        # Sample alpha_alpha and alpha_mu using the stored c_plate
        #with c_plate:
        #    #alpha_log2mu = pyro.sample("alpha_log2mu", dist.Normal(0, 1)) 
        #    alpha_log2mu = pyro.sample("alpha_log2mu", dist.StudentT(df=1, loc=0., scale=5.)) 
        #    #alpha_sigma  = pyro.sample("alpha_sigma", dist.HalfNormal(4.0)) 
        #    alpha_sigma  = pyro.sample("alpha_sigma", dist.HalfCauchy(10.)) 
    
        # One single trans_plate for all T-dependent variables
        with pyro.plate("trans_plate", T, dim=-1):
    
            # Reuse the SAME c_plate inside trans_plate
            with c_plate:  # <-- This ensures the same plate is used
                #alpha_sigma  = pyro.sample("alpha_sigma", dist.HalfNormal(4.0)) 
                #log2_alpha_y = pyro.sample("log2_alpha_y", dist.Normal(
                log2_alpha_y = pyro.sample("log2_alpha_y", dist.StudentT(
                    df=3,
                    #loc=alpha_log2mu,
                    loc=0.0,
                    #scale=alpha_sigma
                    scale=20.0
                ))
                alpha_y = pyro.deterministic("alpha_y", 2 ** log2_alpha_y)
                #print("alpha_y shape:", alpha_y.shape)
            
            o_y = pyro.sample("o_y", dist.Exponential(beta_o))
            phi_y = 1 / (o_y**2)

        # Add baseline cell line of all ones
        alpha_y = alpha_y.to(self.device)
        if alpha_y.ndim == 2:
            alpha_y_full = torch.cat([torch.ones(1, T, device=self.device), alpha_y], dim=0)  # shape => (C, T)
        elif alpha_y.ndim == 3:
            alpha_y_full = torch.cat([torch.ones(alpha_y.size(0), 1, T, device=self.device), alpha_y], dim=1)  # shape => (num_samples, C, T)

        alpha_y_used = alpha_y_full[..., groups_ntc_tensor, :]  # [N, T]
        #phi_y_used = phi_y[..., groups_ntc_tensor, :] # shape => [N, T]
        phi_y_used = phi_y.unsqueeze(-2)
    
        # Sample mu_ntc (baseline expression for each feature)
        with pyro.plate("feature_plate_technical", T, dim=-1):
            mu_ntc = pyro.sample(
                "mu_ntc",
                dist.Gamma((mu_x_mean_tensor**2) / (mu_x_sd_tensor**2), mu_x_mean_tensor / (mu_x_sd_tensor**2))
            )  # [T]

        # Compute mu_y from mu_ntc and alpha_y
        mu_y = alpha_y_used * mu_ntc  # [N, T]

        # Call distribution-specific observation sampler
        from .distributions import get_observation_sampler
        observation_sampler = get_observation_sampler(distribution, 'trans')

        if distribution == 'negbinom':
            # For negbinom, we pass mu_y and sum_factor separately
            # Note: alpha_y already applied in mu_y, so we pass alpha_y_full=None
            observation_sampler(
                y_obs_tensor=y_obs_ntc_tensor,
                mu_y=mu_y,
                phi_y_used=phi_y_used,
                alpha_y_full=None,  # Already applied in mu_y
                groups_tensor=None,
                sum_factor_tensor=sum_factor_ntc_tensor,
                N=N,
                T=T,
                C=C
            )
        elif distribution == 'normal':
            sigma_y = 1.0 / torch.sqrt(phi_y)
            observation_sampler(
                y_obs_tensor=y_obs_ntc_tensor,
                mu_y=mu_y,
                sigma_y=sigma_y,
                alpha_y_full=None,  # Already applied in mu_y
                groups_tensor=None,
                N=N,
                T=T,
                C=C
            )
        elif distribution == 'binomial':
            observation_sampler(
                y_obs_tensor=y_obs_ntc_tensor,
                denominator_tensor=denominator_ntc_tensor,
                mu_y=mu_y,
                alpha_y_full=None,  # Already applied in mu_y
                groups_tensor=None,
                N=N,
                T=T,
                C=C
            )
        elif distribution == 'multinomial':
            observation_sampler(
                y_obs_tensor=y_obs_ntc_tensor,
                mu_y=mu_y,
                N=N,
                T=T,
                K=K
            )
        elif distribution == 'mvnormal':
            # For mvnormal, need covariance matrix
            # For now, use diagonal covariance from phi_y
            cov_y = torch.diag_embed(1.0 / phi_y.unsqueeze(0).expand(T, D))  # [T, D, D]
            observation_sampler(
                y_obs_tensor=y_obs_ntc_tensor,
                mu_y=mu_y,
                cov_y=cov_y,
                alpha_y_full=None,  # Already applied in mu_y
                groups_tensor=None,
                N=N,
                T=T,
                D=D,
                C=C
            )
        else:
            raise ValueError(f"Unknown distribution: {distribution}")



    ########################################################
    # Step 1: Optional Prefit for alpha_y (NTC only) 
    ########################################################
    def fit_technical(
        self,
        covariates: list[str], # ["cell_line"] NOT empty. The point is to fit to the covariates. Lane is typically not included as this tends to be corrected by sum factor adjustment alone
        sum_factor_col: str = None,
        lr: float = 1e-3,
        niters: int = 50000,
        nsamples: int = 1000,
        alpha_ewma: float = 0.05,
        tolerance: float = 1e-4, # recommended to keep based on cell2location
        beta_o_beta: float = 3, # recommended to keep based on cell2location
        beta_o_alpha: float = 9, # recommended to keep based on cell2location
        alpha_alpha_mu: float = 5.8,
        epsilon: float = 1e-6,
        minibatch_size: int = None,
        distribution: str = 'negbinom',
        denominator: np.ndarray = None,
        **kwargs
    ):
        """
        Prefits gene-level "technical" variables (alpha_y) using only NTC samples,
        grouped by arbitrary covariates (e.g. ["cell_line"]).

        If you skip this method, you'd provide alpha_y_fixed manually in the next step.

        Parameters
        ----------
        distribution : str, default='negbinom'
            Distribution type: 'negbinom', 'multinomial', 'binomial', 'normal', 'mvnormal'
        sum_factor_col : str, optional
            Column name for size factors (required for negbinom, ignored otherwise)
        denominator : np.ndarray, optional
            Denominator array for binomial distribution (shape: [n_features, n_cells])

        Notes
        -----
        BREAKING: sum_factor_col now defaults to None (optional, only for negbinom)
        """

        # Validate distribution-specific requirements
        from .distributions import requires_sum_factor, requires_denominator

        if requires_sum_factor(distribution) and sum_factor_col is None:
            raise ValueError(f"Distribution '{distribution}' requires sum_factor_col parameter")

        if requires_denominator(distribution) and denominator is None:
            raise ValueError(f"Distribution '{distribution}' requires denominator parameter")
        
        # Check covariates exist
        if not covariates:
            raise ValueError(
                "The 'covariates' argument must not be empty. "
                "Please provide at least one covariate column to group by."
            )
        missing_cols = [c for c in covariates if c not in self.meta.columns]
        if missing_cols:
            raise ValueError(f"Missing these columns in meta: {missing_cols}")

        print("Running prefit_cellline...")

        self.meta["technical_group_code"] = self.meta.groupby(covariates).ngroup()

        # Subset data to NTC only
        meta_ntc = self.meta[self.meta["target"] == "ntc"].copy()
        counts_ntc = self.counts.loc[:, meta_ntc["cell"]].copy()

        # Remove genes with zero counts in NTC
        gene_sums_ntc = pd.Series(counts_ntc.values.sum(axis=1), index=counts_ntc.index)
        detected_mask_ntc = gene_sums_ntc > 0
        num_removed = (~detected_mask_ntc).sum()
        
        # Check cis gene detection
        if self.cis_gene is not None:
            if not detected_mask_ntc.get(self.cis_gene, False):
                raise ValueError(f"[ERROR] The cis gene '{self.cis_gene}' has zero counts in NTC cells!")
        
        # Subset counts_ntc and self.counts to detected genes only
        counts_ntc = counts_ntc.loc[detected_mask_ntc]
        self.counts = self.counts.loc[detected_mask_ntc]
        
        # Recompute trans genes
        self.trans_genes = [g for g in self.counts.index if g != self.cis_gene]
        
        if num_removed > 0:
            num_trans_removed = num_removed - int(not detected_mask_ntc[self.cis_gene])
            warnings.warn(
                f"[WARNING] {num_trans_removed} trans gene(s) had zero counts in NTC and were removed.",
                UserWarning
            )


        
        # Prepare inputs to pyro
        N = meta_ntc.shape[0]
        T = counts_ntc.shape[0]
        C = meta_ntc['technical_group_code'].nunique()
        y_obs_ntc = counts_ntc.values.T
        groups_ntc = meta_ntc['technical_group_code'].values
        guides = meta_ntc['guide_code'].values

        # Calculate priors
        # Handle sum factors (only for distributions that need them)
        if sum_factor_col is not None:
            y_obs_ntc_factored = y_obs_ntc / meta_ntc[sum_factor_col].values.reshape(-1, 1)
        else:
            y_obs_ntc_factored = y_obs_ntc

        baseline_mask = (groups_ntc == 0)
        mu_x_mean = np.mean(y_obs_ntc_factored[baseline_mask, :],axis=0) + epsilon
        guide_means = np.array([np.mean(y_obs_ntc_factored[guides == g], axis=0) for g in np.unique(guides)])
        mu_x_sd = np.std(guide_means, axis=0) + epsilon

        # Convert to tensors
        beta_o_beta_tensor = torch.tensor(beta_o_beta, dtype=torch.float32, device=self.device)
        beta_o_alpha_tensor = torch.tensor(beta_o_alpha, dtype=torch.float32, device=self.device)
        alpha_alpha_mu_tensor = torch.tensor(alpha_alpha_mu, dtype=torch.float32, device=self.device)

        # Handle sum factors
        if sum_factor_col is not None:
            sum_factor_ntc_tensor = torch.tensor(meta_ntc[sum_factor_col].values, dtype=torch.float32, device=self.device)
        else:
            sum_factor_ntc_tensor = torch.ones(N, dtype=torch.float32, device=self.device)

        groups_ntc_tensor = torch.tensor(groups_ntc, dtype=torch.long, device=self.device)
        y_obs_ntc_tensor = torch.tensor(y_obs_ntc, dtype=torch.float32, device=self.device)
        mu_x_mean_tensor = torch.tensor(mu_x_mean, dtype=torch.float32, device=self.device)
        mu_x_sd_tensor = torch.tensor(mu_x_sd, dtype=torch.float32, device=self.device)
        epsilon_tensor = torch.tensor(epsilon, dtype=torch.float32, device=self.device)

        # Handle denominator (for binomial)
        denominator_ntc_tensor = None
        if denominator is not None:
            denominator_ntc = denominator[:, meta_ntc.index].T  # Subset to NTC cells, transpose to [N, T]
            denominator_ntc_tensor = torch.tensor(denominator_ntc, dtype=torch.float32, device=self.device)

        # Detect data dimensions (for multinomial and mvnormal)
        from .distributions import is_3d_distribution
        K = None
        D = None
        if is_3d_distribution(distribution):
            if y_obs_ntc.ndim == 3:
                if distribution == 'multinomial':
                    K = y_obs_ntc.shape[2]
                elif distribution == 'mvnormal':
                    D = y_obs_ntc.shape[2]

        def init_loc_fn(site):
            if site["name"] == "log2_alpha_y":
                # Compute per-group log2 ratios vs. baseline group 0
                group_codes = meta_ntc["technical_group_code"].values
                group_labels = np.unique(group_codes)
                group_labels = group_labels[group_labels != 0]  # C-1 groups vs. baseline        
                init_values = []
                for g in group_labels:
                    group_mean = np.mean(y_obs_ntc_factored[group_codes == g], axis=0)
                    baseline_mean = np.mean(y_obs_ntc_factored[group_codes == 0], axis=0)
                    log2_ratio = np.log2(group_mean / baseline_mean)
                    init_values.append(log2_ratio)
        
                init_tensor = torch.tensor(np.stack(init_values), dtype=torch.float32, device=self.device)  # shape: [C-1, T]
                return init_tensor
            else:
                return pyro.infer.autoguide.initialization.init_to_median(site)
        
        # Use in guide
        guide_cellline = pyro.infer.autoguide.AutoIAFNormal(self._model_technical, init_loc_fn=init_loc_fn)
        optimizer = pyro.optim.Adam({"lr": lr})
        svi = pyro.infer.SVI(self._model_technical, guide_cellline, optimizer, loss=pyro.infer.Trace_ELBO())
        
        losses = []
        smoothed_loss = None
        for step in range(niters):
            loss = svi.step(
                N,
                T,
                C,
                groups_ntc_tensor,
                y_obs_ntc_tensor,
                sum_factor_ntc_tensor,
                beta_o_alpha_tensor,
                beta_o_beta_tensor,
                alpha_alpha_mu_tensor,
                mu_x_mean_tensor,
                mu_x_sd_tensor,
                epsilon_tensor,
                distribution,
                denominator_ntc_tensor,
                K,
                D,
            )
            losses.append(loss)
            if step % 1000 == 0:
                print(f"Step {step} : loss = {loss:.5e}, device: {mu_x_mean_tensor.device}")
            if smoothed_loss is None:
                smoothed_loss = loss
            else:
                if abs(alpha_ewma * (loss - smoothed_loss)) < tolerance:
                    print(f"Converged at step {step}! Loss = {loss:.5e}")
                    break
                smoothed_loss = alpha_ewma * loss + (1 - alpha_ewma) * smoothed_loss


        # Move to CPU if using too much GPU memory for Predictive
        run_on_cpu = self.device.type != "cpu"

        if run_on_cpu:
            print("[INFO] Running Predictive on CPU to reduce GPU memory pressure...")
            guide_cellline.to("cpu")
            self.device = torch.device("cpu")

            # Move inputs to CPU
            model_inputs = {
                "N": N,
                "T": T,
                "C": C,
                "groups_ntc_tensor": groups_ntc_tensor.cpu(),
                "y_obs_ntc_tensor": y_obs_ntc_tensor.cpu(),
                "sum_factor_ntc_tensor": sum_factor_ntc_tensor.cpu(),
                "beta_o_alpha_tensor": beta_o_alpha_tensor.cpu(),
                "beta_o_beta_tensor": beta_o_beta_tensor.cpu(),
                "alpha_alpha_mu_tensor": alpha_alpha_mu_tensor.cpu(),
                "mu_x_mean_tensor": mu_x_mean_tensor.cpu(),
                "mu_x_sd_tensor": mu_x_sd_tensor.cpu(),
                "epsilon_tensor": epsilon_tensor.cpu(),
                "distribution": distribution,
                "denominator_ntc_tensor": denominator_ntc_tensor.cpu() if denominator_ntc_tensor is not None else None,
                "K": K,
                "D": D,
            }

        else:
            model_inputs = {
                "N": N,
                "T": T,
                "C": C,
                "groups_ntc_tensor": groups_ntc_tensor,
                "y_obs_ntc_tensor": y_obs_ntc_tensor,
                "sum_factor_ntc_tensor": sum_factor_ntc_tensor,
                "beta_o_alpha_tensor": beta_o_alpha_tensor,
                "beta_o_beta_tensor": beta_o_beta_tensor,
                "alpha_alpha_mu_tensor": alpha_alpha_mu_tensor,
                "mu_x_mean_tensor": mu_x_mean_tensor,
                "mu_x_sd_tensor": mu_x_sd_tensor,
                "epsilon_tensor": epsilon_tensor,
                "distribution": distribution,
                "denominator_ntc_tensor": denominator_ntc_tensor,
                "K": K,
                "D": D,
            }

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        import gc
        gc.collect()

        # Mini-batching logic (optional)
        max_samples = nsamples
        keep_sites = kwargs.get("keep_sites", lambda name, site: site["value"].ndim <= 2 or name != "y_obs_ntc")

        if minibatch_size is not None:
            from collections import defaultdict

            print(f"[INFO] Running Predictive in minibatches of {minibatch_size}...")
            predictive_technical = pyro.infer.Predictive(
                self._model_technical,
                guide=guide_cellline,
                num_samples=minibatch_size,
                parallel=True
            )
            all_samples = defaultdict(list)
            with torch.no_grad():
                for i in range(0, max_samples, minibatch_size):
                    samples = predictive_technical(**model_inputs)
                    for k, v in samples.items():
                        if keep_sites(k, {"value": v}):
                            all_samples[k].append(v.cpu())
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    import gc
                    gc.collect()

            posterior_samples = {k: torch.cat(v, dim=0) for k, v in all_samples.items()}

        else:
            predictive_technical = pyro.infer.Predictive(
                self._model_technical,
                guide=guide_cellline,
                num_samples=nsamples#,
                #parallel=True
            )
            with torch.no_grad():
                posterior_samples = predictive_technical(**model_inputs)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                import gc
                gc.collect()

        # Restore self.device if it was changed
        if run_on_cpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("[INFO] Reset self.device to:", self.device)

        for k, v in posterior_samples.items():
            posterior_samples[k] = v.cpu()
        
        self.loss_technical = losses
        self.posterior_samples_technical = posterior_samples
        if self.cis_gene is not None:
            self.alpha_y_prefit = posterior_samples["alpha_y"][:,:,self.counts.index.values != self.cis_gene]
            self.alpha_y_type = 'posterior'
            self.alpha_x_prefit = posterior_samples["alpha_y"][:,:,self.counts.index.values == self.cis_gene].squeeze(-1)
            self.alpha_x_type = 'posterior'
        else:
            self.alpha_y_prefit = posterior_samples["alpha_y"]
            self.alpha_y_type = 'posterior'
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        pyro.clear_param_store()
        import gc
        gc.collect()

        print("Finished fit_technical.")

    ########################################################
    # Step 2: Fit cis effects (model_x)
    ########################################################
    def _model_x(
        self,
        N,
        G,
        alpha_dirichlet_tensor,
        guides_tensor,
        x_obs_tensor,
        sum_factor_tensor,
        beta_o_alpha_tensor,
        beta_o_beta_tensor,
        alpha_alpha_mu_tensor,
        mu_x_mean_tensor,
        mu_x_sd_tensor,
        sigma_eff_mean_tensor,
        sigma_eff_sd_tensor,
        epsilon_tensor,
        C=None,
        groups_tensor=None,
        alpha_x_sample=None,
        o_x_sample=None,
        target_per_guide_tensor=None,
        independent_mu_sigma=False,
    ):

        if groups_tensor is not None:
            cfull_plate = pyro.plate("cfull_plate", C, dim=-2)  # <-- Store it as a variable
            
        ###################################
        ## Technical covariate modelling ##
        ###################################
        if alpha_x_sample is not None:
            alpha_x = alpha_x_sample
        elif groups_tensor is not None:
            with pyro.plate("c_plate", C - 1):
                alpha_alpha = pyro.sample("alpha_alpha", dist.Exponential(1 / alpha_alpha_mu_tensor)) # shape = [C-1]
                alpha_mu = pyro.sample("alpha_mu", dist.Gamma(1, 1))
                alpha_x = pyro.sample("alpha_x", dist.Gamma(alpha_alpha, alpha_alpha/alpha_mu)) # shape = [C-1]
        else:
            alpha_x = None
        
        ####################
        ## Overdispersion ##
        ####################
        if (o_x_sample is None) and (groups_tensor is not None):
            #with cfull_plate:
            #    beta_o = pyro.sample("beta_o", dist.Gamma(beta_o_alpha_tensor, beta_o_beta_tensor))
            #    o_x = pyro.sample("o_x", dist.Exponential(beta_o))
            #    phi_x = 1 / (o_x**2)
            beta_o = pyro.sample("beta_o", dist.Gamma(beta_o_alpha_tensor, beta_o_beta_tensor))
            o_x = pyro.sample("o_x", dist.Exponential(beta_o))
            phi_x = 1 / (o_x**2)
            #phi_x_used = phi_x[..., groups_tensor] # shape => [N]
            phi_x_used = phi_x
        elif (o_x_sample is not None) and (groups_tensor is not None):
            phi_x = 1/(o_x_sample ** 2)
            phi_x_used = phi_x[..., groups_tensor]
        else:
            ####################
            ## Overdispersion ##
            ####################
            beta_o = pyro.sample("beta_o", dist.Gamma(beta_o_alpha_tensor, beta_o_beta_tensor))
            o_x = pyro.sample("o_x", dist.Exponential(beta_o))
            phi_x_used = (1 / (o_x**2))

        ###############################
        ## Mixture model for x_eff_g ##
        ###############################
        if independent_mu_sigma:
            unique_targets = torch.unique(target_per_guide_tensor)
            mu_targets = {}
            sigma_targets = {}
            for t in unique_targets:
                mu_targets[int(t.item())] = pyro.sample(f"mu_target_{int(t.item())}", dist.Normal(mu_x_mean_tensor, mu_x_sd_tensor))
                sigma_targets[int(t.item())] = pyro.sample(f"sigma_target_{int(t.item())}", dist.HalfCauchy(scale=torch.tensor(2.0, device=self.device)))
            # gather mu and sigma for each guide
            mu_target_tensor = torch.stack([mu_targets[int(t.item())] for t in unique_targets], dim=0)
            sigma_target_tensor = torch.stack([sigma_targets[int(t.item())] for t in unique_targets], dim=0)
            mu = mu_target_tensor[target_per_guide_tensor]
            sigma = sigma_target_tensor[target_per_guide_tensor]
        else:
            mu = pyro.sample("mu", dist.Normal(mu_x_mean_tensor, mu_x_sd_tensor))
            sigma = pyro.sample("sigma", dist.HalfCauchy(scale=torch.tensor(2.0, device=self.device)))
            mu = mu.expand(G)
            sigma = sigma.expand(G)

        # Non-centered parameterization
        if (sigma_eff_mean_tensor >= 0.01) and (sigma_eff_sd_tensor >= 0.01):
            rate_alpha = (sigma_eff_sd_tensor ** 2) / (sigma_eff_mean_tensor ** 2)
            rate_beta = (sigma_eff_sd_tensor ** 2) / sigma_eff_mean_tensor
            sigma_eff_alpha = pyro.sample("sigma_eff_alpha", dist.Exponential(rate_alpha))
            sigma_eff_beta = pyro.sample("sigma_eff_beta", dist.Exponential(rate_beta))
        else:
            sigma_eff_alpha = pyro.sample("sigma_eff_alpha", dist.Gamma(1.0, 0.01))
            sigma_eff_beta = pyro.sample("sigma_eff_beta", dist.Gamma(1.0, 0.01))
        with pyro.plate("guides_plate", G):
            eps_x_eff_g = pyro.sample("eps_x_eff_g", dist.StudentT(df=3.0, loc=0.0, scale=1.0))
            log2_x_eff_g = mu + sigma * eps_x_eff_g
            x_eff_g = pyro.deterministic("x_eff_g", torch.tensor(2.0, device=self.device) ** log2_x_eff_g)
        
            sigma_eff = pyro.sample("sigma_eff", dist.Gamma(sigma_eff_alpha, sigma_eff_beta))
                
        ##########################
        ## Cell-level variables ##
        ##########################
        if alpha_x is not None:
            ones_ = torch.ones(alpha_x.shape[:-1] + (1,), device=self.device)
            alpha_x_full = torch.cat([ones_, alpha_x], dim=-1)  # shape = [C] or [S,C]
            alpha_x_used = alpha_x_full[..., groups_tensor]     # shape = [N] or [S,N]
        else:
            alpha_x_used = torch.ones_like(sum_factor_tensor)  # shape = [N] or [S,N]
        
        x_mean = x_eff_g[..., guides_tensor]  # no alpha_x_used here anymore
        
        ######################
        ## Cell-level plate ##
        ######################
        with pyro.plate("data_plate", N):
            log_x_true = pyro.sample( # use log2 of xtrue to allow small values of xtrue
                "log_x_true",
                dist.Normal(torch.log2(x_mean), sigma_eff[..., guides_tensor])
            )
            x_true = pyro.deterministic("x_true", torch.tensor(2.0, device=self.device) ** log_x_true)
            mu_obs = alpha_x_used * x_true * sum_factor_tensor
            pyro.sample(
                "x_obs",
                dist.NegativeBinomial(total_count=phi_x_used,
                                      logits=torch.log(mu_obs) - torch.log(phi_x_used),
                                      validate_args=False
                                     ), # important to use logits to allow small x_true
                obs=x_obs_tensor
            )

    def fit_cis(
        self,
        technical_covariates: list[str] = None,
        sum_factor_col: str = 'sum_factor',
        lr: float = 1e-3,
        niters: int = 50000,
        nsamples: int = 1000,
        alpha_ewma: float = 0.05,
        tolerance: float = 1e-4, # recommended to keep based on cell2location
        beta_o_beta: float = 3, # recommended to keep based on cell2location
        beta_o_alpha: float = 9, # recommended to keep based on cell2location
        alpha_alpha_mu: float = 5.8,
        epsilon: float = 1e-6,
        alpha_dirichlet: float = 0.1,
        minibatch_size: int = None,
        independent_mu_sigma: bool = False,   # <--- NEW
        **kwargs
    ):
        """
        Fits the cis effects (model_x) for your gene_of_interest.
        This step can be repeated multiple times with different priors
        or hyperparameters. 

        Parameters
        ----------
        lr : float
            Learning rate for Adam
        niters : int
            Number of SVI iterations
        nsamples : int
            Number of posterior samples
        alpha_ewma : float
            Exponential weight for smoothing the ELBO
        tolerance : float
            Convergence tolerance
        kwargs :
            Additional arguments controlling priors, etc.
        """
        print("Running fit_cis...")

        if self.cis_gene is None:
            raise ValueError("self.cis_gene must be set.")

        # convert to gpu for fitting
        if self.alpha_x_prefit is not None and self.alpha_x_prefit.device != self.device:
            self.alpha_x_prefit = self.alpha_x_prefit.to(self.device)

        if technical_covariates:
            if "technical_group_code" in self.meta.columns:
                warnings.warn("technical_group already set. Overwriting.")
                if self.alpha_x_prefit is not None:
                    warnings.warn("Overwriting alpha_x prefit, and refitting.")
                    self.alpha_x_prefit = None
                            
            self.meta["technical_group_code"] = self.meta.groupby(technical_covariates).ngroup()
            C = self.meta['technical_group_code'].nunique()
            groups_tensor = torch.tensor(self.meta['technical_group_code'].values, dtype=torch.long, device=self.device)
        elif self.alpha_x_prefit is None:
            C = None
            groups_tensor = None
            warnings.warn("no alpha_x_prefit and no technical_covariates provided, assuming no confounding effect.")
        else:
            C = self.meta['technical_group_code'].nunique()
            groups_tensor = torch.tensor(self.meta['technical_group_code'].values, dtype=torch.long, device=self.device)
        
        N = self.meta.shape[0]
        G = self.meta['guide_code'].nunique()
        
        x_obs_tensor = torch.tensor(self.counts.loc[self.cis_gene].values, dtype=torch.float32, device=self.device)
        guides_tensor = torch.tensor(self.meta['guide_code'].values, dtype=torch.long, device=self.device)
        if independent_mu_sigma:
            if ('target' not in self.meta.columns):
                raise ValueError("independent_mu_sigma is True, self.meta['target'] column not found.")
            elif self.meta['target'].nunique() < 2:
                raise ValueError("independent_mu_sigma is True, but only 1 target type found in self.meta['target'] column.")
            self.meta['target_code'] = pd.factorize(self.meta['target'])[0]
            target_codes_tensor = torch.tensor(self.meta['target_code'].values, dtype=torch.long, device=self.device)
        
            ### BUILD target_per_guide_tensor [G] based on guide → target
            target_per_guide_tensor = torch.empty(G, dtype=torch.long, device=self.device)
            for g in range(G):
                idx = (guides_tensor == g)
                guide_targets = torch.unique(target_codes_tensor[idx])
                if guide_targets.shape[0] != 1:
                    raise ValueError(f"Guide {g} maps to multiple targets: {guide_targets}")
                target_per_guide_tensor[g] = guide_targets[0]
        else:
            target_per_guide_tensor = None
        sum_factor_tensor = torch.tensor(self.meta[sum_factor_col].values, dtype=torch.float32, device=self.device)

        # Compute guide means (adjusting for alpha_x if provided)
        x_obs_factored = x_obs_tensor / sum_factor_tensor
        
        if self.alpha_x_prefit is not None:
            if self.alpha_x_type == 'posterior':
                # Take mean over posterior samples (S, C-1) -> (C-1)
                alpha_x_prefit_mean = self.alpha_x_prefit.mean(dim=0)
            else:
                # If it's a point estimate, ensure correct shape (C-1)
                alpha_x_prefit_mean = self.alpha_x_prefit.reshape((C-1,))
        
            # Ensure correct shape for alpha_x_full: (C, 1)
            alpha_x_full = torch.cat([torch.ones((1,), device=self.device), alpha_x_prefit_mean], dim=0)  # Shape: (2,1)
        
            # Select the correct alpha_x for each observation (expand for broadcasting)
            alpha_x_used = alpha_x_full[groups_tensor]  # groups_tensor indexes into (2,1)
            
            # Adjust x_obs_factored by alpha_x_used
            x_obs_factored /= alpha_x_used  # Ensure correct shape for division
        
        # Continue with the rest of the setup
        beta_o_alpha_tensor = torch.tensor(beta_o_alpha, dtype=torch.float32, device=self.device)
        beta_o_beta_tensor = torch.tensor(beta_o_beta, dtype=torch.float32, device=self.device)
        alpha_alpha_mu_tensor = torch.tensor(alpha_alpha_mu, dtype=torch.float32, device=self.device)
        alpha_dirichlet_tensor = torch.tensor(alpha_dirichlet, dtype=torch.float32, device=self.device)
        epsilon_tensor = torch.tensor(epsilon, dtype=torch.float32, device=self.device)

        unique_guides = torch.unique(guides_tensor)
        guide_means = torch.tensor([
            torch.log2(torch.mean(x_obs_factored[guides_tensor == g]))
            for g in unique_guides
            if torch.sum(x_obs_factored[guides_tensor == g]) > 0
        ], dtype=torch.float32, device=self.device)
        print(f"[DEBUG] guide_means min={guide_means.min()} median={guide_means.median()} mean={guide_means.mean()} max=={guide_means.max()}")
        mu_x_mean_tensor = torch.mean(guide_means)#.to(self.device)
        print(f"[DEBUG] mu_x_mean_tensor: {mu_x_mean_tensor}")
        mu_x_sd_tensor = torch.std(guide_means)#.to(self.device)
        # Compute guide-level MAD (robust estimate of log2 spread)
        guide_mads_tensor = torch.tensor([
            torch.median(torch.abs(torch.log2((x_obs_factored)[guides_tensor == g] + epsilon) - 
                                   torch.median(torch.log2((x_obs_factored)[guides_tensor == g] + epsilon))))
            for g in unique_guides
        ], dtype=torch.float32, device=self.device)
        guide_mads_tensor = guide_mads_tensor * 1.4826  # Gaussian-equivalent spread
        sigma_eff_mean_tensor = torch.mean(guide_mads_tensor)#.to(self.device)
        sigma_eff_sd_tensor = torch.std(guide_mads_tensor)#.to(self.device)        
        mu_x_alpha_tensor = mu_x_mean_tensor ** 2 / (mu_x_sd_tensor ** 2)
        mu_x_beta_tensor = mu_x_mean_tensor / (mu_x_sd_tensor ** 2)

        def init_loc_fn(site):
            name = site["name"]
        
            if name == "mu":
                return mu_x_mean_tensor
            elif name == "sigma":
                return mu_x_sd_tensor.clamp(min=1e-3)
        
            elif name == "sigma_eff_alpha":
                # Corresponds to (mean^2 / var)
                sigma_eff_mean_tensor_tmp = sigma_eff_mean_tensor.clamp(min=1e-2)
                sigma_eff_sd_tensor_tmp = sigma_eff_sd_tensor.clamp(min=1e-2)
                return ((sigma_eff_mean_tensor_tmp ** 2) / (sigma_eff_sd_tensor_tmp ** 2))
            elif name == "sigma_eff_beta":
                # Corresponds to (mean / var)
                sigma_eff_mean_tensor_tmp = sigma_eff_mean_tensor.clamp(min=1e-2)
                sigma_eff_sd_tensor_tmp = sigma_eff_sd_tensor.clamp(min=1e-2)
                return (sigma_eff_mean_tensor_tmp / (sigma_eff_sd_tensor_tmp ** 2))
        
            elif name == "sigma_eff":
                # Assume you’re sampling per-guide (e.g. G = 37)
                return sigma_eff_mean_tensor.clamp(min=1e-2).expand(G)
        
            return pyro.infer.autoguide.initialization.init_to_median(site)
        
        guide_x = pyro.infer.autoguide.AutoNormalMessenger(self._model_x, init_loc_fn=init_loc_fn)
        optimizer = pyro.optim.Adam({"lr": lr})
        svi = pyro.infer.SVI(self._model_x, guide_x, optimizer, 
                             loss=pyro.infer.Trace_ELBO())

        losses = []
        smoothed_loss = None
        for step in range(niters):
            
            if self.alpha_x_prefit is not None:
                samp = torch.randint(high=self.alpha_x_prefit.shape[0], size=(1,)).item()
            alpha_x_sample = (
                self.alpha_x_prefit[samp] if self.alpha_x_type == "posterior"
                else self.alpha_x_prefit
            ) if self.alpha_x_prefit is not None else None
            o_x_sample = None
            
            loss = svi.step(
                N,
                G,
                alpha_dirichlet_tensor,
                guides_tensor,
                x_obs_tensor,
                sum_factor_tensor,
                beta_o_alpha_tensor,
                beta_o_beta_tensor,
                alpha_alpha_mu_tensor,
                mu_x_mean_tensor,
                mu_x_sd_tensor,
                sigma_eff_mean_tensor.clamp(min=1e-2),
                sigma_eff_sd_tensor.clamp(min=1e-2),
                epsilon_tensor,
                C = C,
                groups_tensor=groups_tensor,
                alpha_x_sample=alpha_x_sample,
                o_x_sample=o_x_sample,
                target_per_guide_tensor=target_per_guide_tensor,
                independent_mu_sigma=independent_mu_sigma,
            )
            losses.append(loss)
            if step % 1000 == 0:
                print(f"Step {step} : loss = {loss:.5e}, device: {mu_x_mean_tensor.device}")
            if smoothed_loss is None:
                smoothed_loss = loss
            else:
                if abs(alpha_ewma * (loss - smoothed_loss)) < tolerance:
                    print(f"Converged at step {step}! Loss = {loss:.5e}")
                    break
                smoothed_loss = alpha_ewma * loss + (1 - alpha_ewma) * smoothed_loss
            

        # Move to CPU if using too much GPU memory for Predictive
        run_on_cpu = self.device.type != "cpu"

        if run_on_cpu:
            print("[INFO] Running Predictive on CPU to reduce GPU memory pressure...")
            guide_x.to("cpu")
            self.device = torch.device("cpu")

            model_inputs = {
                "N": N,
                "G": G,
                "alpha_dirichlet_tensor": alpha_dirichlet_tensor.cpu(),
                "guides_tensor": guides_tensor.cpu(),
                "x_obs_tensor": x_obs_tensor.cpu(),
                "sum_factor_tensor": sum_factor_tensor.cpu(),
                "beta_o_alpha_tensor": beta_o_alpha_tensor.cpu(),
                "beta_o_beta_tensor": beta_o_beta_tensor.cpu(),
                "alpha_alpha_mu_tensor": alpha_alpha_mu_tensor.cpu(),
                "mu_x_mean_tensor": mu_x_mean_tensor.cpu(),
                "mu_x_sd_tensor": mu_x_sd_tensor.cpu(),
                "sigma_eff_mean_tensor": sigma_eff_mean_tensor.clamp(min=1e-2).cpu(),
                "sigma_eff_sd_tensor": sigma_eff_sd_tensor.clamp(min=1e-2).cpu(),
                "epsilon_tensor": epsilon_tensor.cpu(),
                "C": C,
                "groups_tensor": groups_tensor.cpu() if groups_tensor is not None else None,
                "alpha_x_sample": self.alpha_x_prefit.mean(dim=0).cpu() if self.alpha_x_type == "posterior" else (self.alpha_x_prefit.cpu() if self.alpha_x_prefit is not None else None),
                "o_x_sample": None,
                "target_per_guide_tensor": target_per_guide_tensor.cpu() if target_per_guide_tensor is not None else None,
                "independent_mu_sigma": independent_mu_sigma,
            }
        else:
            model_inputs = {
                "N": N,
                "G": G,
                "alpha_dirichlet_tensor": alpha_dirichlet_tensor,
                "guides_tensor": guides_tensor,
                "x_obs_tensor": x_obs_tensor,
                "sum_factor_tensor": sum_factor_tensor,
                "beta_o_alpha_tensor": beta_o_alpha_tensor,
                "beta_o_beta_tensor": beta_o_beta_tensor,
                "alpha_alpha_mu_tensor": alpha_alpha_mu_tensor,
                "mu_x_mean_tensor": mu_x_mean_tensor,
                "mu_x_sd_tensor": mu_x_sd_tensor,
                "sigma_eff_mean_tensor": sigma_eff_mean_tensor.clamp(min=1e-2),
                "sigma_eff_sd_tensor": sigma_eff_sd_tensor.clamp(min=1e-2),
                "epsilon_tensor": epsilon_tensor,
                "C": C,
                "groups_tensor": groups_tensor,
                "alpha_x_sample": self.alpha_x_prefit.mean(dim=0) if self.alpha_x_type == "posterior" else (self.alpha_x_prefit if self.alpha_x_prefit is not None else None),
                "o_x_sample": None,
                "target_per_guide_tensor": target_per_guide_tensor if target_per_guide_tensor is not None else None,
                "independent_mu_sigma": independent_mu_sigma,
            }

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        import gc
        gc.collect()

        max_samples = nsamples
        keep_sites = kwargs.get("keep_sites", lambda name, site: site["value"].ndim <= 2 or name != "x_obs")

        if minibatch_size is not None:
            from collections import defaultdict
            print(f"[INFO] Running Predictive in minibatches of {minibatch_size}...")
            predictive_x = pyro.infer.Predictive(
                self._model_x,
                guide=guide_x,
                num_samples=minibatch_size,
                parallel=True
            )
            all_samples = defaultdict(list)
            with torch.no_grad():
                for i in range(0, max_samples, minibatch_size):
                    samples = predictive_x(**model_inputs)
                    for k, v in samples.items():
                        if keep_sites(k, {"value": v}):
                            all_samples[k].append(v.cpu())
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    import gc
                    gc.collect()
            posterior_samples_x = {k: torch.cat(v, dim=0) for k, v in all_samples.items()}
        else:
            predictive_x = pyro.infer.Predictive(
                self._model_x,
                guide=guide_x,
                num_samples=nsamples#,
                #parallel=True
            )
            with torch.no_grad():
                posterior_samples_x = predictive_x(**model_inputs)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                import gc
                gc.collect()

        if run_on_cpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("[INFO] Reset self.device to:", self.device)

        for k, v in posterior_samples_x.items():
            posterior_samples_x[k] = v.cpu()
        
        self.loss_x = losses
        self.posterior_samples_cis = posterior_samples_x
        self.x_true = posterior_samples_x['x_true']
        self.x_true_type = 'posterior'
        self.log2_x_true = posterior_samples_x['log_x_true']
        self.log2_x_true_type = 'posterior'
        if self.alpha_x_prefit is None and technical_covariates:
            self.alpha_x_prefit = posterior_samples_x["alpha_x"]
            self.alpha_x_type = 'posterior'

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        pyro.clear_param_store()

        print("Finished fit_cis.")

    def refit_sumfactor(
        self,
        sum_factor_col_old: str = "sum_factor",
        sum_factor_col_refit: str = "sum_factor_new",
        covariates: list[str] = None, # ["lane", "cell_line"] or could be empty
        n_knots: int = 5,
        degree: int = 3,
        alpha: float = 0.1
    ):
        """
        Step 2 of sum factor adjustment: Remove cis gene contribution.

        Use AFTER fit_cis() and BEFORE fit_trans().
        Fits a spline regression: (sum_factor - baseline_ntc) ~ f(x_true)
        Then removes the predicted x_true contribution from sum factors.

        This ensures trans modeling isn't confounded by cis expression levels.

        Typical workflow:
            1. adjust_ntc_sum_factor() -> creates 'sum_factor_adj'
            2. fit_cis(sum_factor_col='sum_factor_adj')
            3. refit_sumfactor() -> creates 'sum_factor_refit'  <-- This step
            4. fit_trans(sum_factor_col='sum_factor_refit')

        Parameters
        ----------
        sum_factor_col_old : str
            Name of existing sum factor column (typically from adjust_ntc_sum_factor)
        sum_factor_col_refit : str
            Name for refitted sum factor column to create (default: 'sum_factor_new')
        covariates : list of str, optional
            Columns to group by for baseline NTC calculation (e.g., ['cell_line', 'lane'])
        n_knots : int
            Number of spline knots for regression (default: 5)
        degree : int
            Polynomial degree of spline pieces (default: 3)
        alpha : float
            Ridge regression regularization parameter (default: 0.1)

        Returns
        -------
        None
            Creates new column sum_factor_col_refit in self.meta

        Notes
        -----
        Algorithm:
        1. Compute baseline NTC sum factor for each covariate group
        2. Compute leftover = sum_factor - baseline_ntc
        3. Fit spline model: leftover ~ f(x_true)
        4. Predict x_true contribution: y_pred = model(x_true)
        5. Adjusted sum factor = max(0, leftover - y_pred + baseline_ntc)

        Requires:
        - self.x_true must be set (from fit_cis())
        - self.x_true_type must be 'posterior' or 'point'
        """
        sum_factor_data = self.meta[sum_factor_col_old].values  # shape (N,)
        
        if covariates is None:
            covariates = []

        if covariates:
            # Create a single group identifier by concatenating covariate values
            tech_group = self.meta[covariates].astype(str).agg('_'.join, axis=1)
            groups, group_id = np.unique(tech_group, return_inverse=True)
            n_groups = len(groups)
        else:
            # No grouping, treat all samples as a single group
            groups, group_id = np.array(["all"]), np.zeros(len(self.meta), dtype=int)
            n_groups = 1
        
        baseline_ntc_of_group = np.zeros(n_groups)
        for grp, grp_name in enumerate(groups):
            mask_grp = (tech_group == grp_name)
            # Among that group, pick rows with gene == 'ntc'
            mask_ntc = (self.meta['target'] == 'ntc') & mask_grp
        
            # If a group has no NTC, you must decide on a fallback
            # For example, use the overall mean or 1.0, or skip that group
            if not np.any(mask_ntc):
                baseline_ntc_of_group[grp] = 1.0  # fallback
            else:
                # The mean of sum_factor_data among the NTC rows
                baseline_ntc_of_group[grp] = np.mean(sum_factor_data[mask_ntc])
        
        # leftover_data[i] = sum_factor[i] - baseline_ntc_of_group[group_id[i]]
        leftover_data = sum_factor_data - baseline_ntc_of_group[group_id]
    
        # ------------------------------------------------------------------------------
        # Build a pipeline with a Spline transformer + Linear regression
        # ------------------------------------------------------------------------------
        # 'n_knots' is how many spline knots to use;
        # 'degree' is the polynomial degree of each spline piece.
        model_spline_ridge = make_pipeline(
            SplineTransformer(n_knots=n_knots, degree=degree),
            Ridge(alpha=alpha)  # alpha > 0 adds penalty
        )
        
        # ------------------------------------------------------------------------------
        # Fit the model on training data
        # ------------------------------------------------------------------------------
        if self.x_true_type == 'posterior':
            X_true = self.x_true.mean(dim=0)
        else:
            X_true = self.x_true
        model_spline_ridge.fit(X_true.reshape(-1, 1), leftover_data)
        
        # ------------------------------------------------------------------------------
        # Predict on train & test
        # ------------------------------------------------------------------------------
        y_pred = model_spline_ridge.predict(X_true.reshape(-1, 1))
        self.meta[sum_factor_col_refit] = np.max(np.vstack((leftover_data - y_pred + baseline_ntc_of_group[group_id], np.zeros_like(leftover_data))), axis=0)
        print(f"[INFO] Created '{sum_factor_col_refit}' in meta with xtrue-based adjustment.")

    #########################################
    ## Step 3: Fit trans effects (model_y) ##
    #########################################
    def _model_y(
        self,
        N,
        T,
        y_obs_tensor,
        sum_factor_tensor,
        beta_o_alpha_tensor,
        beta_o_beta_tensor,
        alpha_alpha_mu_tensor,
        gamma_alpha_tensor,
        gamma_beta_tensor,
        K_max_tensor,
        K_alpha_tensor,
        Vmax_mean_tensor,
        Vmax_alpha_tensor,
        n_mu_tensor,
        n_sigma_base_tensor,
        Amean_tensor,
        p_n_tensor,
        threshold,
        slope,
        epsilon_tensor,
        x_true_sample,
        log2_x_true_sample,
        alpha_y_sample=None,
        C=None,
        groups_tensor=None,
        predictive_mode=False,
        temperature=1.0,
        use_straight_through=False,
        function_type='single_hill',
        polynomial_degree=6,
        use_alpha=True,
        distribution='negbinom',
        denominator_tensor=None,
        K=None,
        D=None,
    ):

        if use_alpha:
            alpha_dist = dist.RelaxedBernoulliStraightThrough if use_straight_through else dist.RelaxedBernoulli

        if groups_tensor is not None:
            cfull_plate = pyro.plate("cfull_plate", C, dim=-2)  # <-- Store it as a variable
        trans_plate = pyro.plate("trans_plate", T, dim=-1)
        
        ##########
        ## x_true (Now Fully Observed) ##
        ##########
        #x_true = pyro.deterministic("x_true", x_true_sample)
        x_true = x_true_sample
    
        ####################
        ## alpha_y ##
        ####################
        alpha_y = None
        if alpha_y_sample is not None:
            alpha_y = alpha_y_sample
        elif groups_tensor is not None:
            with pyro.plate("cell_lines_plate", C-1, dim=-2):  # **Now correctly uses C-1**
                with trans_plate:
                    alpha_alpha = pyro.sample("alpha_alpha", dist.Exponential(1 / alpha_alpha_mu_tensor))  # shape = [C-1, T]
                    alpha_mu = pyro.sample("alpha_mu", dist.Gamma(1, 1))  # shape = [C-1, T]
                    alpha_y = pyro.sample("alpha_y", dist.Gamma(alpha_alpha, alpha_alpha / alpha_mu))  # shape = [C-1, T]
    
        ####################
        ## Overdispersion ##
        ####################
        beta_o = pyro.sample("beta_o", dist.Gamma(beta_o_alpha_tensor, beta_o_beta_tensor))
        with trans_plate:
            o_y = pyro.sample("o_y", dist.Exponential(beta_o))
            phi_y = 1 / (o_y**2)
        phi_y_used = phi_y.unsqueeze(-2)
    
        #################
        ## Hill-based: ##
        #################
        if function_type in ['single_hill', 'additive_hill', 'nested_hill']:
            # ----------------------------------------------------
            # 1) Define global hyperparameters for n_a
            # ----------------------------------------------------
            sigma_n_a = pyro.sample("sigma_n_a", dist.Exponential(1/5)) #   -> controls how variable n_a can be across genes
            if function_type in ['additive_hill', 'nested_hill']:
                sigma_n_b = pyro.sample("sigma_n_b", dist.Exponential(1/5)) #   -> controls how variable n_a can be across genes
        if function_type in ['polynomial']:
            #sigma_coeff = pyro.sample("sigma_coeff", dist.Exponential(100)) #   -> controls how variable n_a can be across genes
            sigma_coeff = pyro.sample("sigma_coeff", dist.HalfCauchy(scale=1.0))
        
        # Now enter the trans_plate (T dimension)
        with trans_plate:
            
            weight = o_y / (o_y + (beta_o_beta_tensor / beta_o_alpha_tensor))
            Amean_adjusted = ((1 - weight) * Amean_tensor) + (weight * Vmax_mean_tensor) + epsilon_tensor
            A = pyro.sample("A", dist.Exponential(1 / Amean_adjusted))

            if use_alpha:
                # Relaxed Bernoulli: alpha ~ (0,1), becomes more discrete as temperature -> 0
                alpha = pyro.sample("alpha", alpha_dist(temperature=temperature, probs=p_n_tensor))
            else:
                alpha = torch.ones((T,), device=self.device)
            
            if function_type in ['single_hill', 'additive_hill', 'nested_hill']:
                
                #####################################
                ## function priors (depend on o_y) ##
                #####################################
                # Gamma and delta depend on T dimension
                # Reduce over group dimension if necessary
                K_sigma = (K_max_tensor / (2 * torch.sqrt(K_alpha_tensor))) + epsilon_tensor
                Vmax_sigma = (Vmax_mean_tensor / torch.sqrt(Vmax_alpha_tensor)) + epsilon_tensor
                # Replace Laplace with SoftLaplace
                # Instead of directly n_a, we do n_a_raw
                n_a_raw = pyro.sample("n_a_raw", dist.Normal(n_mu_tensor, sigma_n_a))
                #n_a = pyro.sample("n_a", dist.Normal(n_mu_tensor, sigma_n))
                n_a = pyro.deterministic("n_a", (alpha * n_a_raw).clamp(min=-20, max=20)) # clamp for numerical stability
                
                # Scale for Vmax, K is multiplied by alpha
                #eff_Vmax_sigma = alpha * Vmaxa_sigma + epsilon_tensor
                #eff_Ka_sigma   = alpha * Ka_sigma    + epsilon_tensor
        
                Vmax_a = pyro.sample("Vmax_a", dist.Gamma((Vmax_mean_tensor ** 2) / (Vmax_sigma ** 2), Vmax_mean_tensor / (Vmax_sigma ** 2)))
                K_a = pyro.sample("K_a", dist.Gamma(((K_max_tensor/2) ** 2) / (K_sigma ** 2), (K_max_tensor/2) / (K_sigma ** 2)))
                Hill_func_a = Hill_based_positive(x_true.unsqueeze(-1), Vmax=Vmax_a, A=0, K=K_a, n=n_a, epsilon=epsilon_tensor)
                # Sample all required parameters (same for all hill types)            
                if function_type in ['additive_hill', 'nested_hill']:
                    beta = pyro.sample("beta", alpha_dist(temperature=temperature, probs=p_n_tensor))
                    n_b_raw = pyro.sample("n_b_raw", dist.Normal(n_mu_tensor, sigma_n_b))
                    n_b = pyro.deterministic("n_b", (beta * n_b_raw).clamp(min=-20, max=20)) # clamp for numerical stability
                    Vmax_b = pyro.sample("Vmax_b", dist.Gamma((Vmax_mean_tensor ** 2) / (Vmax_sigma ** 2), Vmax_mean_tensor / (Vmax_sigma ** 2)))
                    K_b = pyro.sample("K_b", dist.Gamma(((K_max_tensor/2) ** 2) / (K_sigma ** 2), (K_max_tensor/2) / (K_sigma ** 2)))
                
                # Compute Hill function(s)
                if function_type == 'single_hill':
                    Hilla = Hill_based_positive(x_true.unsqueeze(-1), Vmax=Vmax_a, A=0, K=K_a, n=n_a, epsilon=epsilon_tensor)
                    y_true_mu = A + (alpha * Hilla)
                elif function_type == 'additive_hill':
                    Hilla = Hill_based_positive(x_true.unsqueeze(-1), Vmax=Vmax_a, A=0, K=K_a, n=n_a, epsilon=epsilon_tensor)
                    Hillb = Hill_based_positive(x_true.unsqueeze(-1), Vmax=Vmax_b, A=0, K=K_b, n=n_b, epsilon=epsilon_tensor)
                    y_true_mu = A + (alpha * Hilla) + (beta * Hillb)
                elif function_type == 'nested_hill':
                    Hilla = Hill_based_positive(x_true.unsqueeze(-1), Vmax=Vmax_a, A=0, K=K_a, n=n_a, epsilon=epsilon_tensor)
                    Hillb = Hill_based_positive(Hilla, Vmax=Vmax_b, A=0, K=K_b, n=n_b, epsilon=epsilon_tensor)
                    y_true_mu = A + (alpha * Hillb)
                log_y_true_mu = torch.log(y_true_mu)

            elif function_type == 'polynomial':
                assert polynomial_degree is not None and polynomial_degree >= 1, \
                    "polynomial_degree must be ≥ 1 (no intercept, A is handled separately)"
            
                # 1) take log2 of inputs
                log2_x_true = torch.log2(x_true_sample)  # shape [N]
                
                # Stack polynomial coefficients
                coeffs = []
                for d in range(1, polynomial_degree + 1):  # start at degree 1, no intercept
                    coeff = pyro.sample(f"poly_coeff_{d}", dist.Normal(0., sigma_coeff))
                    #check_tensor(f"coeff_{d}", coeff)
                    coeffs.append(coeff)
                coeffs = torch.stack(coeffs, dim=-2)  # [degree, T]
                if (coeffs.shape[1] == 1) & (coeffs.ndim == 4):
                    coeffs = coeffs.squeeze(1)        # [S, D, T]
                #check_tensor(f"coeffs", coeffs)
                
                # 4) compute polynomial value exactly as before
                poly_val = Polynomial_function(log2_x_true, coeffs)  # [N, T]
                log2_y_true_mu = torch.log2(A) + alpha * poly_val       # [N, T]
                #log2_y_true_mu = torch.log2(A) + poly_val       # [N, T]
                log_y_true_mu  = log2_y_true_mu * torch.log(torch.tensor(2.0, device=self.device))  # Convert log2 to ln
            else:
                raise ValueError(f"Unknown function_type: {function_type}")
            
            
            ##########################
            ## Cell-level variables ##
            ##########################
            if alpha_y is not None:   
                # If in sampling mode, alpha_y will be [S, C-1, T]
                if alpha_y.dim() == 3:  # Predictive case: shape is (S, C-1, T)
                    ones_shape = (alpha_y.shape[0], 1, T)  # Match batch dimension S
                    # Ensure ones tensor has correct batch shape
                    alpha_y_full = torch.cat([torch.ones(ones_shape, device=self.device), alpha_y], dim=1)
                elif alpha_y.dim() == 2:  # Training case: shape is (C-1, T)
                    ones_shape = (1, T)
                    # Ensure ones tensor has correct batch shape
                    alpha_y_full = torch.cat([torch.ones(ones_shape, device=self.device), alpha_y], dim=0)
                else:  # Unexpected extra dimension
                    raise ValueError(f"[ERROR] Unexpected alpha_y shape: {alpha_y.shape}")
                alpha_y_used = alpha_y_full[groups_tensor, :]  # [N, T]
                log_y_true = log_y_true_mu + torch.log(alpha_y_used) #+ 1e-10
            else:
                log_y_true = log_y_true_mu #+ 1e-10
    
            # Now we sample y_obs which is NxT (or NxTxK for multinomial, NxTxD for mvnormal)
            # Use data_plate for N dimension

            # Compute mu_y from log_y_true (this is the dose-response function output)
            mu_y = torch.exp(log_y_true)  # [N, T]

            # Debug checks (keep for troubleshooting)
            if torch.isnan(log_y_true).any() or torch.isinf(log_y_true).any():
                check_tensor("log_y_true", log_y_true)
                check_tensor("y_true_mu", y_true_mu)
                check_tensor("sum_factor_tensor", sum_factor_tensor)
                check_tensor("phi_y_used", phi_y_used)
                check_tensor("A", A)
                if function_type in ['single_hill', 'additive_hill', 'nested_hill']:
                    check_tensor("n_a", n_a)
                    check_tensor("Vmax_a", Vmax_a)
                    check_tensor("K_a", K_a)
                if function_type in ['additive_hill', 'nested_hill']:
                    check_tensor("n_b", n_b)
                    check_tensor("Vmax_b", Vmax_b)
                    check_tensor("K_b", K_b)
                if function_type == 'polynomial':
                    check_tensor("coeffs", coeffs)

            # Call distribution-specific observation sampler
            from .distributions import get_observation_sampler
            observation_sampler = get_observation_sampler(distribution, 'trans')

            # Prepare alpha_y_full (full C cell lines, including reference)
            if alpha_y is not None and groups_tensor is not None:
                if alpha_y.dim() == 3:  # Predictive: (S, C-1, T)
                    ones_shape = (alpha_y.shape[0], 1, T)
                    alpha_y_full = torch.cat([torch.ones(ones_shape, device=self.device), alpha_y], dim=1)
                elif alpha_y.dim() == 2:  # Training: (C-1, T)
                    ones_shape = (1, T)
                    alpha_y_full = torch.cat([torch.ones(ones_shape, device=self.device), alpha_y], dim=0)
                else:
                    raise ValueError(f"Unexpected alpha_y shape: {alpha_y.shape}")
            else:
                alpha_y_full = None

            # Call the appropriate sampler based on distribution
            if distribution == 'negbinom':
                observation_sampler(
                    y_obs_tensor=y_obs_tensor,
                    mu_y=mu_y,
                    phi_y_used=phi_y_used,
                    alpha_y_full=alpha_y_full,
                    groups_tensor=groups_tensor,
                    sum_factor_tensor=sum_factor_tensor,
                    N=N,
                    T=T,
                    C=C
                )
            elif distribution == 'multinomial':
                # For multinomial, mu_y should be probabilities [N, T, K]
                observation_sampler(
                    y_obs_tensor=y_obs_tensor,
                    mu_y=mu_y,  # Should be [N, T, K] probabilities
                    N=N,
                    T=T,
                    K=K
                )
            elif distribution == 'binomial':
                observation_sampler(
                    y_obs_tensor=y_obs_tensor,
                    denominator_tensor=denominator_tensor,
                    mu_y=mu_y,  # Should be probabilities [N, T]
                    alpha_y_full=alpha_y_full,
                    groups_tensor=groups_tensor,
                    N=N,
                    T=T,
                    C=C
                )
            elif distribution == 'normal':
                # For normal, we need sigma_y (standard deviation)
                sigma_y = 1.0 / torch.sqrt(phi_y)  # Convert from precision to std dev
                observation_sampler(
                    y_obs_tensor=y_obs_tensor,
                    mu_y=mu_y,
                    sigma_y=sigma_y,
                    alpha_y_full=alpha_y_full,
                    groups_tensor=groups_tensor,
                    N=N,
                    T=T,
                    C=C
                )
            elif distribution == 'mvnormal':
                # For mvnormal, mu_y should be [N, T, D] and we need covariance
                # Use phi_y to construct covariance (for now, diagonal)
                sigma_y = 1.0 / torch.sqrt(phi_y)  # [T]
                cov_y = torch.diag_embed(sigma_y.unsqueeze(-1).expand(T, D))  # [T, D, D]
                observation_sampler(
                    y_obs_tensor=y_obs_tensor,
                    mu_y=mu_y,  # Should be [N, T, D]
                    cov_y=cov_y,
                    alpha_y_full=alpha_y_full,
                    groups_tensor=groups_tensor,
                    N=N,
                    T=T,
                    D=D,
                    C=C
                )
            else:
                raise ValueError(f"Unknown distribution: {distribution}")

    ########################################################
    # Step 3: Fit trans effects (model_y)
    ########################################################
    def fit_trans(
        self,
        technical_covariates: list[str] = None,
        sum_factor_col: str = None,
        function_type: str = 'single_hill',  # or 'additive', 'nested'
        polynomial_degree: int = 6,
        lr: float = 1e-3,
        niters: int = 50000,
        nsamples: int = 1000,
        alpha_ewma: float = 0.05,
        tolerance: float = 1e-4, # recommended to keep based on cell2location
        beta_o_beta: float = 3, # recommended to keep based on cell2location
        beta_o_alpha: float = 9, # recommended to keep based on cell2location
        alpha_alpha_mu: float = 5.8,
        gamma_alpha: float = 1e-8,
        K_alpha: float = 2,
        Vmax_alpha: float = 2,
        n_mu: float = 0,
        n_sigma_base: float = 5,
        p_n: float = 1e-6,
        epsilon: float = 1e-6,
        threshold: float = 0.1,
        slope: float = 50,
        gamma_beta: float = None,
        gamma_threshold: float = None, # 0.01
        p0: float = None, # 0.01
        init_temp: float = 1.0,
        #final_temp: float = 1e-8,
        final_temp: float = 0.1,
        minibatch_size: int = None,
        distribution: str = 'negbinom',
        denominator: np.ndarray = None,
        **kwargs
    ):
        """
        Fit trans effects using distribution-specific likelihood.

        Parameters
        ----------
        distribution : str
            Distribution type: 'negbinom', 'multinomial', 'binomial', 'normal', 'mvnormal'
        sum_factor_col : str, optional
            Sum factor column name. Required for negbinom, ignored for others.
        denominator : np.ndarray, optional
            Denominator array for binomial distribution (e.g., total counts for PSI)
        technical_covariates : list of str, optional
            Covariates for cell-line effects
        function_type : str
            Dose-response function: 'single_hill', 'additive_hill', 'polynomial'
        **kwargs
            Additional parameters for specific distributions
        """
        print(f"Running fit_trans with distribution={distribution}...")

        # Validate distribution-specific requirements
        from .distributions import requires_sum_factor, requires_denominator

        if requires_sum_factor(distribution) and sum_factor_col is None:
            raise ValueError(f"Distribution '{distribution}' requires sum_factor_col parameter")

        if requires_denominator(distribution) and denominator is None:
            raise ValueError(f"Distribution '{distribution}' requires denominator parameter")

        # convert to gpu for fitting if applicable
        if self.x_true is not None and self.x_true.device != self.device:
            self.x_true = self.x_true.to(self.device)
        if self.alpha_y_prefit is not None and self.alpha_y_prefit.device != self.device:
            self.alpha_y_prefit = self.alpha_y_prefit.to(self.device)

        if not hasattr(self, "log2_x_true") or self.log2_x_true is None:
            if self.x_true is not None:
                self.log2_x_true = torch.log2(self.x_true)
                self.log2_x_true_type = self.x_true_type
        
        if technical_covariates:
            if "technical_group_code" in self.meta.columns:
                warnings.warn("technical_group already set. Overwriting.")
                if self.alpha_y_prefit:
                    warnings.warn("Overwriting alpha_y prefit, and refitting.")
                    self.alpha_y_prefit = None
                
            self.meta["technical_group_code"] = self.meta.groupby(technical_covariates).ngroup()
            C = self.meta['technical_group_code'].nunique()
            groups_tensor = torch.tensor(self.meta['technical_group_code'].values, dtype=torch.long, device=self.device)
        elif self.alpha_y_prefit is None:
            C = None
            groups_tensor = None
            warnings.warn("no alpha_y_prefit and no technical_covariates provided, assuming no confounding effect.")
        else:
            C = self.meta['technical_group_code'].nunique()
            groups_tensor = torch.tensor(self.meta['technical_group_code'].values, dtype=torch.long, device=self.device)
        
        N = self.meta.shape[0]
        if self.cis_gene is not None:
            y_obs = self.counts.drop([self.cis_gene]).values.T
        else:
            warnings.warn("self.cis_gene is None. Assuming self.counts does not include cis_gene. If it does, please abort and set self.cis_gene.")
            y_obs = self.counts.values.T
        T = y_obs.shape[1]

        # Handle sum factors (only for distributions that need them)
        if sum_factor_col is not None:
            sum_factor_tensor = torch.tensor(self.meta[sum_factor_col].values, dtype=torch.float32, device=self.device)
        else:
            # Create dummy sum factors for distributions that don't need them
            sum_factor_tensor = torch.ones(N, dtype=torch.float32, device=self.device)

        # Handle denominator (for binomial)
        denominator_tensor = None
        if denominator is not None:
            denominator_tensor = torch.tensor(denominator, dtype=torch.float32, device=self.device)

        # Detect data dimensions (for multinomial and mvnormal)
        from .distributions import is_3d_distribution
        K = None
        D = None
        if is_3d_distribution(distribution):
            if y_obs.ndim == 3:
                if distribution == 'multinomial':
                    K = y_obs.shape[2]  # Number of categories
                elif distribution == 'mvnormal':
                    D = y_obs.shape[2]  # Dimensionality
            else:
                raise ValueError(f"Distribution '{distribution}' requires 3D data but got shape {y_obs.shape}")
        if self.x_true_type == 'point':
            x_true_mean = self.x_true
        elif self.x_true_type == 'posterior':
            x_true_mean = self.x_true.mean(dim=0)
        beta_o_alpha_tensor = torch.tensor(beta_o_alpha, dtype=torch.float32, device=self.device)
        beta_o_beta_tensor = torch.tensor(beta_o_beta, dtype=torch.float32, device=self.device)
        alpha_alpha_mu_tensor = torch.tensor(alpha_alpha_mu, dtype=torch.float32, device=self.device)
        gamma_alpha_tensor = torch.tensor(gamma_alpha, dtype=torch.float32, device=self.device)
        K_alpha_tensor = torch.tensor(K_alpha, dtype=torch.float32, device=self.device)
        Vmax_alpha_tensor = torch.tensor(Vmax_alpha, dtype=torch.float32, device=self.device)
        n_mu_tensor = torch.tensor(n_mu, dtype=torch.float32, device=self.device)
        n_sigma_base_tensor = torch.tensor(n_sigma_base, dtype=torch.float32, device=self.device)
        y_obs_tensor = torch.tensor(y_obs, dtype=torch.float32, device=self.device)
        epsilon_tensor = torch.tensor(epsilon, dtype=torch.float32, device=self.device)
        p_n_tensor = torch.tensor(p_n, dtype=torch.float32, device=self.device)


        if gamma_beta is None:
            if p0 is None and gamma_threshold is None:
                raise ValueError("gamma_beta or p0 and gamma_threshold have to be specified.")
            else:
                gamma_beta = find_beta(gamma_alpha, gamma_threshold, 1 - (p0 / T)) * ((beta_o_alpha / beta_o_beta) ** 2)
        gamma_beta_tensor = torch.tensor(gamma_beta, dtype=torch.float32, device=self.device)

        guides_tensor = torch.tensor(self.meta['guide_code'].values, dtype=torch.long, device=self.device)
        K_max_tensor = torch.max(torch.stack([torch.mean(x_true_mean[guides_tensor == g]) for g in torch.unique(guides_tensor)]))

        # For negbinom, normalize by sum factors; for other distributions, use raw values
        if sum_factor_col is not None:
            y_obs_factored = y_obs_tensor / sum_factor_tensor.view(-1, 1)
        else:
            y_obs_factored = y_obs_tensor

        Vmax_mean_tensor = torch.max(torch.stack([torch.mean(y_obs_factored[guides_tensor == g, :], dim=0) for g in torch.unique(guides_tensor)]), dim=0)[0]
        Amean_tensor = torch.min(torch.stack([torch.mean(y_obs_factored[guides_tensor == g, :], dim=0) for g in torch.unique(guides_tensor)]), dim=0)[0]
        
        #from pyro.infer.autoguide import AutoNormalMessenger
        #from pyro.infer.autoguide.initialization import init_to_value

        def init_loc_fn(site):
            name = site["name"]
        
            if "poly_coeff" in name:
                return torch.zeros(T)
        
            return pyro.infer.autoguide.initialization.init_to_median(site)
        
        if function_type == "polynomial":
            from torch.optim.lr_scheduler import OneCycleLR
            guide_y = pyro.infer.autoguide.AutoMultivariateNormal(self._model_y, init_loc_fn=init_loc_fn)

            guide_y(
                N,
                T,
                y_obs_tensor,
                sum_factor_tensor,
                beta_o_alpha_tensor,
                beta_o_beta_tensor,
                alpha_alpha_mu_tensor,
                gamma_alpha_tensor,
                gamma_beta_tensor,
                K_max_tensor,
                K_alpha_tensor,
                Vmax_mean_tensor,
                Vmax_alpha_tensor,
                n_mu_tensor,
                n_sigma_base_tensor,
                Amean_tensor,
                p_n_tensor,
                threshold,
                slope,
                epsilon_tensor,
                x_true_sample = self.x_true.mean(dim=0) if self.x_true_type == "posterior" else self.x_true,
                log2_x_true_sample = self.log2_x_true.mean(dim=0) if self.log2_x_true_type == "posterior" else self.log2_x_true,
                alpha_y_sample = self.alpha_y_prefit.mean(dim=0) if self.alpha_y_type == "posterior" else self.alpha_y_prefit,
                C = C,
                groups_tensor=groups_tensor,
                predictive_mode=False,
                temperature=torch.tensor(init_temp, dtype=torch.float32, device=self.device),
                use_straight_through=False,
                function_type=function_type,
                polynomial_degree=polynomial_degree,
                use_alpha=True,
                distribution=distribution,
                denominator_tensor=denominator_tensor,
                K=K,
                D=D,
            )

            # 1) Create the base torch optimizer (with gradient clipping built in)
            base_optimizer = torch.optim.Adam(
                guide_y.parameters(),
                lr=1e-3,                   # initial lr (this will be overridden by the scheduler)
                betas=(0.9, 0.999),
            )
            
            # 2) Wrap it in PyroLRScheduler by passing the constructor + kwargs
            optimizer = pyro.optim.PyroLRScheduler(
                # 1) Which scheduler class?
                scheduler_constructor=OneCycleLR,
            
                # 2) Build optim_args with the optimizer *class* + its init args,
                #    then your scheduler’s own kwargs.
                optim_args={
                    # -- for your torch optimizer --
                    "optimizer": torch.optim.Adam,
                    "optim_args": {
                        "lr":     1e-3,          # placeholder, overridden by scheduler
                        "betas": (0.9, 0.999),
                    },
                    # -- for OneCycleLR itself --
                    "max_lr":          1e-2,
                    "total_steps":     niters,
                    "pct_start":       0.1,
                    "div_factor":      25.0,
                    "final_div_factor":1e4,
                },
            
                # 3) (Optional) still want gradient clipping?
                clip_args={"clip_norm": 5}
            )

            #svi   = pyro.infer.SVI(self._model_y, guide_y, optimizer, pyro.infer.Trace_ELBO(num_particles=5, vectorize_particles=True))
            svi   = pyro.infer.SVI(self._model_y, guide_y, optimizer, pyro.infer.Trace_ELBO())
        else:
            guide_y = pyro.infer.autoguide.AutoNormalMessenger(self._model_y)
            optimizer = pyro.optim.Adam({"lr": lr})
            svi = pyro.infer.SVI(self._model_y, guide_y, optimizer, 
                                 loss=pyro.infer.Trace_ELBO())
        
        for name, value in pyro.get_param_store().items():
            if "poly_coeff" in name and "loc" in name:
                print(name, value.shape, value.min().item(), value.max().item())

        self.losses_trans = []
        smoothed_loss = None
        for step in range(niters):
            # a simple linear schedule from init_temp down to final_temp:
            fraction_done = step / float(niters)
            if function_type in ['single_hill', 'additive_hill', 'nested_hill']:
                current_temp = init_temp + (final_temp - init_temp) * fraction_done
            elif function_type == 'polynomial':
                current_temp = init_temp + (final_temp - init_temp) * (2*fraction_done-1)
                #current_temp = init_temp + (final_temp - init_temp) * fraction_done
            else:
                raise ValueError(f"Unknown function_type: {function_type}")
                
            #if step < 0.7 * niters:
            #    # First 70% of training: linearly decrease from 1.0 to 0.1
            #    current_temp = 1.0 - (0.9 * (step / (0.7 * niters)))
            #else:
            #    # Last 30% of training: exponentially cool down to 0.0005
            #    current_temp = 0.1 * (final_temp/0.1) ** ((step - 0.7 * niters) / (0.3 * niters))

            
            samp = torch.randint(high=self.x_true.shape[0], size=(1,)).item()
            # Sample from posterior
            x_true_sample = (
                self.x_true[samp]
                if self.x_true_type == "posterior" else self.x_true
            )
            log2_x_true_sample = (
                self.log2_x_true[samp]
                if self.log2_x_true_type == "posterior" else self.log2_x_true
            )
            alpha_y_sample = (
                self.alpha_y_prefit[samp]
                if self.alpha_y_type == "posterior" else self.alpha_y_prefit
            ) if self.alpha_y_prefit is not None else None

            #use_straight_through = step >= int(0.7 * niters)
            use_straight_through = False
            
            loss = svi.step(
                N,
                T,
                y_obs_tensor,
                sum_factor_tensor,
                beta_o_alpha_tensor,
                beta_o_beta_tensor,
                alpha_alpha_mu_tensor,
                gamma_alpha_tensor,
                gamma_beta_tensor,
                K_max_tensor,
                K_alpha_tensor,
                Vmax_mean_tensor,
                Vmax_alpha_tensor,
                n_mu_tensor,
                n_sigma_base_tensor,
                Amean_tensor,
                p_n_tensor,
                threshold,
                slope,
                epsilon_tensor,
                x_true_sample = x_true_sample,
                log2_x_true_sample = log2_x_true_sample,
                alpha_y_sample = alpha_y_sample,
                C = C,
                groups_tensor=groups_tensor,
                predictive_mode=False,
                temperature=torch.tensor(current_temp, dtype=torch.float32, device=self.device),
                use_straight_through=use_straight_through,
                function_type=function_type,
                polynomial_degree=polynomial_degree,
                use_alpha=True if function_type != 'polynomial' else True if fraction_done>=0.5 else False,
                distribution=distribution,
                denominator_tensor=denominator_tensor,
                K=K,
                D=D,
            )
            
            self.losses_trans.append(loss)
            if step % 1000 == 0:
                print(f"Step {step} : loss = {loss:.5e}, device: {Vmax_mean_tensor.device}")
            if smoothed_loss is None:
                smoothed_loss = loss
            else:
                if abs(alpha_ewma * (loss - smoothed_loss)) < tolerance:
                    print(f"Converged at step {step}! Loss = {loss:.5e}")
                    break
                smoothed_loss = alpha_ewma * loss + (1 - alpha_ewma) * smoothed_loss

        # Move to CPU if using too much GPU memory for Predictive
        run_on_cpu = self.device.type != "cpu"

        if run_on_cpu:
            print("[INFO] Running Predictive on CPU to reduce GPU memory pressure...")
            guide_y.to("cpu")
            self.device = torch.device("cpu")

            # Move inputs to CPU
            model_inputs = {
                "N": N,
                "T": T,
                "y_obs_tensor": y_obs_tensor.cpu(),
                "sum_factor_tensor": sum_factor_tensor.cpu(),
                "beta_o_alpha_tensor": beta_o_alpha_tensor.cpu(),
                "beta_o_beta_tensor": beta_o_beta_tensor.cpu(),
                "alpha_alpha_mu_tensor": alpha_alpha_mu_tensor.cpu(),
                "gamma_alpha_tensor": gamma_alpha_tensor.cpu(),
                "gamma_beta_tensor": gamma_beta_tensor.cpu(),
                "K_max_tensor": K_max_tensor.cpu(),
                "K_alpha_tensor": K_alpha_tensor.cpu(),
                "Vmax_mean_tensor": Vmax_mean_tensor.cpu(),
                "Vmax_alpha_tensor": Vmax_alpha_tensor.cpu(),
                "n_mu_tensor": n_mu_tensor.cpu(),
                "n_sigma_base_tensor": n_sigma_base_tensor.cpu(),
                "Amean_tensor": Amean_tensor.cpu(),
                "p_n_tensor": p_n_tensor.cpu(),
                "threshold": threshold,
                "slope": slope,
                "epsilon_tensor": epsilon_tensor.cpu(),
                "x_true_sample": self.x_true.mean(dim=0).cpu() if self.x_true_type == "posterior" else self.x_true.cpu(),
                "log2_x_true_sample": self.log2_x_true.mean(dim=0).cpu() if self.log2_x_true_type == "posterior" else self.log2_x_true.cpu(),
                "alpha_y_sample": self.alpha_y_prefit.mean(dim=0).cpu() if self.alpha_y_type == "posterior" else (self.alpha_y_prefit.cpu() if self.alpha_y_prefit is not None else None),
                "C": C,
                "groups_tensor": groups_tensor.cpu(),
                "predictive_mode": True,
                "temperature": torch.tensor(final_temp, dtype=torch.float32, device=self.device),
                "use_straight_through": True,
                "function_type": function_type,
                "polynomial_degree": polynomial_degree,
                "use_alpha": True,
                "distribution": distribution,
                "denominator_tensor": denominator_tensor.cpu() if denominator_tensor is not None else None,
                "K": K,
                "D": D,
            }

        else:
            model_inputs = {
                "N": N,
                "T": T,
                "y_obs_tensor": y_obs_tensor,
                "sum_factor_tensor": sum_factor_tensor,
                "beta_o_alpha_tensor": beta_o_alpha_tensor,
                "beta_o_beta_tensor": beta_o_beta_tensor,
                "alpha_alpha_mu_tensor": alpha_alpha_mu_tensor,
                "gamma_alpha_tensor": gamma_alpha_tensor,
                "gamma_beta_tensor": gamma_beta_tensor,
                "K_max_tensor": K_max_tensor,
                "K_alpha_tensor": K_alpha_tensor,
                "Vmax_mean_tensor": Vmax_mean_tensor,
                "Vmax_alpha_tensor": Vmax_alpha_tensor,
                "n_mu_tensor": n_mu_tensor,
                "n_sigma_base_tensor": n_sigma_base_tensor,
                "Amean_tensor": Amean_tensor,
                "p_n_tensor": p_n_tensor,
                "threshold": threshold,
                "slope": slope,
                "epsilon_tensor": epsilon_tensor,
                "x_true_sample": self.x_true.mean(dim=0) if self.x_true_type == "posterior" else self.x_true,
                "log2_x_true_sample": self.log2_x_true.mean(dim=0) if self.log2_x_true_type == "posterior" else self.log2_x_true,
                "alpha_y_sample": self.alpha_y_prefit.mean(dim=0) if self.alpha_y_type == "posterior" else self.alpha_y_prefit,
                "C": C,
                "groups_tensor": groups_tensor,
                "predictive_mode": True,
                "temperature": torch.tensor(final_temp, dtype=torch.float32, device=self.device),
                "use_straight_through": True,
                "function_type": function_type,
                "polynomial_degree": polynomial_degree,
                "use_alpha": True,
                "distribution": distribution,
                "denominator_tensor": denominator_tensor,
                "K": K,
                "D": D,
            }

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        import gc
        gc.collect()

        max_samples = nsamples
        keep_sites = kwargs.get("keep_sites", lambda name, site: site["value"].ndim <= 2 or name != "y_obs")

        if minibatch_size is not None:
            from collections import defaultdict

            print(f"[INFO] Running Predictive in minibatches of {minibatch_size}...")
            predictive_y = pyro.infer.Predictive(
                self._model_y,
                guide=guide_y,
                num_samples=minibatch_size,
                parallel=True
            )
            all_samples = defaultdict(list)
            with torch.no_grad():
                for i in range(0, max_samples, minibatch_size):
                    samples = predictive_y(**model_inputs)
                    for k, v in samples.items():
                        if keep_sites(k, {"value": v}):
                            all_samples[k].append(v.cpu())
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
                    import gc
                    gc.collect()

            posterior_samples_y = {k: torch.cat(v, dim=0) for k, v in all_samples.items()}

        else:
            predictive_y = pyro.infer.Predictive(
                self._model_y,
                guide=guide_y,
                num_samples=nsamples#,
                #parallel=True
            )
            with torch.no_grad():
                posterior_samples_y = predictive_y(**model_inputs)
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                import gc
                gc.collect()

        if run_on_cpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("[INFO] Reset self.device to:", self.device)

        for k, v in posterior_samples_y.items():
            posterior_samples_y[k] = v.cpu()
        self.posterior_samples_trans = posterior_samples_y
        if self.alpha_y_prefit is None and technical_covariates:
            self.alpha_y_prefit = posterior_samples_y["alpha_y"].mean(dim=0)

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        pyro.clear_param_store()

        print("Finished fit_trans.")


#################################
# End of CRISPRBayesianModel class
#################################
