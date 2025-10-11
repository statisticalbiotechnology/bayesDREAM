import os
import argparse
import pandas as pd
import torch
import pyro
import numpy as np
from bayesDREAM.model import bayesDREAM
import warnings

def set_max_threads(cores: int):
    os.environ["OMP_NUM_THREADS"] = str(cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cores)
    os.environ["MKL_NUM_THREADS"] = str(cores)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cores)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cores)
    torch.set_num_threads(cores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--inlabel", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cis_gene", required=True)
    parser.add_argument("--permtype", required=True)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--function_type", type=str, default=None)
    args = parser.parse_args()

    set_max_threads(args.cores)

    seed = abs(hash(args.label + args.cis_gene + args.permtype)) % (2**32)
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    meta = pd.read_csv(f'{args.outdir}/{args.inlabel}/meta_cis_{args.cis_gene}.csv')
    counts = pd.read_csv(f'{args.outdir}/{args.inlabel}/counts_cis_{args.cis_gene}.csv', index_col=0)
    
    if args.subset is not None:
        if args.subset == 'NTC':
            mycells = meta.loc[meta['target']=='ntc','cell'].values
            meta = meta.loc[meta['target']=='ntc',:]
            counts = counts.loc[:,mycells]
        elif args.subset == 'CRISPRa':
            mycells = meta.loc[meta['cell_line']=='CRISPRa','cell'].values
            meta = meta.loc[meta['cell_line']=='CRISPRa',:]
            counts = counts.loc[:,mycells]
        elif args.subset == 'CRISPRi':
            mycells = meta.loc[meta['cell_line']=='CRISPRi','cell'].values
            meta = meta.loc[meta['cell_line']=='CRISPRi',:]
            counts = counts.loc[:,mycells]
        else:
            warnings.warn("Unknown subset, ignoring.")

    label_name = f'{args.cis_gene}_{args.function_type}_perm_{args.permtype}' if args.subset is None else f'{args.cis_gene}_{args.function_type}_perm_{args.permtype}_{args.subset}'
    model = bayesDREAM(meta=meta, counts=counts, cis_gene=args.cis_gene, cores=args.cores,
                       output_dir=f'{args.outdir}/{args.label}/', label=label_name, random_seed=seed)
    
    run_name = f"{args.cis_gene}_run" if args.subset is None else f"{args.cis_gene}_run_{args.subset}"
    
    if not args.subset in ['CRISPRi', 'CRISPRa']:
        model.set_alpha_x(torch.load(f'{args.outdir}/{args.label}/{run_name}/alpha_x_prefit.pt').mean(dim=0).unsqueeze(-1), is_posterior=False, covariates=["cell_line"])
        model.set_alpha_y(torch.load(f'{args.outdir}/{args.label}/{run_name}/alpha_y_prefit.pt').mean(dim=0), is_posterior=False, covariates=["cell_line"])
        model.adjust_ntc_sum_factor(covariates=["lane", "cell_line"])
    model.set_x_true(torch.load(f'{args.outdir}/{args.label}/{run_name}/x_true.pt').mean(dim=0), is_posterior=False)
    model.posterior_samples_cis = torch.load(f'{args.outdir}/{args.label}/{run_name}/posterior_samples_cis.pt')
    
    if args.subset is not None:
        if args.permtype != 'none':
            raise ValueError('subsetting only implemented for when permtype = none')
    else:
        if args.permtype == 'All':
            model.permute_genes(genes2permute='All')
        elif args.permtype != 'none':
            model.permute_genes(genes2permute=args.permtype)

    if args.subset == 'NTC':
        model.fit_trans(sum_factor_col="sum_factor", function_type=args.function_type, gamma_alpha=0.5, gamma_beta=9.5, tolerance=0, niters=(100000 if args.function_type != 'polynomial' else 200000))
    else:
        if not args.subset in ['CRISPRi', 'CRISPRa']:
            model.refit_sumfactor(covariates=["lane", "cell_line"])
        else:
            model.refit_sumfactor(covariates=["lane"])
        model.fit_trans(sum_factor_col="sum_factor_new", function_type=args.function_type, gamma_alpha=0.5, gamma_beta=9.5, tolerance=0, niters=(100000 if args.function_type != 'polynomial' else 200000))

    if "y_obs" in model.posterior_samples_trans:
        del model.posterior_samples_trans["y_obs"]

    torch.save(model.posterior_samples_trans, f'{args.outdir}/{args.label}/{run_name}/posterior_samples_trans_{args.function_type}_{args.permtype}.pt')
