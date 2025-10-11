import sys
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
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--subset", type=str, default=None)
    args = parser.parse_args()

    set_max_threads(args.cores)

    # Set unique random seed per gene
    seed = abs(hash(args.label + args.cis_gene)) % (2**32)
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
            
    label_name = f'{args.cis_gene}' if args.subset is None else f'{args.cis_gene}_{args.subset}'
    model = bayesDREAM(meta=meta, counts=counts, cis_gene=args.cis_gene, cores=args.cores,
                       output_dir=f'{args.outdir}/{args.label}/', label=label_name, random_seed=seed)

    run_name = f"{args.cis_gene}_run" if args.subset is None else f"{args.cis_gene}_run_{args.subset}"

    if not args.subset in ['CRISPRi', 'CRISPRa']:
        model.adjust_ntc_sum_factor(covariates=["lane", "cell_line"])

        alpha_y = torch.load(f'{args.outdir}/{args.label}/alpha_y_prefit.pt')
        model.set_alpha_x(alpha_y[:,:,model.counts.index.values == model.cis_gene].mean(dim=0), is_posterior=False, covariates=["cell_line"])
        model.set_alpha_y(alpha_y[:,:,model.counts.index.values != model.cis_gene].mean(dim=0), is_posterior=False, covariates=["cell_line"])
        torch.save(alpha_y[:,:,model.counts.index.values == model.cis_gene].squeeze(-1), f'{args.outdir}/{args.label}/{run_name}/alpha_x_prefit.pt')
        torch.save(alpha_y[:,:,model.counts.index.values != model.cis_gene], f'{args.outdir}/{args.label}/{run_name}/alpha_y_prefit.pt')
    else:
        model.adjust_ntc_sum_factor(covariates=["lane"])

    if args.subset == 'NTC':
        model.fit_cis(sum_factor_col="sum_factor", tolerance=0, niters=100000)
    else:
        model.fit_cis(sum_factor_col="sum_factor_adj", tolerance=0, niters=100000)

    if "x_obs" in model.posterior_samples_cis:
        del model.posterior_samples_cis["x_obs"]

    torch.save(model.x_true, f'{args.outdir}/{args.label}/{run_name}/x_true.pt')
    torch.save(model.posterior_samples_cis, f'{args.outdir}/{args.label}/{run_name}/posterior_samples_cis.pt')