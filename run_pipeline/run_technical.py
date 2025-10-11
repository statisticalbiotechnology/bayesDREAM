import os
import argparse
import pandas as pd
import torch
import pyro
import numpy as np
from bayesDREAM.model import bayesDREAM

def set_max_threads(cores: int):
    os.environ["OMP_NUM_THREADS"] = str(cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cores)
    os.environ["MKL_NUM_THREADS"] = str(cores)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(cores)
    os.environ["NUMEXPR_NUM_THREADS"] = str(cores)
    torch.set_num_threads(cores)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inlabel", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--cores", type=int, default=1)
    args = parser.parse_args()

    set_max_threads(args.cores)

    # Seed from label
    seed = abs(hash(args.label + "_technical")) % (2**32)
    pyro.set_rng_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    meta = pd.read_csv(f'{args.outdir}/{args.inlabel}/meta_technical.csv')
    counts = pd.read_csv(f'{args.outdir}/{args.inlabel}/counts_technical.csv', index_col=0)

    model = bayesDREAM(meta=meta, counts=counts, output_dir=f'{args.outdir}/{args.label}/', label=args.label, random_seed=seed, cores=args.cores)
    model.fit_technical(covariates=["cell_line"], tolerance=0)

    torch.save(model.alpha_y_prefit, f'{args.outdir}/{args.label}/alpha_y_prefit.pt')
    if "y_obs_ntc" in model.posterior_samples_technical:
        del model.posterior_samples_technical["y_obs_ntc"]
    torch.save(model.posterior_samples_technical, f'{args.outdir}/{args.label}/posterior_samples_technical.pt')
