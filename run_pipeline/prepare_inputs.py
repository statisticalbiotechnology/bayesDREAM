import sys
import os
import pandas as pd
import subprocess

# === Parse args ===
label = sys.argv[1]
outdir = sys.argv[2]

# === User-defined paths (you may choose to make these arguments too) ===
meta_path = '/cfs/klemming/projects/snic/lappalainen_lab1/users/Leah/data/Domingo2024/processed_Leah/domingo_cellmeta.txt.gz'
counts_path = '/cfs/klemming/projects/snic/lappalainen_lab1/users/Leah/data/Domingo2024/processed_Leah/domingo_GEXcounts.csv'
cis_genes = ['GFI1B', 'TET2', 'MYB', 'NFE2']

# === Load and prepare ===
meta = pd.read_csv(meta_path)
meta['guide'] = meta['short_ID']
meta['target'] = meta['gene']
meta['cell'] = meta['L_cell_barcode']
counts = pd.read_csv(counts_path).drop(['CRISPRi', 'CRISPRa'])

#####################################
## Recalculate sum factors using R ##
#####################################
# Save counts_sub and meta_sub to CSV files for R
os.makedirs(f'{outdir}/{label}/', exist_ok=True)
counts.to_csv(f'{outdir}/{label}/{label}_counts_sub_for_R.csv')
meta.to_csv(f'{outdir}/{label}/{label}_meta_sub_for_R.csv', index=False)

# R script as a string
r_script = f"""
suppressPackageStartupMessages(library(scran))
suppressPackageStartupMessages(library(data.table))

# Read the data
counts_sub <- fread("{outdir}/{label}/{label}_counts_sub_for_R.csv", data.table=FALSE)
rownames(counts_sub) <- counts_sub[[1]]
counts_sub[[1]] <- NULL
counts_sub <- as.matrix(counts_sub)

meta_sub <- fread("{outdir}/{label}/{label}_meta_sub_for_R.csv", data.table=FALSE)
my_cell_order <- copy(meta_sub$L_cell_barcode) # probably unnecessary but just to make sure it really is new

# Perform the calculations
myclusts <- as.character(meta_sub$guide_crispr)
myclusts[grepl('NTC', myclusts)] <- 'NTC'
meta_sub$clustered.sum.factor <- calculateSumFactors(counts_sub, clusters=myclusts, ref.clust='NTC')

# Write the Sum Factors to a CSV file
write.table(as.data.table(meta_sub)[match(my_cell_order, L_cell_barcode)]$clustered.sum.factor, file="{outdir}/{label}/{label}_SumFacs_clu.csv", row.names=FALSE, col.names=FALSE, sep=",", quote=FALSE)
"""

# Define the temporary R script filename
tmp_r_script_name = f'{outdir}/{label}/{label}_sum_factor.R'

# Write the R script to the temporary file
with open(tmp_r_script_name, 'w') as tmp_r_script:
    tmp_r_script.write(r_script)

# Execute the R script
subprocess.run(['Rscript', tmp_r_script_name], check=True)

# Read the Sum Factors
meta['sum_factor'] = pd.read_csv(f"{outdir}/{label}/{label}_SumFacs_clu.csv", header=None).squeeze().values

# === Save technical input (NTC only) ===
meta_technical = meta[meta['target'] == 'ntc'].copy()
counts_technical = counts[meta_technical['cell']]
meta_technical.to_csv(f'{outdir}/{label}/meta_technical.csv', index=False)
counts_technical.to_csv(f'{outdir}/{label}/counts_technical.csv')

# === Save per-cis inputs ===
for gene in cis_genes:
    meta_cis = meta[meta['target'].isin(['ntc', gene])].copy()
    counts_cis = counts[meta_cis['cell']]
    meta_cis.to_csv(f'{outdir}/{label}/meta_cis_{gene}.csv', index=False)
    counts_cis.to_csv(f'{outdir}/{label}/counts_cis_{gene}.csv')
