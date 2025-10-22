#!/bin/bash
#set -euo pipefail
# run in pyroenv

start_reps=1
end_reps=5                # number of times to repeat (inclusive; 2-20 or 1-1)
run_trans=TRUE          # set to TRUE to run trans fits as well

label="run_20250613"
outdir="/cfs/klemming/projects/snic/lappalainen_lab1/users/Leah/data/Domingo2024/processed_Leah/BayesianModel_outs"
#cis_genes=("GFI1B" "TET2" "MYB" "NFE2")
cis_genes=("TET2")
cores=1

BASE_DIR="/cfs/klemming/projects/snic/lappalainen_lab1/users/Leah/CRISPRmodelling/BayesianModelling"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"
PRID="naiss2024-5-581"

mkdir -p "${outdir}/${label}/logs"

# 0. Submit input preparation job
#jid_prepare=$(sbatch -A $PRID -p shared --parsable --job-name=prepare_inputs --time=10 --mem=10G \
#  --cpus-per-task=$cores \
#  --output=${outdir}/${label}/logs/prepare_inputs.log \
#  --wrap="python prepare_inputs.py ${label} ${outdir}")

for i in $(seq $start_reps $end_reps); do
  label_i="${label}_$i"

  # Submit technical fit
  #jid_tech=$(sbatch -A $PRID -p shared --parsable --dependency=afterok:${jid_prepare} --job-name=tech_${i} --time=60 --mem=10G \
  #  --cpus-per-task=$cores \
  #  --output=${outdir}/${label_i}/logs/tech_fit.log \
  #  --wrap="PYTHONPATH=${PYTHONPATH} python run_technical.py --inlabel ${label} --label ${label_i} --outdir ${outdir} --cores $cores")

  #for gene in "${cis_genes[@]}"; do
  for gene in ""; do
    mkdir -p "${outdir}/${label_i}/${gene}_run"
    mkdir -p "${outdir}/${label_i}/logs"

    # Submit cis fit
    #jid_cis=$(sbatch -A $PRID -p shared --parsable --job-name=cis_${gene}_${i} --time=120 --mem=10G \
    jid_cis=$(sbatch -A $PRID -p shared --parsable --dependency=afterok:${jid_tech} --job-name=cis_${gene}_${i} --time=120 --mem=10G \
      --cpus-per-task=$cores \
      --output=${outdir}/${label_i}/logs/cis_${gene}.log \
      --wrap="PYTHONPATH=${PYTHONPATH} python run_cis.py --inlabel ${label} --label ${label_i} --outdir ${outdir} --cis_gene ${gene} --cores $cores")

    if [ "$run_trans" = "TRUE" ]; then

      for permtype in none All; do
        sbatch -A $PRID -p shared --dependency=afterok:${jid_cis} --job-name=trans_${gene}_${permtype}_${i} --time=300 --mem=20G \
          --cpus-per-task=$cores \
          --output=${outdir}/${label_i}/logs/trans_${gene}_${permtype}.log \
          --wrap="PYTHONPATH=${PYTHONPATH} python run_trans.py --inlabel ${label} --label ${label_i} --outdir ${outdir} --cis_gene ${gene} --permtype ${permtype} --cores $cores"
      done

      sbatch -A $PRID -p shared --dependency=afterok:${jid_cis} --job-name=trans_${gene}_each_${i} --time=60 --mem=2G \
        --cpus-per-task=$cores \
        --output=${outdir}/${label_i}/logs/trans_${gene}_each.log \
        --wrap="PYTHONPATH=${PYTHONPATH} bash -c '
          for trans_gene in \$(python -c \"import pandas as pd; counts = pd.read_csv(\\\"${outdir}/${label}/counts_cis_${gene}.csv\\\", index_col=0); print(\\\" \\\".join([g for g in counts.index if g != \\\"${gene}\\\"]))\"); do
            sbatch -A $PRID -p shared --job-name=trans_${gene}_\${trans_gene}_${i} --time=300 --mem=20G \
              --cpus-per-task=$cores \
              --output=${outdir}/${label_i}/logs/trans_${gene}_\${trans_gene}.log \
              --wrap=\"PYTHONPATH=${PYTHONPATH} python run_trans.py --inlabel ${label} --label ${label_i} --outdir ${outdir} --cis_gene ${gene} --permtype \${trans_gene} --cores $cores\"
          done'"
    fi
  done

  for gene in "${cis_genes[@]}"; do
      for subset in NTC CRISPRa CRISPRi; do
        mkdir -p "${outdir}/${label_i}/${gene}_run_${subset}"
        mkdir -p "${outdir}/${label_i}/logs"
    
        # Submit cis fit; when subsetting to NTC you need the technical fit
        #jid_cis=$(sbatch -A $PRID -p shared --parsable --dependency=afterok:${jid_tech} --job-name=cis_${gene}_${subset}_${i} --time=120 --mem=10G \
        jid_cis=$(sbatch -A $PRID -p shared --parsable --job-name=cis_${gene}_${subset}_${i} --time=120 --mem=10G \
          --cpus-per-task=$cores \
          --output=${outdir}/${label_i}/logs/cis_${gene}_${subset}.log \
          --wrap="PYTHONPATH=${PYTHONPATH} python run_cis.py --inlabel ${label} --label ${label_i} --outdir ${outdir} --cis_gene ${gene} --cores $cores --subset ${subset}")
    
        if [ "$run_trans" = "TRUE" ]; then
        
            sbatch -A $PRID -p shared --dependency=afterok:${jid_cis} --job-name=trans_${gene}_${subset}_${i} --time=300 --mem=20G \
              --cpus-per-task=$cores \
              --output=${outdir}/${label_i}/logs/trans_${gene}_${subset}.log \
              --wrap="PYTHONPATH=${PYTHONPATH} python run_trans.py --inlabel ${label} --label ${label_i} --outdir ${outdir} --cis_gene ${gene} --permtype none --cores $cores --subset ${subset}"
        fi
    done
  done
done




















