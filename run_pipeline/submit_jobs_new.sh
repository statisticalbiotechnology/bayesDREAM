#!/bin/bash
#set -euo pipefail
# run in pyroenv

############################################
# Defaults (can be overridden by CLI flags)
############################################
start_reps=1
end_reps=1                  # inclusive
RUN_PREPARE=0               # --prepare to enable
RUN_TECH=0                  # --tech to enable
RUN_CIS=0                   # --cis to enable
RUN_TRANS=0                 # --no-trans to disable
RUN_SUBSETS=0               # --subsets to run CRISPRa/CRISPRi; default is full-data
RUN_FULL=0               # --full-data
RUN_EACH_PERMUTATION=0      # --each-permutation to enable per-trans-gene arrays
RUN_FULL_NONE=0           # --full-none to run permtype=none
RUN_FULL_ALL=0            # --full-all  to run permtype=All
ARRAY_MAX=50                # throttle concurrent array tasks: --array-max 50
FUNCTION_TYPES=("additive_hill" "polynomial")

label="run_20250613"
outdir="/cfs/klemming/projects/snic/lappalainen_lab1/users/Leah/data/Domingo2024/processed_Leah/BayesianModel_outs"
cis_genes=("GFI1B" "TET2" "MYB" "NFE2")
cores=1
PRID="naiss2025-5-479"
PARTITION="shared"

BASE_DIR="/cfs/klemming/projects/snic/lappalainen_lab1/users/Leah/CRISPRmodelling/BayesianModelling"
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

############################################
# CLI parsing
############################################
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prepare) RUN_PREPARE=1; shift ;;
    --tech) RUN_TECH=1; shift ;;
    --cis) RUN_CIS=1; shift ;;
    --trans) RUN_TRANS=1; shift ;;
    --subsets) RUN_SUBSETS=1; shift ;;
    --full) RUN_FULL=1; shift ;;
    --full-none) RUN_FULL_NONE=1; shift ;;
    --full-all)  RUN_FULL_ALL=1;  shift ;;
    --each-permutation) RUN_EACH_PERMUTATION=1; shift ;;
    --array-max) ARRAY_MAX="$2"; shift 2 ;;
    --function-types) IFS=',' read -r -a FUNCTION_TYPES <<< "$2"; shift 2 ;;
    --label) label="$2"; shift 2 ;;
    --outdir) outdir="$2"; shift 2 ;;
    --cis-genes) IFS=',' read -r -a cis_genes <<< "$2"; shift 2 ;;
    --cores) cores="$2"; shift 2 ;;
    --account|-A) PRID="$2"; shift 2 ;;
    --partition|-p) PARTITION="$2"; shift 2 ;;
    --start-reps) start_reps="$2"; shift 2 ;;
    --end-reps)   end_reps="$2";   shift 2 ;;
    *) shift ;;
  esac
done

mkdir -p "${outdir}/${label}/logs"

############################################
# Optional: prepare inputs
############################################
jid_prepare=""
if [[ "$RUN_PREPARE" -eq 1 ]]; then
  jid_prepare=$(sbatch -A "$PRID" -p "$PARTITION" --parsable \
    --job-name=prepare_inputs --time=10 --mem=10G \
    --cpus-per-task="$cores" \
    --output="${outdir}/${label}/logs/prepare_inputs.log" \
    --wrap="python prepare_inputs.py ${label} ${outdir}")
  echo "Submitted prepare job: ${jid_prepare}"
fi

for i in $(seq "$start_reps" "$end_reps"); do
  label_i="${label}_$i"
  mkdir -p "${outdir}/${label_i}/logs"

  ##########################################
  # Optional: technical fit
  ##########################################
  jid_tech=""
  if [[ "$RUN_TECH" -eq 1 ]]; then
    dep_tech=""
    [[ -n "$jid_prepare" ]] && dep_tech="--dependency=afterok:${jid_prepare}"
    jid_tech=$(sbatch -A "$PRID" -p "$PARTITION" --parsable $dep_tech \
      --job-name="tech_${i}" --time=60 --mem=10G \
      --cpus-per-task="$cores" \
      --output="${outdir}/${label_i}/logs/tech_fit.log" \
      --wrap="PYTHONPATH=${PYTHONPATH} python run_technical.py --inlabel ${label} --label ${label_i} --outdir ${outdir} --cores ${cores}")
    echo "Submitted tech job: ${jid_tech}"
  fi

  ##########################################
  # Per-cis gene work (CIS + TRANS)
  ##########################################
  for gene in "${cis_genes[@]}"; do
    mkdir -p "${outdir}/${label_i}/logs"

    # Optional: CIS fits (only needed if you choose to run them)
    jid_cis=""
    if [[ "$RUN_CIS" -eq 1 ]]; then
      dep_cis=""
      [[ -n "$jid_tech" ]] && dep_cis="--dependency=afterok:${jid_tech}"
      jid_cis=$(sbatch -A "$PRID" -p "$PARTITION" --parsable $dep_cis \
        --job-name="cis_${gene}_${i}" --time=120 --mem=10G \
        --cpus-per-task="$cores" \
        --output="${outdir}/${label_i}/logs/cis_${gene}.log" \
        --wrap="PYTHONPATH=${PYTHONPATH} python run_cis.py --inlabel ${label} --label ${label_i} --outdir ${outdir} --cis_gene ${gene} --cores ${cores}")
      echo "Submitted CIS job: ${jid_cis} (${gene})"
    fi

    # ============================
    # TRANS: FULL DATA (controlled by flags)
    # ============================
    if [[ "$RUN_TRANS" -eq 1 && "$RUN_FULL" -eq 1 ]]; then
      dep_trans=""
      [[ -n "$jid_cis" ]] && dep_trans="--dependency=afterok:${jid_cis}"
    
      # Back-compat default: if neither sub-flag set and no individual arrays,
      # run both "none" and "All" like before.
      if [[ "$RUN_FULL_NONE" -eq 0 && "$RUN_FULL_ALL" -eq 0 && "$RUN_EACH_PERMUTATION" -eq 0 ]]; then
        RUN_FULL_NONE=1
        RUN_FULL_ALL=1
      fi
    
      for function_type in "${FUNCTION_TYPES[@]}"; do
        if [[ "$function_type" == "polynomial" ]]; then
          TIME_REQ=600; MEM_REQ=50G
        else
          TIME_REQ=300; MEM_REQ=20G
        fi
    
        # Run permtype=none if requested
        if [[ "$RUN_FULL_NONE" -eq 1 ]]; then
          sbatch -A "$PRID" -p "$PARTITION" $dep_trans \
            --job-name="trans_${gene}_none_${function_type}_${i}" \
            --time="$TIME_REQ" --mem="$MEM_REQ" \
            --cpus-per-task="$cores" \
            --output="${outdir}/${label_i}/logs/trans_${gene}_none_${function_type}.log" \
            --wrap="PYTHONPATH=${PYTHONPATH} python run_trans.py \
              --inlabel ${label} --label ${label_i} --outdir ${outdir} \
              --cis_gene ${gene} --permtype none \
              --function_type ${function_type} --cores ${cores}"
        fi
    
        # Run permtype=All if requested
        if [[ "$RUN_FULL_ALL" -eq 1 ]]; then
          sbatch -A "$PRID" -p "$PARTITION" $dep_trans \
            --job-name="trans_${gene}_All_${function_type}_${i}" \
            --time="$TIME_REQ" --mem="$MEM_REQ" \
            --cpus-per-task="$cores" \
            --output="${outdir}/${label_i}/logs/trans_${gene}_All_${function_type}.log" \
            --wrap="PYTHONPATH=${PYTHONPATH} python run_trans.py \
              --inlabel ${label} --label ${label_i} --outdir ${outdir} \
              --cis_gene ${gene} --permtype All \
              --function_type ${function_type} --cores ${cores}"
        fi
    
        # Individual arrays stay as-is, keyed by RUN_EACH_PERMUTATION
        if [[ "$RUN_EACH_PERMUTATION" -eq 1 ]]; then
          trans_list_file="${outdir}/${label_i}/trans_list_${gene}.txt"
          python - <<PY
import pandas as pd
counts = pd.read_csv("${outdir}/${label}/counts_cis_${gene}.csv", index_col=0)
genes = [g for g in counts.index if g != "${gene}"]
open("${outdir}/${label_i}/trans_list_${gene}.txt","w").write("\n".join(genes))
PY
          N=$(wc -l < "${trans_list_file}")
          if [[ "$N" -gt 0 ]]; then
            sbatch -A "$PRID" -p "$PARTITION" $dep_trans \
              --array=0-$((N-1))%${ARRAY_MAX} \
              --job-name="trans_${gene}_each_${function_type}_${i}" \
              --time="$TIME_REQ" --mem="$MEM_REQ" \
              --cpus-per-task="$cores" \
              --output="${outdir}/${label_i}/logs/trans_${gene}_each_${function_type}_%a.tmp.log" \
              --wrap="PYTHONPATH=${PYTHONPATH} bash -c '
                set -euo pipefail
                trans_gene=\$(sed -n \"\$((SLURM_ARRAY_TASK_ID+1))p\" ${trans_list_file})
                exec > \"${outdir}/${label_i}/logs/trans_${gene}_${function_type}_${i}_\${trans_gene}.log\" 2>&1
                echo \"[INFO] cis=${gene} function_type=${function_type} array_id=\$SLURM_ARRAY_TASK_ID trans_gene=\$trans_gene\"
                python run_trans.py --inlabel ${label} --label ${label_i} --outdir ${outdir} \
                  --cis_gene ${gene} --permtype \"\${trans_gene}\" \
                  --function_type ${function_type} --cores ${cores}
              '"
          else
            echo "No trans genes found for ${gene}; skipping each-permutation arrays."
          fi
        fi
      done
    fi

    # ==========================================
    # SUBSETS (CRISPRa / CRISPRi)
    # ==========================================
    if [[ "$RUN_SUBSETS" -eq 1 ]]; then
      for subset in CRISPRa CRISPRi; do
        jid_cis_subset=""
        if [[ "$RUN_CIS" -eq 1 ]]; then
          dep_cis_subset=""
          [[ -n "$jid_tech" ]] && dep_cis_subset="--dependency=afterok:${jid_tech}"
          jid_cis_subset=$(sbatch -A "$PRID" -p "$PARTITION" --parsable $dep_cis_subset \
            --job-name="cis_${gene}_${subset}_${i}" --time=120 --mem=10G \
            --cpus-per-task="$cores" \
            --output="${outdir}/${label_i}/logs/cis_${gene}_${subset}.log" \
            --wrap="PYTHONPATH=${PYTHONPATH} python run_cis.py --inlabel ${label} --label ${label_i} --outdir ${outdir} --cis_gene ${gene} --cores ${cores} --subset ${subset}")
          echo "Submitted subset CIS job: ${jid_cis_subset} (${gene} / ${subset})"
        fi

        # Only run subset trans if RUN_TRANS is also enabled
        if [[ "$RUN_TRANS" -eq 1 ]]; then
          dep_trans_subset=""
          [[ -n "$jid_cis_subset" ]] && dep_trans_subset="--dependency=afterok:${jid_cis_subset}"

          for function_type in "${FUNCTION_TYPES[@]}"; do
            if [[ "$function_type" == "polynomial" ]]; then
              TIME_REQ=600; MEM_REQ=50G
            else
              TIME_REQ=300; MEM_REQ=20G
            fi

            sbatch -A "$PRID" -p "$PARTITION" $dep_trans_subset \
              --job-name="trans_${gene}_${subset}_${function_type}_${i}" \
              --time="$TIME_REQ" --mem="$MEM_REQ" \
              --cpus-per-task="$cores" \
              --output="${outdir}/${label_i}/logs/trans_${gene}_${subset}_${function_type}.log" \
              --wrap="PYTHONPATH=${PYTHONPATH} python run_trans.py --inlabel ${label} --label ${label_i} --outdir ${outdir} --cis_gene ${gene} --permtype none --function_type ${function_type} --cores ${cores} --subset ${subset}"
          done
        fi
      done
    fi

  done
done
