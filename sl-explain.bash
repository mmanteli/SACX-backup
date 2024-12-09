#!/bin/bash
#SBATCH --job-name=zh-explain
#SBATCH --account=project_462000353
#SBATCH --time=00:15:00
#SBATCH --partition=dev-g
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:mi250:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

rm -f logs/current_${SLURM_JOB_NAME}.err
rm -f logs/current_${SLURM_JOB_NAME}.out
ln -s $SLURM_JOBID.err logs/current_${SLURM_JOB_NAME}.err
ln -s $SLURM_JOBID.out logs/current_${SLURM_JOB_NAME}.out

fold=$1
data="hplt"

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.4
export PYTHONPATH=/scratch/project_462000353/amanda/keywords/PYTHONUSERBASE/lib/python3.10/site-packages:$PYTHONPATH
#source .venv/bin/activate


srun python explain-only.py \
  --base_model "xlm-roberta-large" \
  --trained_model /scratch/project_462000353/amanda/register-clustering/data/models/folds_improved/fold_${fold} \
  --labels '["MT","LY","SP","ID","NA","HI","IN","OP","IP"]' \
  --data_path "/scratch/project_462000353/amanda/register-clustering/data/datasets/${data}/" \
  --data_type "local_huggingface" \
  --language '["zh"]' \
  --parse_separately "zh" \
  --downsample 80 \
  --parser_model "zh_core_web_md" \
  --save_file explanations/${data}/${fold}/ur/exp

seff $SLURM_JOBID


#zh_core_web_md
#--parse_separately "zh" \
# --downsample 80 \
#--parser_model "zh_core_web_md" \