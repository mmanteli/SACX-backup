#!/bin/bash
#SBATCH --job-name=kws
#SBATCH --account=project_462000353
#SBATCH --time=00:30:00
#SBATCH --partition=debug
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=15G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

rm -f logs/current_${SLURM_JOB_NAME}.err
rm -f logs/current_${SLURM_JOB_NAME}.out
ln -s $SLURM_JOBID.err logs/current_${SLURM_JOB_NAME}.err
ln -s $SLURM_JOBID.out logs/current_${SLURM_JOB_NAME}.out

data_name="hplt"
lang=$1
case $lang in
  "zh")
    data="/scratch/project_462000353/amanda/keywords-zh/explanations/${data_name}/*/${lang}/exp_123_${lang}_shard*.tsv"
  ;;
  *)
    data="/scratch/project_462000353/amanda/keywords/SACX-backup/explanations/${data_name}/*/${lang}/exp_123_${lang}_shard*.tsv"
  ;;
esac


module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.4
export PYTHONPATH=/scratch/project_462000353/amanda/keywords/PYTHONUSERBASE/lib/python3.10/site-packages:$PYTHONPATH
#source .venv/bin/activate


srun python kws.py \
  --data=$data \
  --save_file "keywords/${data_name}-bge/${lang}/${lang}_"

seff $SLURM_JOBID


