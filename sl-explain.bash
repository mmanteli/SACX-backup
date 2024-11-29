#!/bin/bash
#SBATCH --job-name=explain
#SBATCH --account=project_462000353
#SBATCH --time=3:00:00
#SBATCH --partition=small-g
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10G
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

rm -f logs/current_${SLURM_JOBNAME}.err
rm -f logs/current_${SLURM_JOBNAME}.out
ln -s $SLURM_JOBID.err logs/current_${SLURM_JOBNAME}.err
ln -s $SLURM_JOBID.out logs/current_${SLURM_JOBNAME}.out

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.4
source .venv/bin/activate

srun python explain_multilabel.py \
  --seed $1 \
  --downsample 20 \
  --language '["en","fr"]' \
  --save_reports tests/en-fr/rep \
  --save_file tests/en-fr/exp

seff $SLURM_JOBID
