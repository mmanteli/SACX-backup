#!/bin/bash
#SBATCH --job-name=ext_kws
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

rm -f logs/current.err
rm -f logs/current.out
ln -s $SLURM_JOBID.err logs/current.err
ln -s $SLURM_JOBID.out logs/current.out

module purge
module use /appl/local/csc/modulefiles
module load pytorch/2.4
export PYTHONPATH=/scratch/project_462000353/amanda/keywords/PYTHONUSERBASE/lib/python3.10/site-packages:$PYTHONPATH
#source .venv/bin/activate

srun python train_and_explain.py \
  --seed $1 \
  --downsample 20 \
  --language '["en","fr"]' \
  --save_reports tests/en-fr/rep \
  --save_file tests/en-fr/exp

seff $SLURM_JOBID
