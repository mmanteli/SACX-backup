#!/bin/bash
#SBATCH --job-name=en-fr
#SBATCH --account=project_2002026
#SBATCH --time=3:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=7G
#SBATCH --gres=gpu:v100:1,nvme:5
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/%x%j.out
#SBATCH -e logs/%x%j.err

rm logs/current.err
rm logs/current.out
ln -s $SLURM_JOBID.err logs/current.err
ln -s $SLURM_JOBID.out logs/current.out

module purge
module load pytorch

srun python train_and_explain.py \
  --seed $1 \
  --downsample 20 \
  --language '["en","fr"]' \
  --save_reports tests/en-fr/rep \
  --save_file tests/en-fr/exp

seff $SLURM_JOBID
