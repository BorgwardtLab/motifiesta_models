#!/bin/bash
#SBATCH --job-name="motifiesta_eval"
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=1G
#SBATCH --gres=gpu:1
#SBATCH --partition=p.hpcl91
#SBATCH --time=01:00:00
#SBATCH --output slurm_logs/mf_test.o

cd ..
pwd

hostname; date
echo $CUDA_VISIBLE_DEVICES
source .venv/bin/activate

pwd


python experiments/train.py task=enzyme_class pretrained_path=logs/pretrain/alphafold/motif/0/model.pt model.name=motif task.split=random training.debug=true

exit 0
