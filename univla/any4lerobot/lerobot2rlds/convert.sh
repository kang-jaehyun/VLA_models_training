#!/bin/bash
#SBATCH --job-name=inference_pi0fast
#SBATCH --output=tmp/slurm-%j-%x.log
#SBATCH --partition=batch
#SBATCH --gpus=1

### for debugging, you can run the following command to get a shell in the container:
# srun --gpus=1 --nodes=1 --pty /bin/bash

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate conversion_env

python lerobot2rlds.py \
    --src-dir /virtual_lab/rlwrld/david/pi_0_fast/openpi/data/rlwrld_dataset/allex-cube-dataset \
    --output-dir /virtual_lab/rlwrld/david/pi_0_fast/openpi/data/rlwrld_dataset \
    --task-name allex-cube-dataset