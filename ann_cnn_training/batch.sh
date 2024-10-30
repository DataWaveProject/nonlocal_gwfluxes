#!/bin/bash
#SBATCH --job-name=3x3_uvw
#SBATCH --partition=serc
#SBATCH -c 10
#SBATCH -G 1
#SBATCH --gpus-per-node=1
#SBATCH --time=96:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=gpu_slurm-%j.out
#SBATCH -C GPU_MEM:80GB
#SBATCH --mem-per-cpu=5GB

# from Mark:
# use sh_node_feat -p serc (or gpu) to see the node structure of the partition and what GPUs are available
# -c indicates cpu_per_task
# -G is the number of GPUs you want to request
# -p is the partition
# requesting SBATCH -G 4 || AND || --gpus-per-node=4 allocated 4 GPUs within a single node
# if you want the distributed over two nodes, do: -G 4 || --gpus-per-node=2

# for more information on GPUs on sherlock: https://www.sherlock.stanford.edu/docs/user-guide/gpu/#gpu-types

source /home/groups/aditis2/ag4680/miniconda3/etc/profile.d/conda.sh
conda activate siv2

# 3x3_S3 means 3x3 stencil, stratosphere_only and three features: uvtheta
# 5x5_G4 means 5x5 stencil, global, and four features: uvthetaw

# TRAINING
stencil=3
# Usage: python training.py <domain> <vertical> <features> <stencil>
python training.py global global uvw $stencil


# INFERENCE
# Usage: python inference.py <domain> <vertical> <features> <epoch_no> <month> <stencil>
#for month in 1 2 3 4 5 6 7 8 9 10 11 12;
#do
#	python inference.py global stratosphere_update uvtheta 100 $month 1
#	python inference.py global stratosphere_update uvw 100 $month 1
#	python inference.py global stratosphere_update uvtheta 68 $month 3
#done

#python inference_ifs.py global stratosphere_update uvtheta 100 12 1
#python inference_ifs.py global stratosphere_update uvw 100 12 1
#python inference_ifs.py global stratosphere_update uvtheta 68 12 3


# submit the same script for 1x1 uvthetaw, 3x3 uvthetaw, and 3x3 uvw when finished

