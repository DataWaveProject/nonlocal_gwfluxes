#!/bin/bash
#SBATCH --job-name=TL_3x3
#SBATCH --partition=serc
#SBATCH -c 10
#SBATCH -G 1
#SBATCH --gpus-per-node=1
#SBATCH --time=72:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=gpu_slurm-%j.out
#SBATCH -C GPU_MEM:40GB
#SBATCH --mem-per-cpu=10GB

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

python training_ifs_transfer_learning.py \
	-m ann \
	-s 1 \
	-d global \
	-v stratosphere_only \
	-f uvtheta \
	-e 93 \
	-i /path/to/ifs/fluxes/ \
        -c /glade/derecho/scratch/agupta/torch_saved_models/ \
        -o /glade/derecho/scratch/agupta/gw_inference_files/



python ANN_inference.py \
	-s 1 \
	-d global \
	-v global \
	-f uvtheta \
	-e 200 \
	-t ERA5 \
	-m 1 \
	-i /glade/derecho/scratch/agupta/era5_training_data/ \
        -c /glade/derecho/scratch/agupta/torch_saved_models/ \
        -o /glade/derecho/scratch/agupta/torch_saved_models/




python attn_inference.py \
	-d horizontal \
	-v global \
	-f uvtheta \
	-e 200 \
	-t IFS \
	-m 1 \
	-i /glade/derecho/scratch/agupta/era5_training_data/ \
        -c /glade/derecho/scratch/agupta/torch_saved_models/ \
        -o /glade/derecho/scratch/agupta/torch_saved_models/


# 'attention/ann' 'global'(horizontal) 'global'/'stratosphere_only'(vertical) and 'feature_set', 'CHECKPOINT_EPOCH'
# TRAINING - ATTENTION
#python training_ifs_transfer_learning.py attention global global uvtheta 110
#python training_ifs_transfer_learning.py attention global global uvthetaw 119

#python training_ifs_transfer_learning.py attention global stratosphere_only uvtheta 119
#python training_ifs_transfer_learning.py attention global stratosphere_only uvthetaw 105

#python training_ifs_transfer_learning.py attention global stratosphere_update uvtheta 131
#python training_ifs_transfer_learning.py attention global stratosphere_update uvthetaw 119
#python training_ifs_transfer_learning.py attention global stratosphere_update uvw 119


# TRAINING - ANN_CNN
# 'attention/ann' 'global'(horizontal) 'global'/'stratosphere_only'(vertical) and 'feature_set', 'CHECKPOINT_EPOCH', <stencil>
# 1x1
#python training_ifs_transfer_learning.py ann global global uvtheta 94 1
#python training_ifs_transfer_learning.py ann global global uvthetaw 94 1

#python training_ifs_transfer_learning.py ann global stratosphere_only uvtheta 88 1
#python training_ifs_transfer_learning.py ann global stratosphere_only uvthetaw 100 1

#python training_ifs_transfer_learning.py ann global stratosphere_update uvtheta 100 1
###python training_ifs_transfer_learning.py ann global stratosphere_update uvthetaw XXX 1
#python training_ifs_transfer_learning.py ann global stratosphere_update uvw 100 1


# 3x3
#python training_ifs_transfer_learning.py ann global global uvtheta 52 3
#python training_ifs_transfer_learning.py ann global global uvthetaw 80 3

#python training_ifs_transfer_learning.py ann global stratosphere_only uvtheta 93 3
#python training_ifs_transfer_learning.py ann global stratosphere_only uvthetaw 38 3

#python training_ifs_transfer_learning.py ann global stratosphere_update uvtheta 68 3
###python training_ifs_transfer_learning.py ann global stratosphere_update uvthetaw XXX 3
###python training_ifs_transfer_learning.py ann global stratosphere_update uvw 100 3
