#!/bin/bash -l
#PBS -N 3x3_uvw
#PBS -A USTN0009
#PBS -l select=1:ncpus=4:ngpus=1:mem=80GB
#PBS -l walltime=01:00:00
#PBS -q main
#PBS -l gpu_type=a100
#PBS -m abe
#PBS -M ag4680@stanford.edu
#PBS -o output.log
#PBS -l job_priority=regular

# regular, premium, economy queues on Derecho
module purge
module load ncarenv/23.09 intel-oneapi/2023.2.1 craype/2.7.23 cray-mpich/8.1.27
module load cuda/11.8.0
source ~/nonlocal_gwfluxes/.nlgw/bin/activate

# 3x3_S3 means 3x3 stencil, stratosphere_only and three features: uvtheta
# 5x5_G4 means 5x5 stencil, global, and four features: uvthetaw

# TRAINING
stencil=3
# Usage: python training.py <domain> <vertical> <features> <stencil> <input_file_dir> <torch_model_dir>
python training.py global stratosphere_update uvw $stencil /glade/derecho/scratch/agupta/era5_training_data/ /glade/derecho/scratch/agupta/torch_saved_models/ 

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

