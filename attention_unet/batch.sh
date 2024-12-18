#!/bin/bash -l
#PBS -N unet_uvw
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



# 'global'/'stratosphere_only'/'stratosphere_update' and 'feature_set'
# TRAINING
#python training_attention_unet.py global uvw /glade/derecho/scratch/agupta/era5_training_data/ /glade/derecho/scratch/agupta/torch_saved_models/
#python training_attention_unet.py stratosphere_only uvthetawN2


python training.py \
        -d global  \
        -v stratosphere_update \
        -f uvw \
        -i /glade/derecho/scratch/agupta/era5_training_data/ \
        -o /glade/derecho/scratch/agupta/torch_saved_models/

#for month in 1 2 3 4 5 6 7 8 9 10 11 12;
#do
#	python inference.py global uvtheta 110 $month
#	python inference.py global uvthetaw 119 $month
	
#	python inference.py stratosphere_only uvtheta 119 $month
#	python inference.py stratosphere_only uvthetaw 105 $month

#	python inference.py stratosphere_update uvtheta 131 $month
#	python inference.py stratosphere_update uvthetaw 119 $month
#	python inference.py stratosphere_update uvw 119 $month
#done


#python inference.py \
#       -v stratosphere_update \
#       -f uvw \
# 	-e 100 \
# 	-m 1 \
#       -i /glade/derecho/scratch/agupta/era5_training_data/ \
#	-c /glade/derecho/scratch/agupta/torch_saved_models/ \
#       -o /glade/derecho/scratch/agupta/gw_inference_files/


#python inference_ifs.py global uvw 150 1

#python inference_ifs.py \
#       -v stratosphere_update \
#       -f uvw \
#       -e 100 \
#       -i /glade/derecho/scratch/agupta/era5_training_data/ \
#       -c /glade/derecho/scratch/agupta/torch_saved_models/ \
#       -o /glade/derecho/scratch/agupta/gw_inference_files/


# INFERENCE
# Most optimal epochs to use for respective configs
# stratosphere_only | uvtheta  | epoch=100 | month
# stratosphere_only | uvthetaw | epoch=100 | month
# Usage: python inference.py <vertical> <features> <epoch> <month>
#python probabilistic_inference.py stratosphere_only uvthetaw 100 $month 1
#python probabilistic_inference.py stratosphere_only uvtheta 100 $month 1



