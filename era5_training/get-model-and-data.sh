#!/usr/bin/env sh

mkdir -p model-huggingface
mkdir -p inputs

echo "retrieving model weights..."
cd model-huggingface
wget https://huggingface.co/amangupta2/iccs_coupling_checkpoints/resolve/main/retrained_ann_cnn_1x1_global_global_era5_uvthetaw__train_epoch45.pt
wget https://huggingface.co/amangupta2/iccs_coupling_checkpoints/resolve/main/ann_cnn_1x1_global_global_era5_uvthetaw__train_epoch94.pt
wget https://huggingface.co/amangupta2/iccs_coupling_checkpoints/resolve/main/attnunet_era5_global_global_uvthetaw_mseloss_train_epoch119.pt
cd ..

mv model-huggingface/retrained_ann_cnn_1x1_global_global_era5_uvthetaw__train_epoch45.pt model-huggingface/ann_cnn_1x1_global_global_era5_uvthetaw__train_epoch45.pt

echo "retrieving test input..."
(cd inputs && wget https://g-b56e81.7a577b.6fbd.data.globus.org/1x1_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling01.nc)
