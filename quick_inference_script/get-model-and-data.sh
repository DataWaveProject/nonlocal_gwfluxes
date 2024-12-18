#!/usr/bin/env sh

mkdir -p model-huggingface
mkdir -p inputs

echo "retrieving model weights..."
(cd model-huggingface && wget https://huggingface.co/amangupta2/nonlocal_gwfluxes/resolve/main/ANN_1x1/ann_cnn_1x1_global_stratosphere_only_era5_uvtheta__train_epoch100.pt)

echo "retrieving test input..."
(cd inputs && wget https://g-b56e81.7a577b.6fbd.data.globus.org/stratosphere_1x1_inputfeatures_u_v_theta_w_N2_uw_vw_era5_training_data_hourly_2010_constant_mu_sigma_scaling01.nc)
