# Inference script for DETERMINISTIC inference on Attention UNet models
# header
import numpy as np
import xarray as xr
import pandas as pd
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -----------------------------------------------------------
import logging
import argparse

# -------- for data parallelism ----------
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

from dataloader_attention_unet import Dataset_AttentionUNet
from model_attention_unet import Attention_UNet
from function_training import Training_AttentionUNet, Inference_and_Save_AttentionUNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

bs_train = 40  # 80 (80 works for most). (does not work for global uvthetaw)
bs_test = bs_train

# --------------------------------------------------
domain = "global"  # 'regional'
vertical = sys.argv[1]  #'stratosphere_only' # 'global'
features = sys.argv[
    2
]  #'uvthetaw' # 'uvtheta', ''uvthetaw', or 'uvw' for troposphere | additionally 'uvthetaN2' and 'uvthetawN2' for stratosphere_only
dropout = 0  # can choose this to be non-zero during inference for uncertainty quantification. A little dropout goes a long way. Choose a small value - 0.03ish?
epoch = int(sys.argv[3])

# model checkpoint
pref = "/scratch/users/ag4680/torch_saved_models/attention_unet/"
ckpt = f"attnunet_era5_{domain}_{vertical}_{features}_mseloss_train_epoch{str(epoch).zfill(2)}.pt"

log_filename = f"./inference_attnunet_{domain}_{vertical}_{features}_ckpt_epoch_{epoch}.txt"
logger = logging.getLogger(__name__)
logging.basicConfig(filename=log_filename, level=logging.INFO)

if device != "cpu":
    ngpus = torch.cuda.device_count()
    logger.info(f"NGPUS = {ngpus}")

# Define test files

# ------- To test on one year of ERA5 data
test_files = []
test_years = np.array([2015])
test_month = int(sys.argv[4])  # np.arange(1,13)
logger.info(f"Inference for month {test_month}")
if vertical == "stratosphere_only":
    pre = "/scratch/users/ag4680/training_data/era5/stratosphere_1x1_inputfeatures_u_v_theta_w_N2_uw_vw_era5_training_data_hourly_"
elif vertical == "global" or vertical == "stratosphere_update":
    pre = "/scratch/users/ag4680/training_data/era5/1x1_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_"
for year in test_years:
    for months in np.arange(test_month, test_month + 1):
        test_files.append(f"{pre}{year}_constant_mu_sigma_scaling{str(months).zfill(2)}.nc")

# -------- To test on four months of IFS data
# if vertical == 'stratosphere_only':
#    test_files=['/scratch/users/ag4680/coarsegrained_ifs_gwmf_helmholtz/NDJF/stratosphere_only_1x1_inputfeatures_u_v_theta_w_N2_uw_vw_era5_training_data_hourly_constant_mu_sigma_scaling.nc']
# elif vertical == 'global' or vertical=='stratosphere_update':
#    test_files=['/scratch/users/ag4680/coarsegrained_ifs_gwmf_helmholtz/NDJF/troposphere_and_stratosphere_1x1_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_constant_mu_sigma_scaling.nc']

logger.info(
    f"Inference the Attention UNet model on {domain} horizontal and {vertical} vertical model, with features {features} and dropout={dropout}."
)
logger.info(f"Test files = {test_files}")

# initialize dataloade
testset = Dataset_AttentionUNet(
    files=test_files, domain=domain, vertical=vertical, manual_shuffle=False, features=features
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=bs_train, drop_last=False, shuffle=False, num_workers=8
)

ch_in = testset.idim
ch_out = testset.odim

# Model checkpoint to use
# ---- define model
model = Attention_UNet(ch_in=ch_in, ch_out=ch_out, dropout=dropout)
loss_fn = nn.MSELoss()
logger.info(
    f"Model created. \n --- model size: {model.totalsize():.2f} MBs,\n --- Num params: {model.totalparams()/10**6:.3f} mil. "
)
# ---- load model
PATH = pref + ckpt
if device == "cpu":
    checkpoint = torch.load(PATH, map_location=torch.device("cpu"))
else:
    checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()


# create netCDF file
S = ckpt.split(".")
if dropout == 0:
    out = f"/scratch/users/ag4680/gw_inference_ncfiles/inference_{S[0]}_{test_years[0]}_{test_month}.nc"
    # out=f'/scratch/users/ag4680/gw_inference_ncfiles/inference_{S[0]}_{test_years[0]}_{test_month}_testedonIFS.nc'
else:
    out = f"/scratch/users/ag4680/gw_inference_ncfiles/inference_{S[0]}_{test_years[0]}_{test_month}_dropoutON_{sys.argv[5]}.nc"
logger.info(f"Output NC file: {out}")

# better to create the file within the inference_and_save function
logger.info("Initiating inference")
Inference_and_Save_AttentionUNet(model, testset, testloader, bs_test, device, logger, out)

logger.info("Inference complete")
