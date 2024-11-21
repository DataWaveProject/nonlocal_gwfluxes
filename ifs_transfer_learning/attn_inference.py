# This script is for inference on transfer learning models only
# Attention models only

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

# -------- for data parallelism ----------
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

from dataloader_definition import Dataset_AttentionUNet
from model_definition import Attention_UNet
from function_training import Inference_and_Save_AttentionUNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

bs_train = 40  # 80 (80 works for most). (does not work for global uvthetaw)
bs_test = bs_train

# --------------------------------------------------
domain = "global"  # 'regional'
vertical = sys.argv[1]  #'stratosphere_only' # 'global', or 'stratosphere_update'
features = sys.argv[
    2
]  #'uvthetaw' # 'uvtheta', ''uvthetaw', or 'uvw' for troposphere | additionally 'uvthetaN2' and 'uvthetawN2' for stratosphere_only
dropout = 0.0  # can choose this to be non-zero during inference for uncertainty quantification. A little dropout goes a long way. Choose a small value - 0.03ish?
epoch = int(sys.argv[3])
teston = sys.argv[4]

# model checkpoint
pref = "/scratch/users/ag4680/torch_saved_models/transfer_learning_IFS/attention_unet/"
ckpt = f"TLIFS_attnunet_era5_ifs_{domain}_{vertical}_{features}_mseloss_train_epoch{str(epoch).zfill(2)}.pt"


log_filename = f"./TLIFS_inference_attnunet_{domain}_{vertical}_{features}_ckpt_epoch_{epoch}.txt"


# log_filename=f"./icml_train_ann-cnn_1x1_global_4hl_dropout0p1_hdim-2idim_restart_epoch_{init_epoch}_to_{init_epoch+nepochs-1}.txt"
def write_log(*args):
    line = " ".join([str(a) for a in args])
    log_file = open(log_filename, "a")
    log_file.write(line + "\n")
    log_file.close()
    print(line)


if device != "cpu":
    ngpus = torch.cuda.device_count()
    write_log(f"NGPUS = {ngpus}")

# Define test files
# --------- To test on one year of ERA5 data
if teston == "ERA5":
    test_files = []
    test_years = np.array([2015])
    test_month = int(sys.argv[5])  # np.arange(1,13)
    write_log(f"Inference for month {test_month}")
    if vertical == "stratosphere_only":
        pre = "/scratch/users/ag4680/training_data/era5/stratosphere_1x1_inputfeatures_u_v_theta_w_N2_uw_vw_era5_training_data_hourly_"
    elif vertical == "global" or vertical == "stratosphere_update":
        pre = "/scratch/users/ag4680/training_data/era5/1x1_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_"
    for year in test_years:
        for months in np.arange(test_month, test_month + 1):
            test_files.append(f"{pre}{year}_constant_mu_sigma_scaling{str(months).zfill(2)}.nc")

# -------- To test on three months of IFS data
elif teston == "IFS":
    if vertical == "stratosphere_only":
        test_files = [
            f"/scratch/users/ag4680/coarsegrained_ifs_gwmf_helmholtz/NDJF/stratosphere_only_1x1_inputfeatures_u_v_theta_w_N2_uw_vw_era5_training_data_hourly_constant_mu_sigma_scaling.nc"
        ]
    elif vertical == "global" or vertical == "stratosphere_update":
        test_files = [
            f"/scratch/users/ag4680/coarsegrained_ifs_gwmf_helmholtz/NDJF/troposphere_and_stratosphere_1x1_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_constant_mu_sigma_scaling.nc"
        ]

write_log(
    f"Inference the Attention UNet model on {domain} horizontal and {vertical} vertical model, with features {features} and dropout={dropout}."
)
write_log(f"Test files = {test_files}")

# initialize dataloader
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
write_log(
    f"Model created. \n --- model size: {model.totalsize():.2f} MBs,\n --- Num params: {model.totalparams()/10**6:.3f} mil. "
)
# ---- load model
PATH = pref + ckpt
checkpoint = torch.load(PATH, map_location=torch.device(device))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()


# create netCDF file
S = ckpt.split(".")
if dropout == 0:
    if teston == "ERA5":
        out = f"/scratch/users/ag4680/gw_inference_ncfiles/TLIFS_inference_{S[0]}_{test_years[0]}_{test_month}_testedonERA5.nc"
    else:
        out = f"/scratch/users/ag4680/gw_inference_ncfiles/TLIFS_inference_{S[0]}_testedonIFS.nc"
else:
    if teston == "ERA5":
        out = f"/scratch/users/ag4680/gw_inference_ncfiles/TLIFS_inference_{S[0]}_{test_years[0]}_{test_month}_dropoutON_testedonERA5_{sys.argv[6]}.nc"
    else:
        out = f"/scratch/users/ag4680/gw_inference_ncfiles/TLIFS_inference_{S[0]}_{test_years[0]}_{test_month}_dropoutON_testedonIFS_{sys.argv[5]}.nc"
write_log(f"Output NC file: {out}")

# better to create the file within the inference_and_save function
write_log("Initiating inference")
Inference_and_Save_AttentionUNet(model, testset, testloader, bs_test, device, log_filename, out)

write_log("Inference complete")
