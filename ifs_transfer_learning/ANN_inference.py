# This script is for inference on transfer learning models only
# ANN_CNN models only

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

from dataloader_definition import Dataset_ANN_CNN
from model_definition import ANN_CNN
from function_training import Inference_and_Save_ANN_CNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
domain = "global"  # 'regional'
vertical = sys.argv[1]  #'stratosphere_only' # 'global'
features = sys.argv[
    2
]  #'uvthetaw' # 'uvtheta', ''uvthetaw', or 'uvw' for troposphere | additionally 'uvthetaN2' and 'uvthetawN2' for stratosphere_only
dropout = 0.0  # can choose this to be non-zero during inference for uncertainty quantification. A little dropout goes a long way. Choose a small value - 0.03ish?
epoch = int(sys.argv[3])
teston = sys.argv[4]
stencil = int(sys.argv[6])

if stencil == 1:
    bs_train = 20
    bs_test = bs_train
else:
    bs_train = 10
    bs_test = bs_train


# model checkpoint
pref = "/scratch/users/ag4680/torch_saved_models/transfer_learning_IFS/ann_cnn/"
ckpt = f"TLIFS_ann_cnn_{stencil}x{stencil}_era5_ifs_{domain}_{vertical}_{features}_mseloss_train_epoch{str(epoch).zfill(2)}.pt"


log_filename = f"./TLIFS_inference_ann_cnn_{stencil}x{stencil}_{domain}_{vertical}_{features}_ckpt_epoch_{epoch}.txt"


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
        if stencil == 1:
            pre = f"/scratch/users/ag4680/training_data/era5/stratosphere_{stencil}x{stencil}_inputfeatures_u_v_theta_w_N2_uw_vw_era5_training_data_hourly_"
        else:
            pre = f"/scratch/users/ag4680/training_data/era5/stratosphere_nonlocal_{stencil}x{stencil}_inputfeatures_u_v_theta_w_N2_uw_vw_era5_training_data_hourly_"
    elif vertical == "global" or vertical == "stratosphere_update":
        if stencil == 1:
            pre = f"/scratch/users/ag4680/training_data/era5/{stencil}x{stencil}_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_"
        else:
            pre = f"/scratch/users/ag4680/training_data/era5/nonlocal_{stencil}x{stencil}_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_"

    for year in test_years:
        for months in np.arange(test_month, test_month + 1):
            test_files.append(f"{pre}{year}_constant_mu_sigma_scaling{str(months).zfill(2)}.nc")

# -------- To test on three months of IFS data
elif teston == "IFS":
    if vertical == "stratosphere_only":
        test_files = [
            f"/scratch/users/ag4680/coarsegrained_ifs_gwmf_helmholtz/NDJF/stratosphere_only_{stencil}x{stencil}_inputfeatures_u_v_theta_w_N2_uw_vw_era5_training_data_hourly_constant_mu_sigma_scaling.nc"
        ]
    elif vertical == "global" or vertical == "stratosphere_update":
        test_files = [
            f"/scratch/users/ag4680/coarsegrained_ifs_gwmf_helmholtz/NDJF/troposphere_and_stratosphere_{stencil}x{stencil}_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_constant_mu_sigma_scaling.nc"
        ]

write_log(
    f"Inference the ANN_CNN model on {domain} horizontal and {vertical} vertical model, with features {features} and dropout={dropout}."
)
write_log(f"Test files = {test_files}")

# initialize dataloader
testset = Dataset_ANN_CNN(
    files=test_files,
    domain=domain,
    vertical=vertical,
    stencil=stencil,
    manual_shuffle=False,
    features=features,
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=bs_train, drop_last=False, shuffle=False, num_workers=8
)

idim = testset.idim
odim = testset.odim
hdim = 4 * idim

# Model checkpoint to use
# ---- define model
model = ANN_CNN(idim=idim, odim=odim, hdim=hdim, stencil=stencil, dropout=dropout)
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
        out = f"/scratch/users/ag4680/gw_inference_ncfiles/TLIFS_inference_{S[0]}_{test_years[0]}_{test_month}_dropoutON_testedonERA5_{sys.argv[7]}.nc"
    else:
        out = f"/scratch/users/ag4680/gw_inference_ncfiles/TLIFS_inference_{S[0]}_{test_years[0]}_{test_month}_dropoutON_testedonIFS_{sys.argv[6]}.nc"
write_log(f"Output NC file: {out}")

# better to create the file within the inference_and_save function
write_log("Initiating inference")
Inference_and_Save_ANN_CNN(
    model=model,
    testset=testset,
    testloader=testloader,
    bs_test=bs_test,
    device=device,
    log_filename=log_filename,
    outfile=out,
    stencil=stencil,
)

write_log("Inference complete")
