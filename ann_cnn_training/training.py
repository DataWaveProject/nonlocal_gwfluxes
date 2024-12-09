import sys
import math
import numpy as np

# from netCDF4 import Dataset
from time import time as time2
import xarray as xr

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------ for data parallelism --------------------------
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

# ------------------------------------------------------------
import torch.optim as optim

# from torch.utils.data import DataLoader, random_split
# from torch.utils.data import Dataset as TensorDataset
from collections import OrderedDict
import pandas as pd

from dataloader_definition import Dataset_ANN_CNN
from model_definition import ANN_CNN
from function_training import Training_ANN_CNN

torch.set_printoptions(edgeitems=2)
torch.manual_seed(123)

# f = "/home/users/ag4680/myjupyter/137levs_ak_bk.npz"
# data = np.load(f, mmap_mode="r")

# lev = data["lev"]
# ht = data["ht"]
# ak = data["ak"]
# bk = data["bk"]
# R = 287.05
# T = 250
# rho = 100 * lev / (R * T)

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)  # 'cuda' to select all available GPUs

# PARAMETERS AND HYPERPARAMETERS
restart = False
init_epoch = 1  # where to resume. Should have checkpoint saved for init_epoch-1. 1 for fresh runs.
nepochs = 100
# ----------------------
domain = sys.argv[1]  # global' # 'regional'
vertical = sys.argv[2]  #'global' # or 'stratosphere_only' or 'stratosphere_update'
# ----------------------
features = sys.argv[3]  #'uvtheta'
stencil = int(sys.argv[4])  # stencil size
# ----------------------
lr_min = 1e-4
lr_max = 5e-4
# ----------------------
if stencil == 1:
    bs_train = 20
    bs_test = bs_train
else:
    bs_train = 10
    bs_test = bs_train
dropout = 0.1

log_filename = f"./ann_cnns_{stencil}x{stencil}_{domain}_{vertical}_{features}_epoch_{init_epoch}_to_{init_epoch+nepochs-1}.txt"


def write_log(*args):
    line = " ".join([str(a) for a in args])
    log_file = open(log_filename, "a")
    log_file.write(line + "\n")
    log_file.close()
    print(line)


if device != "cpu":
    ngpus = torch.cuda.device_count()
    write_log(f"NGPUS = {ngpus}")

write_log(
    f"Training the {stencil}x{stencil} ANN-CNNs, {domain} horizontal and {vertical} vertical model with features {features} with min-max learning rates {lr_min} to {lr_max} for a CyclicLR, and dropout={dropout}.\n"
)

idir = sys.argv[5]
odir = sys.argv[6]

if vertical == "stratosphere_only":
    if stencil == 1:
        pre = (
            idir + f"stratosphere_1x1_inputfeatures_u_v_theta_w_N2_uw_vw_era5_training_data_hourly_"
        )
    else:
        pre = (
            idir
            + f"stratosphere_nonlocal_{stencil}x{stencil}_inputfeatures_u_v_theta_w_N2_uw_vw_era5_training_data_hourly_"
        )
elif vertical == "global" or vertical == "stratosphere_update":
    if stencil == 1:
        pre = idir + "1x1_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_"
    else:
        pre = (
            idir
            + f"nonlocal_{stencil}x{stencil}_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_"
        )

train_files = []
train_years = np.array([2010, 2012, 2014])
for year in train_years:
    for months in np.arange(1, 13):
        train_files.append(f"{pre}{year}_constant_mu_sigma_scaling{str(months).zfill(2)}.nc")

test_files = []
test_years = np.array([2015])
for year in test_years:
    for months in np.arange(1, 13):
        # for months in np.arange(1,3):
        test_files.append(f"{pre}{year}_constant_mu_sigma_scaling{str(months).zfill(2)}.nc")

write_log(
    f"Training the {domain} horizontal and {vertical} vertical model, with features {features} with min-max learning rates {lr_min} to {lr_max}, and dropout={dropout}. Starting from epoch {init_epoch}. Training on years {train_years} and testing on years {test_years}.\n"
)

# write_log(train_files)
# write_log(test_files)
write_log("Defined input files")
write_log(f"train batch size = {bs_train}")
write_log(f"validation batch size = {bs_test}")


# print(train_files)
# print(test_files)

trainset = Dataset_ANN_CNN(
    files=train_files,
    domain="global",
    vertical=vertical,
    features=features,
    stencil=stencil,
    manual_shuffle=False,
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=bs_train, drop_last=False, shuffle=False, num_workers=4
)  # , persistent_workers=True)
testset = Dataset_ANN_CNN(
    files=test_files,
    domain="global",
    vertical=vertical,
    features=features,
    stencil=stencil,
    manual_shuffle=False,
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=bs_test, drop_last=False, shuffle=False, num_workers=4
)

idim = trainset.idim
odim = trainset.odim
hdim = 4 * idim
write_log(f"Input dim: {idim}, hidden dim: {hdim}, output dim: {odim}")

model = ANN_CNN(idim=idim, odim=odim, hdim=hdim, dropout=dropout, stencil=trainset.stencil)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=lr_min,
    max_lr=lr_max,
    step_size_up=50,
    step_size_down=50,
    cycle_momentum=False,
)
loss_fn = nn.MSELoss()
write_log(
    f"model created. \n --- model size: {model.totalsize():.2f} MBs,\n --- Num params: {model.totalparams()/10**6:.3f} mil. "
)

# fac not used for vertical scaling yet, but good to have it
fac = torch.ones(122)  # torch.from_numpy(rho[15:]**0.1)
fac = (1.0 / fac).to(torch.float32)
fac = fac.to(device)
# write_log('fac_created')


file_prefix = odir + f"{vertical}/ann_cnn_{stencil}x{stencil}_{domain}_{vertical}_era5_{features}_"
# write_log(f'file prefix: {file_prefix}')
if restart:
    # load checkpoint before resuming training
    PATH = f"{file_prefix}_train_epoch{init_epoch-1}.pt"
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

write_log("Starting training ...")
# Training loop
model, loss_train, loss_test = Training_ANN_CNN(
    nepochs=nepochs,
    init_epoch=init_epoch,
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    trainloader=trainloader,
    testloader=testloader,
    stencil=trainset.stencil,
    bs_train=bs_train,
    bs_test=bs_test,
    save=True,
    file_prefix=file_prefix,
    scheduler=scheduler,
    device=device,
    log_filename=log_filename,
)
