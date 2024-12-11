# Script to train the Attention U-Net model
# Header
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

from dataloader_attention_unet import Dataset_AttentionUNet
from model_attention_unet import Attention_UNet
from function_training import Training_AttentionUNet

f = "/home/users/ag4680/myjupyter/137levs_ak_bk.npz"
data = np.load(f, mmap_mode="r")
lev = data["lev"]
ht = data["ht"]
ak = data["ak"]
bk = data["bk"]
R = 287.05
T = 250
rho = 100 * lev / (R * T)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# ------------------------------------------------------------------------------

# Define output file using hyperparameters
# define writelog function

# IMPORTANT RUN PARAMETERS TO SET BEFORE SUBMISSION
restart = False
init_epoch = (
    1  # which epoch to resume from. Should have restart file from init_epoch-1 ready, otherwise 1
)
nepochs = 150
bs_train = 40  # 80 (80 works for most). (does not work for global uvthetaw)
bs_test = bs_train

# --------------------------------------------------
domain = "global"  # 'regional'
vertical = sys.argv[1]  #'stratosphere_only' # 'global', # stratosphere_update
features = sys.argv[
    2
]  #'uvthetaw' # 'uvtheta', ''uvthetaw', or 'uvw' for troposphere | additionally 'uvthetaN2' and 'uvthetawN2' for stratosphere_only
# --------------------------------------------------
if vertical == "stratosphere_only" or vertical == "stratosphere_update":
    lr_min = 1e-4
    lr_max = 5e-4
else:  # lower for the troposphere
    lr_min = 1e-4  # -4
    lr_max = 5e-4  # -4
dropout = 0.05  # dropout probability


idir = sys.argv[3]
odir = sys.argv[4]

log_filename = f"./attnunet_{domain}_{vertical}_{features}_smalldropout_epoch_{init_epoch}_to_{init_epoch+nepochs-1}.txt"


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

write_log(
    f"Training the {domain} horizontal and {vertical} vertical model with features {features} with min-max learning rates {lr_min} to {lr_max}."
)


if vertical == "stratosphere_only":
    pre = idir + f"stratosphere_1x1_inputfeatures_u_v_theta_w_N2_uw_vw_era5_training_data_hourly_"
elif vertical == "global" or vertical == "stratosphere_update":
    pre = idir + f"1x1_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_"

train_files = []
train_years = np.array([2010, 2012, 2014])
for year in train_years:
    # for year in np.array([2010]):
    for months in np.arange(1, 13):
        # for months in np.arange(1,3):
        train_files.append(f"{pre}{year}_constant_mu_sigma_scaling{str(months).zfill(2)}.nc")

test_files = []
test_years = np.array([2015])
for year in test_years:
    for months in np.arange(1, 13):
        # for months in np.arange(1,3):
        test_files.append(f"{pre}{year}_constant_mu_sigma_scaling{str(months).zfill(2)}.nc")


write_log(
    f"Training the {domain} horizontal and {vertical} vertical model, with features {features} with min-max learning rates {lr_min} to {lr_max}, and dropout={dropout}. Starting from epoch {init_epoch}. Training on years {train_years} and testing on years {test_years}."
)

write_log("Defined input files")


# create dataloaders
trainset = Dataset_AttentionUNet(
    files=train_files, domain=domain, vertical=vertical, manual_shuffle=False, features=features
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=bs_train, drop_last=False, shuffle=False, num_workers=8
)  # change this before job submission
testset = Dataset_AttentionUNet(
    files=test_files, domain=domain, vertical=vertical, manual_shuffle=False, features=features
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=bs_train, drop_last=False, shuffle=False, num_workers=8
)  # change this before job submission


# create model and set hyperparameters
ch_in = trainset.idim
ch_out = trainset.odim

write_log(f"Input channel: {ch_in}")
write_log(f"Output channel: {ch_out}")


model = Attention_UNet(ch_in=ch_in, ch_out=ch_out, dropout=dropout)
# port model to GPU. ensures optimizer is loaded to GPU as well
model = model.to(device)
write_log(
    f"Model created. \n --- model size: {model.totalsize():.2f} MBs,\n --- Num params: {model.totalparams()/10**6:.3f} mil. "
)

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


file_prefix = odir + f"attention_unet/attnunet_era5_{domain}_{vertical}_{features}_mseloss"
if restart:
    PATH = f"{file_prefix}_train_epoch{init_epoch-1}.pt"
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    optim.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=lr_min,
        max_lr=lr_max,
        step_size_up=50,
        step_size_down=50,
        cycle_momentum=False,
    )
    # scheduler.load_state_dict(checkpoint['scheduler'])
    # epoch = checkpoint['epoch']
    loss_fn = checkpoint["loss"]


# train
model, loss_train, loss_test = Training_AttentionUNet(
    nepochs=nepochs,
    init_epoch=1,
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    trainloader=trainloader,
    testloader=testloader,
    bs_train=bs_train,
    bs_test=bs_test,
    save=True,
    file_prefix=file_prefix,
    scheduler=scheduler,
    device=device,
    log_filename=log_filename,
)


write_log("Model training complete")
