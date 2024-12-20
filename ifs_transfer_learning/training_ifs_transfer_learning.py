# ================== Transfer Learning from multi-year low-resolution ERA5 to 4-month high-resolution IFS =================
# Load a pre-trained model
# Freeze all layers
# Re-define the last layer
# Train
# How to get more high-res data?

import sys
import math
import numpy as np

# from matplotlib import pyplot as plt
# from netCDF4 import Dataset
# import pygrib as pg
from time import time as time2
import xarray as xr
# import dask
# from files import train_files, test_files

# %matplotlib inline
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


from dataloader_definition import Dataset_ANN_CNN, Dataset_AttentionUNet
from model_definition import ANN_CNN, Attention_UNet
from function_training import (
    Training_ANN_CNN_TransferLearning,
    Training_AttentionUNet_TransferLearning,
    Model_Freeze_Transfer_Learning,
)
# from files import train_files, test_files

torch.set_printoptions(edgeitems=2)
torch.manual_seed(123)

f = "/home/users/ag4680/myjupyter/137levs_ak_bk.npz"
data = np.load(f, mmap_mode="r")

lev = data["lev"]
ht = data["ht"]
ak = data["ak"]
bk = data["bk"]
R = 287.05
T = 250
rho = 100 * lev / (R * T)


# https://stackoverflow.com/questions/54216920/how-to-use-multiple-gpus-in-pytorch
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)  # to select all available GPUs

# 1. This is the best GPU check so far - If 4 GPUS, this should give an error for 4 and above, and only accept 0 to 3
# 2. https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel
# 3. DistributedDataParallel is proven to be significantly faster than torch.nn.DataParallel for single-node multi-GPU data parallel training.
# torch.cuda.set_device(4)


# PARAMETERS AND HYPERPARAMETERS
model_type = sys.argv[1]  #'attention' # or 'ann'
init_epoch = 1  # which epoch to resume from. Should have restart file from init_epoch-1 ready
nepochs = 200
# ----------------------
features = sys.argv[4]  #'uvtheta'
if model_type == "attention":
    stencil = 1
else:
    stencil = int(sys.argv[6])
# ----------------------
domain = sys.argv[
    2
]  # global' # 'regional'. Most likely won't set regional for these experiments. The functions might not be constructed to handle them properly
vertical = sys.argv[3]  #'global' # or 'stratosphere_only' or 'stratosphere_update'
# ----------------------
lr_min = 1e-4
lr_max = 9e-4
# ----------------------
if model_type == "ann":
    dropout = 0.1
elif model_type == "attention":
    dropout = 0.05
ckpt_epoch = sys.argv[5]
if model_type == "ann":
    PATH = f"/scratch/users/ag4680/torch_saved_models/JAMES/{vertical}/ann_cnn_{stencil}x{stencil}_{domain}_{vertical}_era5_{features}__train_epoch{ckpt_epoch}.pt"
elif model_type == "attention":
    PATH = f"/scratch/users/ag4680/torch_saved_models/attention_unet/attnunet_era5_{domain}_{vertical}_{features}_mseloss_train_epoch{ckpt_epoch}.pt"

if vertical == "global" or vertical == "stratosphere_update":
    if model_type == "ann":
        log_filename = f"./IFStransfer_{vertical}_ann_cnn_{stencil}x{stencil}_{features}_6hl_epoch_{init_epoch}_to_{init_epoch+nepochs-1}.txt"
    elif model_type == "attention":
        log_filename = f"./IFStransfer_{vertical}_attention_{features}_epoch_{init_epoch}_to_{init_epoch+nepochs-1}.txt"
elif vertical == "stratosphere_only":
    if model_type == "ann":
        log_filename = f"./IFStransfer_{vertical}_ann_cnn_{stencil}x{stencil}_{features}_6hl_epoch_{init_epoch}_to_{init_epoch+nepochs-1}.txt"
    elif model_type == "attention":
        log_filename = f"./IFStransfer_{vertical}_attention_{features}_epoch_{init_epoch}_to_{init_epoch+nepochs-1}.txt"


def write_log(*args):
    line = " ".join([str(a) for a in args])
    log_file = open(log_filename, "a")
    log_file.write(line + "\n")
    log_file.close()
    print(line)


if device != "cpu":
    ngpus = torch.cuda.device_count()
    write_log(f"NGPUS = {ngpus}")

if model_type == "ann":
    write_log(
        f"Transfer learning: retraining ERA5 trained ANN-CNNs with stencil {stencil}x{stencil}, vertical={vertical}, horizontal={domain} model with features {features}. CyclicLR scheduler to cycle learning rates between lr_min={lr_min} to lr_max={lr_max}."
    )
elif model_type == "attention":
    write_log(
        f"Transfer learning: retraining ERA5 trained {model_type}, vertical={vertical}, horizontal={domain} model with features {features}. CyclicLR scheduler to cycle learning rates between lr_min={lr_min} to lr_max={lr_max}."
    )

# ====================================================================================================
# DEFINING INPUT FILES
# This should point to the IFS files
if vertical == "global" or vertical == "stratosphere_update":
    f = f"/scratch/users/ag4680/coarsegrained_ifs_gwmf_helmholtz/NDJF/troposphere_and_stratosphere_{stencil}x{stencil}_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_constant_mu_sigma_scaling.nc"
elif vertical == "stratosphere_only":
    f = f"/scratch/users/ag4680/coarsegrained_ifs_gwmf_helmholtz/NDJF/stratosphere_only_{stencil}x{stencil}_inputfeatures_u_v_theta_w_N2_uw_vw_era5_training_data_hourly_constant_mu_sigma_scaling.nc"
write_log(f"File name: {f}")
train_files = [f]
test_files = [f]

# setting Shuffle=True automatically takes care of permuting in time - but not control over seeding, so...
# set manual_shuffle=True and control seed from the function definition
# ===============================================================
# DEFINING RUN HYPERPARAMETERS AND SETTING UP THE RUN
# multiple batches from the vectorized matrices
if model_type == "ann":
    if stencil == 1:
        # for three variables, bs_train = 80 works well. Might have to reduce for four variables
        bs_train = 20  # 80#40#37
        bs_test = 20  # 80#40#37
    elif stencil > 1:
        bs_train = 10
        bs_test = 10
elif model_type == "attention":
    bs_train = 80
    bs_test = 80
write_log(f"train batch size = {bs_train}")
write_log(f"validation batch size = {bs_test}")

# ============================= 1x1 ===========================
tstart = time2()
# IF REGIONAL, SPECIFY THE REGION AS WELL
# 1andes, 2scand, 3himalaya, 4newfound, 5south_ocn, 6se_asia, 7natlantic, 8npacific
rgn = "1andes"
write_log(f"Region: {rgn}")
# shuffle=False leads to much faster reading! Since 3x3 and 5x5 is slow, set this to False

# Since not a lot of IFS data, opting for no validation set
if model_type == "ann":
    trainset = Dataset_ANN_CNN(
        files=train_files,
        domain="global",
        vertical=vertical,
        features=features,
        stencil=stencil,
        manual_shuffle=False,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=bs_train, drop_last=False, shuffle=False, num_workers=8
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
        testset, batch_size=bs_test, drop_last=False, shuffle=False, num_workers=8
    )  # , persistent_workers=True)
    idim = trainset.idim
    odim = trainset.odim
    hdim = 4 * idim
    write_log(f"Input dim: {idim}, hidden dim: {hdim}, output dim: {odim}")

elif model_type == "attention":
    trainset = Dataset_AttentionUNet(
        files=train_files, domain=domain, vertical=vertical, features=features, manual_shuffle=False
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=bs_train, drop_last=False, shuffle=False, num_workers=8
    )  # , persistent_workers=True)
    testset = Dataset_AttentionUNet(
        files=test_files,
        domain="global",
        vertical=vertical,
        features=features,
        manual_shuffle=False,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=bs_test, drop_last=False, shuffle=False, num_workers=8
    )  # , persistent_workers=True)
    idim = trainset.idim
    odim = trainset.odim

    print(idim)
    print(odim)


write_log(f"Loading model checkpoint from {PATH}")
# lr 10-6 to 10-4 over 100 up and 100 down steps works well waise
# Important note: The optimizer is loaded on to the same device at the model. So best to first define the model and port to the GPU, and then define the optimizer.
# Otherwise, will have to port both the model and optimizer onto the device at the end, to prevent device mismatch error
if model_type == "ann":
    model = ANN_CNN(idim=idim, odim=odim, hdim=hdim, dropout=dropout, stencil=trainset.stencil)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=lr_min,
        max_lr=lr_max,
        step_size_up=10,
        step_size_down=10,
        cycle_momentum=False,
    )  # since low IFS data, step size is small=10
    loss_fn = nn.MSELoss()

    # Load model checkpoint
    # if device=='cpu':
    #    checkpoint=torch.load(PATH, map_location=torch.device('cpu'))
    # else:
    #    checkpoint=torch.load(PATH)
    checkpoint = torch.load(PATH, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=lr_min,
        max_lr=lr_max,
        step_size_up=10,
        step_size_down=10,
        cycle_momentum=False,
    )

if model_type == "attention":
    # ADD THIS
    model = Attention_UNet(ch_in=idim, ch_out=odim, dropout=dropout)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=lr_min,
        max_lr=lr_max,
        step_size_up=10,
        step_size_down=10,
        cycle_momentum=False,
    )  # since low IFS data, step size is small=10
    loss_fn = nn.MSELoss()

    # Load model checkpoint
    # if device=='cpu':
    #    checkpoint=torch.load(PATH, map_location=torch.device('cpu'))
    # else:
    #    checkpoint=torch.load(PATH)
    checkpoint = torch.load(PATH, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=lr_min,
        max_lr=lr_max,
        step_size_up=10,
        step_size_down=10,
        cycle_momentum=False,
    )

model = Model_Freeze_Transfer_Learning(model=model, model_type=model_type)


for params in model.parameters():
    write_log(f"{params.requires_grad}")

write_log(
    f"model loaded \n --- model size: {model.totalsize():.2f} MBs,\n --- Num params: {model.totalparams()/10**6:.3f} mil. "
)
write_log("Model checkpoint loaded and prepared for re-training")


# Set final weights names
if model_type == "ann":
    file_prefix = f"/scratch/users/ag4680/torch_saved_models/transfer_learning_IFS/ann_cnn/TLIFS_ann_cnn_{stencil}x{stencil}_era5_ifs_{domain}_{vertical}_{features}_mseloss"
elif model_type == "attention":
    file_prefix = f"/scratch/users/ag4680/torch_saved_models/transfer_learning_IFS/attention_unet/TLIFS_attnunet_era5_ifs_{domain}_{vertical}_{features}_mseloss"

# might not need to restart - so haven't added that part here. If needed, borrow it from other files

write_log("Re-Training final two layers...")
if model_type == "ann":
    model, loss_train = Training_ANN_CNN_TransferLearning(
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
elif model_type == "attention":
    model, loss_train = Training_AttentionUNet_TransferLearning(
        nepochs=nepochs,
        init_epoch=init_epoch,
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

write_log("Training complete.")
