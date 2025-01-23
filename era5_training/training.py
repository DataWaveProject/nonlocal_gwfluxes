import sys
import math
import numpy as np

# ----------------------------------------------------------
from time import time as time2
import xarray as xr

# ----------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------
import logging
import argparse
from pathlib import Path

# ------------ for data parallelism --------------------------
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim

# -----------------------------------------------------------
from collections import OrderedDict
import pandas as pd

# -----------------------------------------------------------
sys.path.append("../utils/")
from dataloader_definition import Dataset_ANN_CNN, Dataset_AttentionUNet
from model_definition import ANN_CNN, ANN_CNN10, Attention_UNet
from function_training import Training_ANN_CNN, Training_AttentionUNet

torch.set_printoptions(edgeitems=2)
torch.manual_seed(123)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-M",
    "--model",
    choices=["ann", "attention"],
    default="ann",
    help="Model to be trained",
)
parser.add_argument(
    "-d",
    "--horizontal",
    choices=["global"],
    default="global",
    help="Horizontal domain for training",
)
parser.add_argument(
    "-v",
    "--vertical",
    choices=["global", "stratosphere_update"],
    default="global",
    help="Vertical domain for training",
)
parser.add_argument(
    "-f",
    "--features",
    choices=["uvtheta", "uvw", "uvthetaw"],
    default="uvtheta",
    help="Feature set for training",
)
parser.add_argument(
    "-s", "--stencil", type=int, choices=[1, 3, 5], default=1, help="Horizontal stencil for the NN"
)
parser.add_argument(
    "-i", "--input_dir", default=Path.cwd(), help="Input directory with training data", type=Path
)
parser.add_argument(
    "-o",
    "--output_dir",
    default=Path.cwd(),
    help="Output directory to store checkpoints",
    type=Path,
)
args = parser.parse_args()
# print parsed args
print(f"model={args.model}")
print(f"horizontal={args.horizontal}")
print(f"vertical={args.vertical}")
print(f"features={args.features}")
print(f"stencil={args.stencil}")
print(f"input_dir={args.input_dir}")
print(f"output_dir={args.output_dir}")

model = args.model
# PARAMETERS AND HYPERPARAMETERS
restart = False
ablation = False
init_epoch = 1  # where to resume. Should have checkpoint saved for init_epoch-1. 1 for fresh runs.
nepochs = 100 if model == "ann" else 150
# ----------------------
domain = args.horizontal  # global' # 'regional'
vertical = args.vertical  #'global' # or 'stratosphere_only' or 'stratosphere_update'
# ----------------------
features = args.features  #'uvtheta'
stencil = args.stencil  # stencil size
# ----- model sanity check ----------
if model == "attention" and stencil > 1:
    raise ValueError(
        "The selected model is Attention UNet but stencil > 1 is only allowed for ANNs."
    )
if stencil % 2 == 0:
    raise ValueError("Stencil must be odd.")
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


if not ablation:
    log_filename = f"./{model}_{stencil}x{stencil}_{domain}_{vertical}_{features}_epoch_{init_epoch}_to_{init_epoch+nepochs-1}.txt"
else:
    log_filename = f"./{model}_{stencil}x{stencil}_{domain}_{vertical}_{features}_epoch_{init_epoch}_to_{init_epoch+nepochs-1}_ABLATION_10hiddenlayers.txt"
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename=log_filename, level=logging.INFO
)  # must specify logging.INFO, otherwise will only output for level WARNINGS and above


if device != "cpu":
    ngpus = torch.cuda.device_count()
    logger.info(f"NGPUS = {ngpus}")

if model == "ann":
    logger.info(
        f"Training the {stencil}x{stencil} ANN-CNNs, {domain} horizontal and {vertical} vertical model with features {features} with min-max learning rates {lr_min} to {lr_max} for a CyclicLR, and dropout={dropout}.\n"
    )
elif model == "attention":
    logger.info(
        f"Training the Attention UNet, {domain} horizontal and {vertical} vertical model with features {features} with min-max learning rates {lr_min} to {lr_max} for a CyclicLR, and dropout={dropout}.\n"
    )

idir = str(args.input_dir) + "/"
odir = str(args.output_dir) + "/"

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
        test_files.append(f"{pre}{year}_constant_mu_sigma_scaling{str(months).zfill(2)}.nc")

logger.info(
    f"Training the {domain} horizontal and {vertical} vertical model, with features {features} with min-max learning rates {lr_min} to {lr_max}, and dropout={dropout}. Starting from epoch {init_epoch}. Training on years {train_years} and testing on years {test_years}.\n"
)

logger.info("Defined input files")
logger.info(f"Model = {model}")
logger.info(f"train batch size = {bs_train}")
logger.info(f"validation batch size = {bs_test}")

if model == "ann":
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
    logger.info(f"Input dim: {idim}, hidden dim: {hdim}, output dim: {odim}")

    if not ablation:
        model = ANN_CNN(idim=idim, odim=odim, hdim=hdim, dropout=dropout, stencil=trainset.stencil)
    else:
        model = ANN_CNN10(
            idim=idim, odim=odim, hdim=hdim, dropout=dropout, stencil=trainset.stencil
        )
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
    logger.info(
        f"model created. \n --- model size: {model.totalsize():.2f} MBs,\n --- Num params: {model.totalparams()/10**6:.3f} mil. "
    )

    if not ablation:
        file_prefix = (
            odir + f"{vertical}/ann_cnn_{stencil}x{stencil}_{domain}_{vertical}_era5_{features}_"
        )
    else:
        file_prefix = (
            odir
            + f"{vertical}/ann_cnn_{stencil}x{stencil}_{domain}_{vertical}_era5_{features}_ABLATION_10hiddenlayers_"
        )
    if restart:
        # load checkpoint before resuming training
        PATH = f"{file_prefix}_train_epoch{init_epoch-1}.pt"
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info("Starting training ...")
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
    )

elif model == "attention":
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

    logger.info(f"Input channel: {ch_in}")
    logger.info(f"Output channel: {ch_out}")

    model = Attention_UNet(ch_in=ch_in, ch_out=ch_out, dropout=dropout)
    # port model to GPU. ensures optimizer is loaded to GPU as well
    model = model.to(device)
    logger.info(
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
    )


logger.info("Model training complete")
