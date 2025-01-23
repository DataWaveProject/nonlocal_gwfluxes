# Inference script for DETERMINISTIC inference on trained models: both ANNs and UNets
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
from pathlib import Path

# -------- for data parallelism ----------
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.append("../utils/")
from dataloader_definition import Dataset_ANN_CNN, Dataset_AttentionUNet
from model_definition import ANN_CNN, ANN_CNN10, Attention_UNet
from function_training import Inference_and_Save_ANN_CNN, Inference_and_Save_AttentionUNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
parser.add_argument("-e", "--epoch", type=int, default=100, help="Epoch to use for inference")
parser.add_argument(
    "-m",
    "--month",
    type=int,
    choices=range(1, 13),
    metavar="{1,2,...,11,12}",
    default=1,
    help="Month to run inference on",
)
parser.add_argument(
    "-s", "--stencil", type=int, choices=[1, 3, 5], default=1, help="Horizontal stencil for the NN"
)
parser.add_argument(
    "-t",
    "--teston",
    choices=["era5", "ifs"],
    default="era5",
    help="Dataset on which to test the model",
)
parser.add_argument(
    "-i",
    "--input_dir",
    default=Path.cwd(),
    help="Input directory to fetch validation data",
    type=Path,
)
parser.add_argument("-c", "--ckpt_dir", default=Path.cwd(), help="Checkpoint directory", type=Path)
parser.add_argument(
    "-o", "--output_dir", default=Path.cwd(), help="Output directory to save outputs", type=Path
)
args = parser.parse_args()

# print parsed args
print(f"model={args.model}")
print(f"horizontal={args.horizontal}")
print(f"vertical={args.vertical}")
print(f"features={args.features}")
print(f"epoch={args.epoch}")
print(f"stencil={args.stencil}")
print(f"month={args.month}")
print(f"checkpoint_dir={args.ckpt_dir}")
print(f"input_dir={args.input_dir}")
print(f"output_dir={args.output_dir}")


bs_train = 40  # 80 (80 works for most). (does not work for global uvthetaw)
bs_test = bs_train

# --------------------------------------------------
model = args.model
domain = args.horizontal
vertical = args.vertical
features = args.features  # sys.argv[2]  #'uvthetaw' # 'uvtheta', ''uvthetaw', or 'uvw' for troposphere | additionally 'uvthetaN2' and 'uvthetawN2' for stratosphere_only
dropout = 0
epoch = args.epoch
stencil = args.stencil
teston = args.teston

# ----- model sanity check ----------
if model == "attention" and stencil > 1:
    print(
        "The selected model is Attention UNet but stencil > 1 is only allowed for ANNs. Overriding stencil value to 1."
    )
    stencil = 1
if stencil % 2 == 0:
    raise ValueError("Stencil must be odd.")

# -------- model checkpoint ------------
idir = str(args.input_dir) + "/"
odir = str(args.output_dir) + "/"
pref = str(args.ckpt_dir) + "/"  # "/scratch/users/ag4680/torch_saved_models/attention_unet/"
if model == "ann":
    ckpt = f"ann_cnn_{stencil}x{stencil}_{domain}_{vertical}_era5_{features}__train_epoch{epoch}.pt"
    log_filename = f"./{teston}_inference_ann_cnn_{stencil}x{stencil}_{domain}_{vertical}_{features}_ckpt_epoch_{epoch}.txt"
elif model == "attention":
    ckpt = (
        f"attnunet_era5_{domain}_{vertical}_{features}_mseloss_train_epoch{str(epoch).zfill(2)}.pt"
    )
    log_filename = (
        f"./{teston}_inference_attnunet_{domain}_{vertical}_{features}_ckpt_epoch_{epoch}.txt"
    )

logger = logging.getLogger(__name__)
logging.basicConfig(filename=log_filename, level=logging.INFO)

if device != "cpu":
    ngpus = torch.cuda.device_count()
    logger.info(f"NGPUS = {ngpus}")


# Define test files
# ------- To test on one year of ERA5 data
test_files = []
test_years = np.array([2015])
test_month = args.month  # int(sys.argv[4])  # np.arange(1,13)
logger.info(f"Inference for month {test_month}")
if teston == "era5":
    if vertical == "stratosphere_only":
        if stencil == 1:
            pre = (
                idir
                + f"stratosphere_1x1_inputfeatures_u_v_theta_w_N2_uw_vw_era5_training_data_hourly_"
            )
        else:
            pre = (
                idir
                + f"stratosphere_nonlocal_{stencil}x{stencil}_inputfeatures_u_v_theta_w_N2_uw_vw_era5_training_data_hourly_"
            )
    elif vertical == "global" or vertical == "stratosphere_update":
        if stencil == 1:
            pre = idir + f"1x1_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_"
        else:
            pre = (
                idir
                + f"nonlocal_{stencil}x{stencil}_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_"
            )

    for year in test_years:
        for months in np.arange(test_month, test_month + 1):
            test_files.append(f"{pre}{year}_constant_mu_sigma_scaling{str(months).zfill(2)}.nc")

elif teston == "ifs":
    if vertical == "stratosphere_only":
        test_files = [
            idir
            + f"stratosphere_only_{stencil}x{stencil}_inputfeatures_u_v_theta_w_N2_uw_vw_era5_training_data_hourly_constant_mu_sigma_scaling.nc"
        ]
    elif vertical == "global" or vertical == "stratosphere_update":
        test_files = [
            idir
            + f"troposphere_and_stratosphere_{stencil}x{stencil}_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_constant_mu_sigma_scaling.nc"
        ]


logger.info(
    f"Inference the {model} model on {domain} horizontal and {vertical} vertical model, with features {features} and dropout={dropout}. Testing on {teston} dataset."
)
logger.info(f"Test files = {test_files}")


if model == "ann":
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
        testset, batch_size=bs_test, drop_last=False, shuffle=False, num_workers=8
    )

    idim = testset.idim
    odim = testset.odim
    hdim = 4 * idim

    # ---- define model
    model = ANN_CNN(idim=idim, odim=odim, hdim=hdim, dropout=dropout, stencil=stencil)
    loss_fn = nn.MSELoss()
    logger.info(
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
    if teston == "era5":
        out = odir + f"inference_{S[0]}_{test_years[0]}_{test_month}.nc"
    elif teston == "ifs":
        out = odir + f"inference_{S[0]}_testedonIFS.nc"
    logger.info(f"Output NC file: {out}")

    # better to create the file within the inference_and_save function
    logger.info("Initiating inference")
    Inference_and_Save_ANN_CNN(model, testset, testloader, bs_test, device, stencil, logger, out)

elif model == "attention":
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
    checkpoint = torch.load(PATH, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # create netCDF file
    S = ckpt.split(".")
    if teston == "era5":
        out = odir + f"inference_{S[0]}_{test_years[0]}_{test_month}.nc"
    elif teston == "ifs":
        out = odir + f"inference_{S[0]}_testedonIFS.nc"
    logger.info(f"Output NC file: {out}")

    # better to create the file within the inference_and_save function
    logger.info("Initiating inference")
    Inference_and_Save_AttentionUNet(model, testset, testloader, bs_test, device, logger, out)

logger.info("Inference complete")
