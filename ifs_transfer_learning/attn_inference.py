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

# -----------------------------------------------------------
import logging
import argparse
from pathlib import Path

# -------- for data parallelism ----------
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.append("../utils/")
from dataloader_definition import Dataset_AttentionUNet
from model_definition import Attention_UNet
from function_training import Inference_and_Save_AttentionUNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--month",
    type=int,
    choices=range(1, 13),
    metavar="{1,2,...,11,12}",
    help="Month to run inference on. Only valid when tested on set to ERA5",
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
    "-t",
    "--teston",
    choices=["ERA5", "IFS"],
    help="Model to run inference on",
)
parser.add_argument(
    "-e",
    "--epoch",
    type=int,
    help="Checkpoint (epoch)of the model to be used for transfer learning",
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
print(f"month={args.month}")
print(f"horizontal={args.horizontal}")
print(f"vertical={args.vertical}")
print(f"features={args.features}")
print(f"epoch={args.epoch}")
print(f"teston={args.teston}")
print(f"input_dir={args.input_dir}")
print(f"checkpoint_dir={args.ckpt_dir}")
print(f"output_dir={args.output_dir}")


bs_train = 40  # 80 (80 works for most). (does not work for global uvthetaw)
bs_test = bs_train

# --------------------------------------------------
domain = args.horizontal  # 'regional'
vertical = args.vertical  #'stratosphere_only' # 'global', or 'stratosphere_update'
features = args.features  # sys.argv[2]  #'uvthetaw' # 'uvtheta', ''uvthetaw', or 'uvw' for troposphere | additionally 'uvthetaN2' and 'uvthetawN2' for stratosphere_only
dropout = 0.0  # can choose this to be non-zero during inference for uncertainty quantification. A little dropout goes a long way. Choose a small value - 0.03ish?
epoch = args.epoch  # int(sys.argv[3])
teston = args.teston  # sys.argv[4]

# model checkpoint
idir = str(args.input_dir) + "/"
odir = str(args.output_dir) + "/"
pref = (
    str(args.ckpt_dir) + "/"
)  # "/scratch/users/ag4680/torch_saved_models/transfer_learning_IFS/attention_unet/"
ckpt = f"TLIFS_attnunet_era5_ifs_{domain}_{vertical}_{features}_mseloss_train_epoch{str(epoch).zfill(2)}.pt"


log_filename = f"./TLIFS_inference_attnunet_{domain}_{vertical}_{features}_ckpt_epoch_{epoch}.txt"
logger = logging.getLogger(__name__)
logging.basicConfig(filename=log_filename, level=logging.INFO)

if device != "cpu":
    ngpus = torch.cuda.device_count()
    logger.info(f"NGPUS = {ngpus}")

# Define test files
# --------- To test on one year of ERA5 data
if teston == "ERA5":
    test_files = []
    test_years = np.array([2015])
    test_month = args.month  # np.arange(1,13)
    logger.info(f"Inference for month {test_month}")
    if vertical == "stratosphere_only":
        pre = (
            idir + f"stratosphere_1x1_inputfeatures_u_v_theta_w_N2_uw_vw_era5_training_data_hourly_"
        )
    elif vertical == "global" or vertical == "stratosphere_update":
        pre = idir + f"1x1_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_"
    for year in test_years:
        for months in np.arange(test_month, test_month + 1):
            test_files.append(f"{pre}{year}_constant_mu_sigma_scaling{str(months).zfill(2)}.nc")

# -------- To test on three months of IFS data
elif teston == "IFS":
    if vertical == "stratosphere_only":
        test_files = [
            idir
            + f"stratosphere_only_1x1_inputfeatures_u_v_theta_w_N2_uw_vw_era5_training_data_hourly_constant_mu_sigma_scaling.nc"
        ]
    elif vertical == "global" or vertical == "stratosphere_update":
        test_files = [
            idir
            + f"troposphere_and_stratosphere_1x1_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_constant_mu_sigma_scaling.nc"
        ]

logger.info(
    f"Inference the Attention UNet model on {domain} horizontal and {vertical} vertical model, with features {features} and dropout={dropout}."
)
logger.info(f"Test files = {test_files}")

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
if dropout == 0:
    if teston == "ERA5":
        out = odir + f"TLIFS_inference_{S[0]}_{test_years[0]}_{test_month}_testedonERA5.nc"
    else:
        out = odir + f"TLIFS_inference_{S[0]}_testedonIFS.nc"
else:
    if teston == "ERA5":
        out = (
            odir + f"TLIFS_inference_{S[0]}_{test_years[0]}_{test_month}_dropoutON_testedonERA5.nc"
        )
    else:
        out = odir + f"TLIFS_inference_{S[0]}_{test_years[0]}_{test_month}_dropoutON_testedonIFS.nc"
logger.info(f"Output NC file: {out}")

# better to create the file within the inference_and_save function
logger.info("Initiating inference")
Inference_and_Save_AttentionUNet(model, testset, testloader, bs_test, device, out)

logger.info("Inference complete")
