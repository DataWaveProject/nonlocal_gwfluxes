# Uses model checkpoints from hugging face and two snapshot input to create a two snapshot output

# BOTH model checkpoints and test files are stored on hugging face: https://huggingface.co/amangupta2/nonlocal_gwfluxes/tree/main

import numpy as np

# import xarray as xr
# import pandas as pd
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim

# -----------------------------------------------------------
import logging
import argparse

# -------- for data parallelism ----------
# from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.distributed as dist
# import torch.multiprocessing as mp

# from dataloader_definition import Dataset_ANN_CNN
from model_definition import ANN_CNN, Attention_UNet
# from function_training import Inference_and_Save_ANN_CNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

domain = "global"
vertical = "global"  # global or stratosphere_update
features = "uvthetaw"

# turn these into batch arguments
# needed to load the appropriate model
model = "attention"  # or 'ann'
stencil = 3
epoch = 94

if model == "attention":
    stencil = 1


# assumes ckpts stored in model_ckpt dir
if model == "attention":
    PATH = f"/glade/derecho/scratch/agupta/hugging_face_checkpoints/attnunet_era5_{domain}_{vertical}_{features}_mseloss_train_epoch{epoch}.pt"
elif model == "ann" and stencil == 1:
    PATH = f"/glade/derecho/scratch/agupta/hugging_face_checkpoints/ann_cnn_{stencil}x{stencil}_{domain}_{vertical}_era5_{features}__train_epoch{epoch}.pt"

# eventually we will not need the files above. Just reading from the netCDF directly right now

# idim, odim, ch_in, ch_out
# input: vertical+features
DIMS = {
    "global" + "uvw": [369, 244, 366, 244],
    "global" + "uvtheta": [369, 244, 366, 244],
    "global" + "uvthetaw": [491, 244, 488, 244],
    "stratosphere_update" + "uvw": [369, 120, 366, 120],
    "stratosphere_update" + "uvtheta": [369, 120, 366, 120],
    "stratosphere_update" + "uvthetaw": [491, 120, 488, 120],
}

# defining and loading model - can this be further simplified/optimized?
# Loading the model into memory each time is not very efficient!
if model == "ann":
    idim = DIMS[vertical + features][0]
    odim = DIMS[vertical + features][1]
    hdim = 4 * idim

    model = ANN_CNN(idim=idim, odim=odim, hdim=hdim, dropout=dropout, stencil=stencil)
    loss_fn = nn.MSELoss()
    checkpoint = torch.load(PATH, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

elif model == "attention":
    ch_in = DIMS[vertical + features][2]
    ch_out = DIMS[vertical + features][3]

    model = Attention_UNet(ch_in=ch_in, ch_out=ch_out, dropout=dropout)
    loss_fn = nn.MSELoss()
    checkpoint = torch.load(PATH, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

# Load input into INP - single batch - as we would have from CAM
OUT = model(INP)


# turn this into a function eventually
