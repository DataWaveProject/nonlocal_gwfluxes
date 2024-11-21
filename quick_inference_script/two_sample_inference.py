# Uses model checkpoints from hugging face and two snapshot input to create a two snapshot output

# BOTH model checkpoints and test files are stored on hugging face: https://huggingface.co/amangupta2/nonlocal_gwfluxes/tree/main

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

domain='global'
vertical='global'
features='uvtheta'

model='attention' # or 'ann'
stencil=3
epoch=94

if model=='attention':
    stencil=1

bs_test=2

# assumes ckpts stored in model_ckpt dir
if model=='attention':
    PATH=f'model_ckpt/attnunet_era5_{domain}_{vertical}_{features}_mseloss_train_epoch{epoch}.pt'
elif model=='ann' and stencil==1:
    PATH=f'model_ckpt/ann_cnn_{stencil}x{stencil}_{domain}_{vertical}_era5_{features}__train_epoch{epoch}.pt'

# path to test_files
if model=='attention':
    test_file=[f'test_files/test_1x1_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling08.nc']
elif model=='ann':
    if stencil==1:
        test_file=[f'test_files/test_1x1_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling08.nc']
    elif stencil==3:
        test_file=[f'test_files/test_nonlocal_3x3_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_2015_constant_mu_sigma_scaling08.nc']

if model=='ann':
    testset     = Dataset_ANN_CNN(files=test_files,domain=domain, vertical=vertical, stencil=stencil, manual_shuffle=False, features=features)
    testloader  = torch.utils.data.DataLoader(testset, batch_size=bs_test,
                                          drop_last=False, shuffle=False, num_workers=0)

    idim  = testset.idim
    odim  = testset.odim
    hdim  = 4*idim  # earlier runs has 2*dim for 5x5 stencil. Stick to 4*dim this time

    # ---- define model
    model = ANN_CNN(idim=idim, odim=odim, hdim=hdim, dropout=dropout, stencil=stencil)
    loss_fn   = nn.MSELoss()
    print(f'Model created. \n --- model size: {model.totalsize():.2f} MBs,\n --- Num params: {model.totalparams()/10**6:.3f} mil. ')
    # ---- load model
    checkpoint=torch.load(PATH, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model=model.to(device)
    model.eval()

elif model=='attention':
    testset     = Dataset_AttentionUNet(files=test_files,domain=domain, vertical=vertical, manual_shuffle=False, features=features)
    testloader  = torch.utils.data.DataLoader(testset, batch_size=bs_train,
                                          drop_last=False, shuffle=False, num_workers=2)

    ch_in  = testset.idim
    ch_out = testset.odim

    # ---- define model
    model = Attention_UNet(ch_in=ch_in, ch_out=ch_out, dropout=dropout)
    loss_fn   = nn.MSELoss()
    print(f'Model created. \n --- model size: {model.totalsize():.2f} MBs,\n --- Num params: {model.totalparams()/10**6:.3f} mil. ')
    # ---- load model
    checkpoint=torch.load(PATH, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
    model=model.to(device)
    model.eval()

out=f"{model}_output.npz"
if model=='ann':
    Inference_and_Save_ANN_CNN(model,testset,testloader,bs_test,device,stencil,out)
elif model=='attention':
    Inference_and_Save_AttentionUNet(model,testset,testloader,bs_test,device,out)
