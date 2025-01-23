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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


print(device)
