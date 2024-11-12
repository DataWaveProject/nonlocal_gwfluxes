# Dataloader for attention unet only

import numpy as np
import xarray as xr
import pandas as pd
import os
import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim


class Dataset_AttentionUNet(torch.utils.data.Dataset):
    def __init__(self, files, domain, vertical, manual_shuffle, features, region="1andes"):
        # domain = regional or global
        # vertical = global or stratosphere_only

        # super().__init__()

        _files = sorted(files)

        # this concatenates all the files into a single index
        # chunk={"time":1} works the fastest
        ds: xr.Dataset = xr.open_mfdataset(
            paths=_files, chunks={"time": 1}, combine="nested", concat_dim="time"
        )

        self.ds = ds

        # dimensions
        self.idim = len(self.ds["idim"])
        self.odim = len(self.ds["odim"])
        self.features = features
        self.inp = self.ds["features"]
        self.out = self.ds["output"]
        self.lat = self.ds["lat"]
        self.lon = self.ds["lon"]

        self.nt = len(self.ds.time)

        self.domain = domain  # acceptable values: singlepoint, regional, global
        self.vertical = vertical
        self.manual_shuffle = manual_shuffle

        # omitting lat, lon, zs for attention unet. To include, change 3 -> 0
        if self.vertical == "global":
            # 122 channels for each feature
            if self.features == "uvtheta":
                self.v = np.arange(3, 369)  # for u,v,theta
            elif self.features == "uvthetaw":
                self.v = np.arange(3, 491)  # for u,v,theta,w
            elif self.features == "uvw":
                self.v = np.concatenate(
                    (np.arange(3, 247), np.arange(369, 491)), axis=0
                )  # for u,v,w
            self.w = np.arange(0, self.odim)  # all vertical channels

        elif self.vertical == "stratosphere_only":
            # 60 channels for each feature
            if self.features == "uvtheta":
                self.v = np.arange(3, 183)  # for u,v,theta
            elif self.features == "uvthetaw":
                self.v = np.arange(3, 243)  # for u,v,theta,w
            elif self.features == "uvw":
                self.v = np.concatenate(
                    (np.arange(3, 123), np.arange(183, 243)), axis=0
                )  # for u,v,w
            elif self.features == "uvthetaN2":
                self.v = np.concatenate(
                    (np.arange(3, 183), np.arange(243, 303)), axis=0
                )  # for u,v,theta,N2
            elif self.features == "uvthetawN2":
                self.v = self.v = np.arange(3, 303)  # for u,v,theta,w,N2
            self.w = np.arange(0, self.odim)  # all vertical channels

        elif self.vertical == "stratosphere_update":
            # 122 channels for each feature
            if self.features == "uvtheta":
                self.v = np.arange(3, 369)  # for u,v,theta
            elif self.features == "uvthetaw":
                self.v = np.arange(3, 491)  # for u,v,theta,w
            elif self.features == "uvw":
                self.v = np.concatenate(
                    (np.arange(3, 247), np.arange(369, 491)), axis=0
                )  # for u,v,w
            self.w = np.concatenate(
                (np.arange(0, 60), np.arange(122, 182)), axis=0
            )  # select upper 60 channels for uw and upper 60 for vw
            self.odim = len(self.w)  # update self.odim accordingly

        self.idim = len(self.v)

        # create permutations
        if self.manual_shuffle:
            self.seed = 51
            self.tperm = np.random.RandomState(seed=self.seed).permutation(np.arange(0, self.nt))

        if self.domain == "regional":
            if self.region == "1andes":
                self.y1 = 3
                self.y2 = 21
                self.x1 = 96
                self.x2 = 113
            if self.region == "2scand":
                self.y1 = 45
                self.y2 = 58
                self.x1 = 0
                self.x2 = 12
            if self.region == "3himalaya":
                self.y1 = 41
                self.y2 = 54
                self.x1 = 26
                self.x2 = 44
            if self.region == "4newfound":
                self.y1 = 47
                self.y2 = 58
                self.x1 = 103
                self.x2 = 119
            if self.region == "5south_ocn":
                self.y1 = 8
                self.y2 = 17
                self.x1 = 10
                self.x2 = 25
            if self.region == "6se_asia":
                self.y1 = 33
                self.y2 = 42
                self.x1 = 32
                self.x2 = 49
            if self.region == "7natlantic":
                self.y1 = 31
                self.y2 = 44
                self.x1 = 112
                self.x2 = 124
            if self.region == "8npacific":
                self.y1 = 27
                self.y2 = 47
                self.x1 = 67
                self.x2 = 87

    def __len__(self):
        return self.nt
        # if self.domain == 'singlepoint':
        #    return (self.nt)
        # else:
        #    return (self.nt)

    def __getitem__(self, ind):
        if self.manual_shuffle:
            it = self.tperm[ind]
        else:
            it = ind

        if self.domain == "regional":
            I = torch.from_numpy(
                self.inp[it, self.v, self.y1 : self.y2, self.x1 : self.x2].data.compute()
            )
            O = torch.squeeze(
                torch.from_numpy(
                    self.out[it, self.w, self.y1 : self.y2, self.x1 : self.x2].data.compute()
                )
            )

            return I, O

        elif self.domain == "global":
            I = torch.from_numpy(self.inp[it, self.v, :, :].data.compute())
            O = torch.from_numpy(self.out[it, self.w, :, :].data.compute())

            return I, O

    def refresh_index(self):
        self.index = 1
        return True

    def return_ds(self):
        return self.ds
