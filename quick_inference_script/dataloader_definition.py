import numpy as np
import xarray as xr
import pandas as pd
import os
import torch


# Customized Dataloader - good for loading batches from:
# -- carefully designed input files for both single-column and nonlocal global training, and
# -- single point single-column and nonlocal training

# tricks to speed up pytorch dataloading: https://gist.github.com/ZijiaLewisLu/eabdca955110833c0ce984d34eb7ff39
# dask scheduling suggestion: https://discuss.pytorch.org/t/problems-using-dataloader-for-dask-xarray-netcdf-data/108270
# dask.config.set(scheduler='synchronous')
# ===================================================================================


# Dataloader for native training and transfer learning of single column ANNs and nonlocal ANN+CNN models both in the troposphere and the stratosphere
class Dataset_ANN_CNN(torch.utils.data.Dataset):
    def __init__(self, files, domain, vertical, stencil, manual_shuffle, features, region="1andes"):
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

        # self.bs = batch_size
        self.domain = domain  # acceptable values: singlepoint, regional, global
        self.vertical = vertical
        self.stencil = stencil  # for nonlocal training
        self.fac = int(self.stencil / 2.0)
        self.manual_shuffle = manual_shuffle

        if self.vertical == "global":
            # 122 channels for each feature
            if self.features == "uvtheta":
                self.v = np.arange(0, 369)  # for u,v,theta
            elif self.features == "uvthetaw":
                self.v = np.arange(0, 491)  # for u,v,theta,w
            elif self.features == "uvw":
                self.v = np.concatenate(
                    (np.arange(0, 247), np.arange(369, 491)), axis=0
                )  # for u,v,w
        elif self.vertical == "stratosphere_only":
            # 60 channels for each feature
            if self.features == "uvtheta":
                self.v = np.arange(0, 183)  # for u,v,theta
            elif self.features == "uvthetaw":
                self.v = np.arange(0, 243)  # for u,v,theta,w
            elif self.features == "uvw":
                self.v = np.concatenate(
                    (np.arange(0, 123), np.arange(183, 243)), axis=0
                )  # for u,v,w
            elif self.features == "uvthetaN2":
                self.v = np.concatenate(
                    (np.arange(0, 183), np.arange(243, 303)), axis=0
                )  # for u,v,theta,N2
            elif self.features == "uvthetawN2":
                self.v = self.v = np.arange(0, 303)  # for u,v,theta,w,N2

        self.idim = len(self.v)

        # create permutations
        if self.manual_shuffle:
            self.seed = 51
            self.tperm = np.random.RandomState(seed=self.seed).permutation(np.arange(0, self.nt))

        if self.domain == "singlepoint":
            self.y0 = 14
            self.x0 = 102

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
        if self.domain == "singlepoint":
            return self.nt
        else:
            return self.nt

    def __getitem__(self, ind):
        if self.manual_shuffle:
            it = self.tperm[ind]
        else:
            it = ind

        if self.domain == "singlepoint":
            # Note: assumes that the file is four dimensional (time, channels, lat, lon)
            if self.stencil == 1:
                I = torch.from_numpy(self.inp[it, self.v, self.y0, self.x0].data.compute())
                O = torch.from_numpy(self.out[it, :, self.y0, self.x0].data.compute())
                return I, O
            else:
                # Add boundary conditions!
                I = torch.from_numpy(
                    self.inp[
                        it,
                        self.v,
                        self.y0 - self.fac : self.y0 + self.fac + 1,
                        self.x0 - self.fac : self.x0 + self.fac + 1,
                    ].data.compute()
                )
                O = torch.from_numpy(self.out[it, :, self.y0, self.x0].data.compute())
                return I, O

        elif self.domain == "regional":
            if self.stencil == 1:
                I = torch.from_numpy(self.inp[it, self.v, y1:y2, x1:x2].data.compute())
                O = torch.from_numpy(self.out[it, :, y1:y2, x1:x2].data.compute())

                # print(I.shape)
                # print(O.shape)
                # reshape
                I = torch.permute(I, (1, 2, 0))
                O = torch.permute(O, (1, 2, 0))
                S = I.shape
                I = I.reshape(S[0] * S[1], -1)
                S = O.shape
                O = O.reshape(S[0] * S[1], -1)

                return I, O

            else:
                # (time x pressure x lat x lon x st x st)
                # convolution layer will be applied on the last two dimensions
                I = torch.from_numpy(
                    self.inp[it, self.v, self.y1 : self.y2, self.x1 : self.x2, :, :].data.compute()
                )
                O = torch.squeeze(
                    torch.from_numpy(
                        self.out[
                            it, :, self.y1 : self.y2, self.x1 : self.x2, self.fac, self.fac
                        ].data.compute()
                    )
                )
                # reorder it
                I = torch.permute(I, (1, 2, 0, 3, 4))
                O = torch.permute(O, (1, 2, 0))
                S = I.shape
                I = I.reshape(S[0] * S[1], S[2], S[3], S[4])
                S = O.shape
                O = O.reshape(S[0] * S[1], -1)

                return I, O

        elif self.domain == "global":
            if self.stencil == 1:
                I = torch.from_numpy(self.inp[it, self.v, :, :].data.compute())
                O = torch.from_numpy(self.out[it, :, :, :].data.compute())

                # print(I.shape)
                # print(O.shape)
                # reshape
                I = torch.permute(I, (1, 2, 0))
                O = torch.permute(O, (1, 2, 0))
                S = I.shape
                I = I.reshape(S[0] * S[1], -1)
                S = O.shape
                O = O.reshape(S[0] * S[1], -1)

                return I, O

            else:
                # (time x pressure x lat x lon x st x st)
                # convolution layer will be applied on the last two dimensions
                I = torch.from_numpy(self.inp[it, self.v, :, :, :, :].data.compute())
                O = torch.squeeze(
                    torch.from_numpy(self.out[it, :, :, :, self.fac, self.fac].data.compute())
                )
                # first reduce to a 4D tensor by vectorising to (time*lat*lon x pressure x st x st) shape
                I = torch.permute(I, (1, 2, 0, 3, 4))
                O = torch.permute(O, (1, 2, 0))
                S = I.shape
                I = I.reshape(S[0] * S[1], S[2], S[3], S[4])
                S = O.shape
                O = O.reshape(S[0] * S[1], -1)

                return I, O

    def refresh_index(self):
        self.index = 1
        return True

    def return_ds(self):
        return self.ds


# Dataloader for native training and transfer learning of Attention Unet models both in the troposphere and the stratosphere
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
                    self.out[it, :, self.y1 : self.y2, self.x1 : self.x2].data.compute()
                )
            )

            return I, O

        elif self.domain == "global":
            I = torch.from_numpy(self.inp[it, self.v, :, :].data.compute())
            O = torch.from_numpy(self.out[it, :, :, :].data.compute())

            return I, O

    def refresh_index(self):
        self.index = 1
        return True

    def return_ds(self):
        return self.ds
