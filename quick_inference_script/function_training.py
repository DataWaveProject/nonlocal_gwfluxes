import numpy as np
import torch
import xarray as xr


def Inference_and_Save_ANN_CNN(model, testset, testloader, bs_test, device, stencil, outfile):
    # ---------------------------------------------------------------------------------------
    idim = testset.idim
    odim = testset.odim
    lat = testset.lat * 90.0
    lon = testset.lon * 360.0
    ny = len(lat)
    nx = len(lon)

    model.eval()
    # model.dropout.train()  # this enables dropout during inference. By default dropout is OFF when model.eval()=True
    testloss = 0.0
    count = 0
    print("hi")
    for i, (INP, OUT) in enumerate(testloader):
        INP = INP.to(device)
        OUT = OUT.to(device)
        if stencil == 1:
            T = INP.shape
            INP = INP.reshape(T[0] * T[1], T[2])
            T = OUT.shape
            OUT = OUT.reshape(T[0] * T[1], -1)
        elif stencil > 1:
            T = INP.shape
            INP = INP.reshape(T[0] * T[1], T[2], T[3], T[4])
            T = OUT.shape
            OUT = OUT.reshape(T[0] * T[1], -1)
        PRED = model(INP)

        print("saving data...")
        data = {
            "input": INP,
            "output": OUT,
            "predict": PRED,
        }

        for k, v in data.items():
            xdata = xr.DataArray(v.detach().cpu().numpy())
            xdata.to_netcdf(k + ".nc")
            print(f"{k} = ", v.detach().cpu().numpy())

        S = PRED.shape
        nt = int(S[0] / (nx * ny))

        # Reshape input and output from (batch_size*ny*nx,nz) to (batch_size,nz,ny,nx)
        OUT = OUT.reshape(nt, ny, nx, odim)
        OUT = torch.permute(OUT, (0, 3, 1, 2))
        PRED = PRED.reshape(nt, ny, nx, odim)
        PRED = torch.permute(PRED, (0, 3, 1, 2))

        # # write to .npz
        # if device != "cpu":
        #     np.savez(
        #         outfile,
        #         output=OUT[:].detach().cpu().numpy(),
        #         prediction=PRED[:].detach().cpu().numpy(),
        #     )
        # else:
        #     np.savez(outfile, output=OUT[:].numpy(), prediction=PRED[:].numpy())
        count += 1


def Inference_and_Save_AttentionUNet(model, testset, testloader, bs_test, device, outfile):
    # ---------------------------------------------------------------------------------------
    idim = testset.idim
    odim = testset.odim
    lat = testset.lat * 90.0
    lon = testset.lon * 360.0
    ny = len(lat)
    nx = len(lon)

    model.eval()
    model.dropout.train()  # this enables dropout during inference. By default dropout is OFF when model.eval()=True
    count = 0
    for i, (INP, OUT) in enumerate(testloader):
        INP = INP.to(device)
        S = OUT.shape
        OUT = OUT.to(device)
        S = OUT.shape
        PRED = model(INP)

        # write to .npz
        if device != "cpu":
            np.savez(
                outfile,
                output=OUT[:].detach().cpu().numpy(),
                prediction=PRED[:].detach().cpu().numpy(),
            )
        else:
            np.savez(outfile, output=OUT[:].numpy(), prediction=PRED[:].numpy())
        count = count + S[0]
