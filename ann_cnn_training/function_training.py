from netCDF4 import Dataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# For ANNs and ANN+CNNs
def Training_ANN_CNN(
    nepochs,
    model,
    optimizer,
    loss_fn,
    trainloader,
    testloader,
    stencil,
    bs_train,
    bs_test,
    save,
    file_prefix,
    device,
    logger,
    init_epoch=1,
    scheduler=0,
):
    # print('Training')
    LOSS_TRAIN = np.zeros((nepochs))
    LOSS_TEST = np.zeros((nepochs))

    logger.info(f"In training loop ...", file=log)
    logger.info("In training loop ...")
    for epoch in np.arange(init_epoch + 0, init_epoch + nepochs):
        # --------- training ----------
        model.train()
        trainloss = 0.0
        count = 0.0
        for i, (inp, out) in enumerate(trainloader):
            # print(i)
            inp = inp.to(device)
            out = out.to(device)
            if stencil == 1:
                S = inp.shape
                inp = inp.reshape(S[0] * S[1], S[2])
                S = out.shape
                out = out.reshape(S[0] * S[1], -1)
            elif stencil > 1:
                S = inp.shape
                inp = inp.reshape(S[0] * S[1], S[2], S[3], S[4])
                S = out.shape
                out = out.reshape(S[0] * S[1], -1)
            pred = model(inp)
            loss = loss_fn(pred, out)  # loss_fn(pred*fac,out*fac) #+ weight_decay*l2_norm  #/fac) +
            optimizer.zero_grad()  # flush the gradients from the last step and set to zeros, they accumulate otherwise
            # backward propagation
            loss.backward()
            # parameter update step
            # print('5')
            optimizer.step()
            if scheduler != 0:
                scheduler.step()
            trainloss += loss  # .item()#.item()
            count += 1

        LOSS_TRAIN[epoch - 1 - init_epoch] = trainloss / count

        # --------- testing ------------
        model.eval()
        # print('===== TESTING ============')
        testloss = 0.0
        count = 0.0
        for i, (inp, out) in enumerate(testloader):
            inp = inp.to(device)
            out = out.to(device)
            if stencil == 1:
                S = inp.shape
                inp = inp.reshape(S[0] * S[1], S[2])
                S = out.shape
                out = out.reshape(S[0] * S[1], -1)
            elif stencil > 1:
                S = inp.shape
                inp = inp.reshape(S[0] * S[1], S[2], S[3], S[4])
                S = out.shape
                out = out.reshape(S[0] * S[1], -1)
            pred = model(inp)
            loss2 = loss_fn(pred, out)
            testloss += loss2.item()
            count += 1

        LOSS_TEST[epoch - 1 - init_epoch] = testloss / count

        logger.info(
            f"Epoch {epoch}, {(epoch-init_epoch+1)}/{nepochs}, training mseloss: {LOSS_TRAIN[epoch-1-init_epoch]:.6f}, testing mseloss: {LOSS_TEST[epoch-1-init_epoch]:.6f}"
        )
        logger.info(
            f"Epoch {epoch}, {(epoch-init_epoch+1)}/{nepochs}, training mseloss: {LOSS_TRAIN[epoch-1-init_epoch]:.6f}, testing mseloss: {LOSS_TEST[epoch-1-init_epoch]:.6f}"
        )
        # Saving the model at any given epoch
        if save:
            savepath = f"{file_prefix}_train_epoch{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss_fn,
                    "loss_train": LOSS_TRAIN,
                    "loss_test": LOSS_TEST,
                    "activation": "LeakyRelu()",
                    "scheduler": "CyclicLR",
                },
                savepath,
            )

    return model, LOSS_TRAIN, LOSS_TEST


def Inference_and_Save_ANN_CNN(
    model, testset, testloader, bs_test, device, stencil, logger, outfile
):
    # ---------------------------------------------------------------------------------------
    idim = testset.idim
    odim = testset.odim
    lat = testset.lat * 90.0
    lon = testset.lon * 360.0
    ny = len(lat)
    nx = len(lon)
    # print([idim,odim,ny,nx])

    # create netcdf file
    out = Dataset(outfile, "w", format="NETCDF4")
    otime = out.createDimension("time", None)
    # oidim  = out.createDimension("idim", idim)
    oodim = out.createDimension("odim", odim)
    olat = out.createDimension("lat", ny)
    olon = out.createDimension("lon", nx)

    times = out.createVariable("time", "i4", ("time",))
    times.units = "hourly timestep of the month"
    odims = out.createVariable("odim", "i4", ("odim",))
    odims.units = "output channels"
    lats = out.createVariable("lat", "f4", ("lat",))
    lats.units = "degrees_north"
    lons = out.createVariable("lon", "f4", ("lon",))
    lons.units = "degrees_east"

    o_output = out.createVariable(
        "output",
        "f4",
        (
            "time",
            "odim",
            "lat",
            "lon",
        ),
    )
    o_output.units = "ERA5 {uw,vw} true output"
    o_pred = out.createVariable(
        "prediction",
        "f4",
        (
            "time",
            "odim",
            "lat",
            "lon",
        ),
    )
    o_pred.units = "ERA5 {uw,vw} attention unet prediction"

    lats[:] = lat[:]
    lons[:] = lon[:]
    odims[:] = np.arange(1, odim + 1)
    # ----------------------------------------------------------------------------------------

    model.eval()
    model.dropout.train()  # this enables dropout during inference. By default dropout is OFF when model.eval()=True
    testloss = 0.0
    count = 0
    for i, (INP, OUT) in enumerate(testloader):
        # print(i)
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

        S = PRED.shape
        # print(f'S[0]:{S[0]}, S[0]/(nx*ny) = {S[0]/(nx*ny)}')
        nt = int(S[0] / (nx * ny))
        if count == 0:
            logger.info(f"Minibatch={i}, count={count}, output shape={S}")

        # print(f'OUT.shape = {OUT.shape}')
        # print(f'PRED.shape = {PRED.shape}')

        # Reshape input and output from (batch_size*ny*nx,nz) to (batch_size,nz,ny,nx)
        OUT = OUT.reshape(nt, ny, nx, odim)
        OUT = torch.permute(OUT, (0, 3, 1, 2))
        PRED = PRED.reshape(nt, ny, nx, odim)
        PRED = torch.permute(PRED, (0, 3, 1, 2))

        # print(f'New OUT.shape = {OUT.shape}')
        # print(f'New PRED.shape = {PRED.shape}')

        # write to netCDF
        if device != "cpu":
            # print('Writing')
            o_output[count : count + nt, :, :, :] = OUT[:].detach().cpu().numpy()
            o_pred[count : count + nt, :, :, :] = PRED[:].detach().cpu().numpy()
        else:
            # print('Writing')
            o_output[count : count + nt, :, :, :] = OUT[:].numpy()
            o_pred[count : count + nt, :, :, :] = PRED[:].numpy()
        count += nt

    out.close()


def Inference_and_Save_AttentionUNet(model, testset, testloader, bs_test, device, logger, outfile):
    # ---------------------------------------------------------------------------------------
    idim = testset.idim
    odim = testset.odim
    lat = testset.lat * 90.0
    lon = testset.lon * 360.0
    ny = len(lat)
    nx = len(lon)
    # print([idim,odim,ny,nx])

    # create netcdf file
    out = Dataset(outfile, "w", format="NETCDF4")
    otime = out.createDimension("time", None)
    # oidim  = out.createDimension("idim", idim)
    oodim = out.createDimension("odim", odim)
    olat = out.createDimension("lat", ny)
    olon = out.createDimension("lon", nx)

    times = out.createVariable("time", "i4", ("time",))
    times.units = "hourly timestep of the month"
    odims = out.createVariable("odim", "i4", ("odim",))
    odims.units = "output channels"
    lats = out.createVariable("lat", "f4", ("lat",))
    lats.units = "degrees_north"
    lons = out.createVariable("lon", "f4", ("lon",))
    lons.units = "degrees_east"

    o_output = out.createVariable(
        "output",
        "f4",
        (
            "time",
            "odim",
            "lat",
            "lon",
        ),
    )
    o_output.units = "ERA5 {uw,vw} true output"
    o_pred = out.createVariable(
        "prediction",
        "f4",
        (
            "time",
            "odim",
            "lat",
            "lon",
        ),
    )
    o_pred.units = "ERA5 {uw,vw} attention unet prediction"

    lats[:] = lat[:]
    lons[:] = lon[:]
    odims[:] = np.arange(1, odim + 1)
    # ----------------------------------------------------------------------------------------

    model.eval()
    model.dropout.train()  # this enables dropout during inference. By default dropout is OFF when model.eval()=True
    count = 0
    for i, (INP, OUT) in enumerate(testloader):
        # print([i,count])
        INP = INP.to(device)
        S = OUT.shape
        o_output[count : count + S[0], :, :, :] = OUT[
            :
        ].numpy()  # write before porting to GPU itself
        OUT = OUT.to(device)
        S = OUT.shape
        if count == 0:
            logger.info(f"Minibatch={i}, count={count}, output shape={S}")
        PRED = model(INP)
        # write to netCDF
        if device != "cpu":
            # print('Writing')
            o_pred[count : count + S[0], :, :, :] = PRED[:].detach().cpu().numpy()
        count = count + S[0]

    out.close()
