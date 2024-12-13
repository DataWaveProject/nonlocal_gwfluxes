# This is for training the Attention U-Net only - not the nonlocal ANN+CNNs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from netCDF4 import Dataset


def Training_AttentionUNet(
    nepochs,
    model,
    optimizer,
    loss_fn,
    trainloader,
    testloader,
    bs_train,
    bs_test,
    save,
    file_prefix,
    device,
    logger,
    init_epoch=1,
    scheduler=0,
):
    LOSS_TRAIN = np.zeros((nepochs))
    LOSS_TEST = np.zeros((nepochs))

    print("Training ...")
    for epoch in np.arange(init_epoch + 0, init_epoch + nepochs):
        # --------- training ----------
        model.train()
        trainloss = 0.0
        count = 0.0
        for i, (inp, out) in enumerate(trainloader):
            # print(i)
            inp = inp.to(device)
            out = out.to(device)
            # print(f'1 {inp.shape}')
            # print(f'2 {out.shape}')
            pred = model(inp)
            # print('predicted')
            loss = loss_fn(pred, out)
            # print('loss-ed')
            optimizer.zero_grad()
            # print('optimized')
            # backward propagation
            loss.backward()
            # print('back propagated')
            # parameter update step
            # print('5')
            optimizer.step()
            if scheduler != 0:
                scheduler.step()
            trainloss += loss  # .item()#.item()
            count += 1
            # print('moving on')

        LOSS_TRAIN[epoch - 1 - init_epoch] = trainloss / count

        # --------- testing ------------
        model.eval()
        # print('===== TESTING ============')
        testloss = 0.0
        count = 0.0
        for i, (inp, out) in enumerate(testloader):
            inp = inp.to(device)
            out = out.to(device)
            pred = model(inp)
            loss2 = loss_fn(pred, out)
            testloss += loss2.item()
            count += 1

        LOSS_TEST[epoch - 1 - init_epoch] = testloss / count

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
