import logging

import click
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader_definition import Dataset_ANN_CNN
from function_training import Training_ANN_CNN
from model_definition import ANN_CNN

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--config-file", default="train.toml", help="configuration file.", type=click.Path())
@click.option("-i", "--input-dir", help="path to input data", type=click.Path(), required=True)
@click.option("-o", "--output-dir", help="path to output data", type=click.Path(), required=True)
@click.option(
    "--stencil", help="size of stencil e.g., 3 for a 3x3 stencil.", type=int, required=True
)
@click.option(
    "--vertical",
    type=click.Choice(["global", "stratosphere_only", "stratosphere_update"]),
    help="how many vertical levels",
    required=True,
)
@click.option(
    "--domain", type=click.Choice(["global", "regional"]), help="domain type.", required=True
)
@click.option(
    "--features", type=click.Choice(["uvtheta"]), help="features to train on.", default="uvtheta"
)
def main(config_file, stencil, vertical, domain, features, input_dir, output_dir):
    logging.basicConfig(filename="training.log", level=logging.INFO)
    logger.info("Reading config file: %s" % config_file)

    # read configuration toml file
    config = read_config(config_file)

    # add CLI arguments to config options
    config["stencil"] = stencil
    config["vertical"] = vertical
    config["domain"] = domain
    config["features"] = features
    config["input_dir"] = input_dir
    config["output_dir"] = output_dir

    logger.info("start training")
    training(config)
    logger.info("training finished")


def read_config(filename):
    with open(filename, mode="rb") as configfile:
        return tomllib.load(configfile)


def training(config):
    torch.set_printoptions(edgeitems=2)
    torch.manual_seed(config["torch"]["manual_seed"])

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )  # 'cuda' to select all available GPUs

    # parameters
    domain = config["domain"]
    vertical = config["vertical"]
    features = config["features"]
    stencil = config["stencil"]

    # hyperparameters
    lr_min = config["hyperparameters"]["lr_min"]
    lr_max = config["hyperparameters"]["lr_max"]

    if stencil == 1:
        bs_train = 20
        bs_test = bs_train
    else:
        bs_train = 10
        bs_test = bs_train
    dropout = 0.1

    # where to resume. Should have checkpoint saved for init_epoch-1. 1 for fresh runs.
    init_epoch = config["options"]["init_epoch"]
    nepochs = config["options"]["nepochs"]

    if device != "cpu":
        ngpus = torch.cuda.device_count()
        logger.info(f"NGPUS = {ngpus}")

    input_dir = config["input_dir"]
    output_dir = config["output_dir"]

    if vertical == "stratosphere_only":
        if stencil == 1:
            pre = (
                input_dir
                + "stratosphere_1x1_inputfeatures_u_v_theta_w_N2_uw_vw_era5_training_data_hourly_"
            )
        else:
            pre = (
                input_dir
                + f"stratosphere_nonlocal_{stencil}x{stencil}_inputfeatures_u_v_theta_w_N2_uw_vw_era5_training_data_hourly_"
            )
    elif vertical == "global" or vertical == "stratosphere_update":
        if stencil == 1:
            pre = input_dir + "1x1_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_"
        else:
            pre = (
                input_dir
                + f"nonlocal_{stencil}x{stencil}_inputfeatures_u_v_theta_w_uw_vw_era5_training_data_hourly_"
            )

    train_files = []
    train_years = np.array([2010])
    for year in train_years:
        for months in np.arange(1, 2):
            train_files.append(f"{pre}{year}_constant_mu_sigma_scaling{str(months).zfill(2)}.nc")

    test_files = []
    test_years = np.array([2010])
    for year in test_years:
        for months in np.arange(1, 2):
            # for months in np.arange(1,3):
            test_files.append(f"{pre}{year}_constant_mu_sigma_scaling{str(months).zfill(2)}.nc")

    logger.info("Training the %s horizontal and %s vertical model" % (domain, vertical))
    logger.info("learning rate (min - max): %e - %e" % (lr_min, lr_max))
    logger.info("dropout                  : %f" % dropout)
    logger.info("init_epoch               : %d" % init_epoch)
    logger.info("train_years              : %s" % (",".join(str(year) for year in train_years)))
    logger.info("test_years               : %s" % (",".join(str(year) for year in test_years)))

    logger.info("Defined input files")
    logger.info(f"train batch size = {bs_train}")
    logger.info(f"validation batch size = {bs_test}")

    trainset = Dataset_ANN_CNN(
        files=train_files,
        domain="global",
        vertical=vertical,
        features=features,
        stencil=stencil,
        manual_shuffle=False,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=bs_train, drop_last=False, shuffle=False, num_workers=0
    )  # , persistent_workers=True)
    testset = Dataset_ANN_CNN(
        files=test_files,
        domain="global",
        vertical=vertical,
        features=features,
        stencil=stencil,
        manual_shuffle=False,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=bs_test, drop_last=False, shuffle=False, num_workers=0
    )

    idim = trainset.idim
    odim = trainset.odim
    hdim = 4 * idim
    logger.info(f"Input dim: {idim}, hidden dim: {hdim}, output dim: {odim}")

    model = ANN_CNN(idim=idim, odim=odim, hdim=hdim, dropout=dropout, stencil=trainset.stencil)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=lr_min,
        max_lr=lr_max,
        step_size_up=50,
        step_size_down=50,
        cycle_momentum=False,
    )
    loss_fn = nn.MSELoss()
    logger.info(
        f"model created. \n --- model size: {model.totalsize():.2f} MBs,\n --- Num params: {model.totalparams()/10**6:.3f} mil. "
    )

    # fac not used for vertical scaling yet, but good to have it
    fac = torch.ones(122)  # torch.from_numpy(rho[15:]**0.1)
    fac = (1.0 / fac).to(torch.float32)
    fac = fac.to(device)
    # logger.info('fac_created')

    file_prefix = (
        output_dir + f"{vertical}/ann_cnn_{stencil}x{stencil}_{domain}_{vertical}_era5_{features}_"
    )
    # logger.info(f'file prefix: {file_prefix}')
    if config["options"]["restart"]:
        # load checkpoint before resuming training
        PATH = f"{file_prefix}_train_epoch{init_epoch-1}.pt"
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info("Starting training ...")

    log_filename = f"./ann_cnns_{stencil}x{stencil}_{domain}_{vertical}_{features}_epoch_{init_epoch}_to_{init_epoch+nepochs-1}.txt"
    # Training loop
    model, loss_train, loss_test = Training_ANN_CNN(
        nepochs=nepochs,
        init_epoch=init_epoch,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        trainloader=trainloader,
        testloader=testloader,
        stencil=trainset.stencil,
        bs_train=bs_train,
        bs_test=bs_test,
        save=True,
        file_prefix=file_prefix,
        scheduler=scheduler,
        device=device,
        log_filename=log_filename,
    )


if __name__ == "__main__":
    main()
