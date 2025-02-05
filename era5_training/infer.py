import argparse
from pathlib import Path

import numpy as np
import torch
import xarray as xr


def main():
    args = cli_options()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device = ", device)

    prefix = ""
    if args.model == "ann":
        prefix = "ann-cnn"
    elif args.model == "attention":
        prefix = "unet"

    input_data = load_nc_dataset(args.test_data_dir / Path(prefix + "-input.nc"))
    pred_reference = load_nc_dataset(args.test_data_dir / Path(prefix + "-predict.nc"))

    model_path = args.scripted_model_dir / Path(f"nlgw_{prefix}_gpu_scripted.pt")
    print(f"loading model {model_path}...")
    model = torch.jit.load(model_path)

    # run model inference
    pred = model(torch.tensor(input_data).to(device))

    pred = pred.cpu().detach().numpy()
    print("pred.shape = ", pred.shape)

    print("max diff = ", np.max(np.abs(pred - pred_reference)))

    if np.allclose(pred, pred_reference):
        print("passed")
    else:
        raise ValueError("ref data doesn't match predicted data")


def cli_options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-M",
        "--model",
        choices=["ann", "attention"],
        default="ann",
        help="Model to be trained",
    )
    parser.add_argument(
        "-t",
        "--test-data-dir",
        default=Path.cwd(),
        help="Directory containing test data",
        type=Path,
    )
    parser.add_argument(
        "-s", "--scripted-model-dir", default=Path.cwd(), help="Scripted model directory", type=Path
    )
    return parser.parse_args()


def load_nc_dataset(path):
    data = xr.load_dataset(path)["__xarray_dataarray_variable__"].to_numpy()
    print(f"loaded dataset {path} :: shape {data.shape}")
    return data


if __name__ == "__main__":
    main()
