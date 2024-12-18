"""Load a PyTorch model and convert it to TorchScript."""

import os
import sys
from typing import Optional

import numpy as np

# FPTLIB-TODO
# Add a module import with your model here:
# This example assumes the model architecture is in an adjacent module `my_ml_model.py`
import torch

from dataloader_definition import Dataset_ANN_CNN
from function_training import Inference_and_Save_ANN_CNN
from model_definition import ANN_CNN


def script_to_torchscript(
    model: torch.nn.Module, filename: Optional[str] = "scripted_model.pt"
) -> None:
    """
    Save PyTorch model to TorchScript using scripting.

    Parameters
    ----------
    model : torch.NN.Module
        a PyTorch model
    filename : str
        name of file to save to
    """
    print("Saving model using scripting...", end="")
    # FIXME: torch.jit.optimize_for_inference() when PyTorch issue #81085 is resolved
    scripted_model = torch.jit.script(model)
    # print(scripted_model.code)
    scripted_model.save(filename)
    print("done.")


def trace_to_torchscript(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    filename: Optional[str] = "traced_model.pt",
) -> None:
    """
    Save PyTorch model to TorchScript using tracing.

    Parameters
    ----------
    model : torch.NN.Module
        a PyTorch model
    dummy_input : torch.Tensor
        appropriate size Tensor to act as input to model
    filename : str
        name of file to save to
    """
    print("Saving model using tracing...", end="")
    # FIXME: torch.jit.optimize_for_inference() when PyTorch issue #81085 is resolved
    traced_model = torch.jit.trace(model, dummy_input)
    # traced_model.save(filename)
    frozen_model = torch.jit.freeze(traced_model)
    ## print(frozen_model.graph)
    ## print(frozen_model.code)
    frozen_model.save(filename)
    print("done.")


def load_torchscript(filename: Optional[str] = "saved_model.pt") -> torch.nn.Module:
    """
    Load a TorchScript from file.

    Parameters
    ----------
    filename : str
        name of file containing TorchScript model
    """
    model = torch.jit.load(filename)

    return model


if __name__ == "__main__":
    # =====================================================
    # Load model and prepare for saving
    # =====================================================

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    domain = "global"
    vertical = "stratosphere_only"
    features = "uvtheta"

    model = "ann"  # or 'ann'
    stencil = 1
    epoch = 100
    dropout = 0.1

    if model == "attention":
        stencil = 1

    bs_test = 1

    # FPTLIB-TODO
    # Load a pre-trained PyTorch model
    # Insert code here to load your model as `trained_model`.
    # This example assumes my_ml_model has a method `initialize` to load
    # architecture, weights, and place in inference mode
    # assumes ckpts stored in model_ckpt dir
    PATH = f"model-huggingface/ann_cnn_{stencil}x{stencil}_{domain}_{vertical}_era5_{features}__train_epoch{epoch}.pt"

    test_files = [
        "inputs/stratosphere_1x1_inputfeatures_u_v_theta_w_N2_uw_vw_era5_training_data_hourly_2010_constant_mu_sigma_scaling01.nc"
    ]
    test_years = np.array([2010])
    testset = Dataset_ANN_CNN(
        files=test_files,
        domain=domain,
        vertical=vertical,
        stencil=stencil,
        manual_shuffle=False,
        features=features,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=bs_test, drop_last=False, shuffle=False, num_workers=0
    )

    idim = testset.idim
    odim = testset.odim
    hdim = 4 * idim  # earlier runs has 2*dim for 5x5 stencil. Stick to 4*dim this time
    print("dropout = ", dropout)

    # ---- define model
    model = ANN_CNN(idim=idim, odim=odim, hdim=hdim, dropout=dropout, stencil=stencil)
    checkpoint = torch.load(PATH, map_location=torch.device(device), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    trained_model = model

    # Switch off specific layers/parts of the model that behave
    # differently during training and inference.
    # This may have been done by the user already, so just make sure here.
    trained_model.eval()

    # =====================================================
    # Prepare dummy input and check model runs
    # =====================================================

    # FPTLIB-TODO
    # Generate a dummy input Tensor `dummy_input` to the model of appropriate size.
    # This example assumes one input of size (1x3x244x244) -- one RGB (3) image
    # of resolution 244x244 in a batch size of 1.
    trained_model_dummy_input = torch.ones(1, 8192, 183)

    # FPTLIB-TODO
    # Uncomment the following lines to save for inference on GPU (rather than CPU):
    device = torch.device("cuda")
    trained_model = trained_model.to(device)
    trained_model.eval()
    trained_model_dummy_input = trained_model_dummy_input.to(device)

    # FPTLIB-TODO
    # Run model for dummy inputs
    # If something isn't working This will generate an error
    trained_model_dummy_outputs = trained_model(
        trained_model_dummy_input,
    )

    print("trained_model = ", trained_model)

    # =====================================================
    # Save model
    # =====================================================

    # FPTLIB-TODO
    # Set the name of the file you want to save the torchscript model to:
    saved_ts_filename = "saved_nlgw_model_gpu.pt"
    # A filepath may also be provided. To do this, pass the filepath as an argument to
    # this script when it is run from the command line, i.e. `./pt2ts.py path/to/model`.

    # FPTLIB-TODO
    # Save the PyTorch model using either scripting (recommended if possible) or tracing
    # -----------
    # Scripting
    # -----------
    # script_to_torchscript(trained_model, filename=saved_ts_filename)

    # -----------
    # Tracing
    # -----------
    trace_to_torchscript(trained_model, trained_model_dummy_input, filename=saved_ts_filename)

    print(f"Saved model to TorchScript in '{saved_ts_filename}'.")

    # =====================================================
    # Check model saved OK
    # =====================================================

    # Load torchscript and run model as a test
    # FPTLIB-TODO
    # Scale inputs as above and, if required, move inputs and mode to GPU
    trained_model_testing_outputs = trained_model(
        trained_model_dummy_input,
    )
    ts_model = load_torchscript(filename=saved_ts_filename)
    ts_model_outputs = ts_model(
        trained_model_dummy_input,
    )

    print("ts_model_outputs.shape = ", ts_model_outputs.shape)

    # if not isinstance(ts_model_outputs, tuple):
    #     ts_model_outputs = (ts_model_outputs,)
    # if not isinstance(trained_model_testing_outputs, tuple):
    #     trained_model_testing_outputs = (trained_model_testing_outputs,)
    # for ts_output, output in zip(ts_model_outputs, trained_model_testing_outputs):
    #     if torch.all(ts_output.eq(output)):
    #         print("Saved TorchScript model working as expected in a basic test.")
    #         print("Users should perform further validation as appropriate.")
    #     else:
    #         model_error = (
    #             "Saved Torchscript model is not performing as expected.\n"
    #             "Consider using scripting if you used tracing, or investigate further."
    #         )
    #         raise RuntimeError(model_error)

    # # Check that the model file is created
    # filepath = os.path.dirname(__file__) if len(sys.argv) == 1 else sys.argv[1]
    # if not os.path.exists(os.path.join(filepath, saved_ts_filename)):
    #     torchscript_file_error = (
    #         f"Saved TorchScript file {os.path.join(filepath, saved_ts_filename)} "
    #         "cannot be found."
    #     )
    #     raise FileNotFoundError(torchscript_file_error)
