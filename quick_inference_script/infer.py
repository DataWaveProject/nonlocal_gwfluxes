import numpy as np
import torch
import xarray as xr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = ", device)

input_data = xr.load_dataset("input.nc")["__xarray_dataarray_variable__"].to_numpy()
output_data = xr.load_dataset("output.nc")["__xarray_dataarray_variable__"].to_numpy()
reference_prediction = xr.load_dataset("predict.nc")["__xarray_dataarray_variable__"].to_numpy()
print("input_data.shape = ", input_data.shape)
print("reference_prediction.shape = ", reference_prediction.shape)

model = torch.jit.load("saved_nlgw_model_gpu.pt")

# this doesn't work... how can we incorporate this? It looks like we shouldn't use this for inference anyway
# model.dropout.train()
# print("model = ", model.code)

# run model inference
test_prediction = model(torch.tensor(input_data).to(device))

test_prediction = test_prediction.cpu().detach().numpy()
print("test_prediction.shape = ", test_prediction.shape)

print("test_prediction = ", test_prediction)
print("reference_prediction = ", reference_prediction)

# check by manually editing data
# test_prediction[0, 3] = 0.0

if np.allclose(test_prediction, reference_prediction):
    print("passed")
else:
    raise ValueError("ref data doesn't match predicted data")
