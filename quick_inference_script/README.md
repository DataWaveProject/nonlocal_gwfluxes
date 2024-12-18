# Torchscript Tracing


## Download model weights and sample inputs
Firstly we need some test data and a model to validate. The pre-trained models are stored on
[huggingface](https://huggingface.co/amangupta2/nonlocal_gwfluxes/tree/main) and the model inputs are stored on `derecho`. Both
of these can be obtained on the command line using `wget` which is provided by the handy script `get-model-and-data.sh`.

## Produce reference data for comparison
To validate the model has been successfully traced it is essential we have some test data and expect model outputs. To get this
I have modified `function_training.py` to add the following lines:

```python
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
```

This writes the input, output and predicted (inferred) tensors to netCDF files which can be read easily in the test programs:

- `infer.py`
- `infer.f90`

To generate the reference data run the following command:

```
python two_sample_inference.py
```

This will generate the following files which I refer to as `reference data`:

- `input.nc`
- `output.nc`
- `predict.nc`

## Python and Fortran test programs

Once we have reference data we need to "trace" the PyTorch model using `torchscript`. This is achieved using `pt2ts.py`.

```
python pt2ts.py
```

This will generate a new `.pt` file in this case called `saved_nlgw_model_gpu.pt` which is a binary representation of the NN.
This is the key ingredient that we require for the two test programs, `infer.py` and `infer.f90`.

The test programs use the reference data which has been previously generated to run inference (using `input.nc`) and validate
the predicted output (using `predict.nc`).

To run the python version use:

```
python infer.py
```

For the fortran version, you need to compile with netCDF and FTorch. An example compile and run script is provided `run.sh`.

```
./run.sh
```
