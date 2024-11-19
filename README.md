# Nonlocal Gravity Wave Flux Prediction

The code in this repository implements three classes of deep learning models to predict atmospheric gravity wave momentum fluxes, provided the atmospheric background conditions as input.

## Models description
The three kinds of models are:
1. **M1:** A single column ANN which takes single column atmospheric variables as input and predicts the gravity wave momentum fluxes in the single column (pixel-to-pixel regression)
2. **M2:** A regionally nonlocal CNN+ANN which takes a (2N+1)x(2N+1) horizontal columns of atmospheric variables as input and predicts the fluxes in the central single column (image-to-pixel regression)
3. **M3:** A global attention UNet model which taked global horizontal maps of variables as input to predict fluxes over the whole horizontal domain (image-to-image regression)
The three models are schematically depicted in the animation below



![Schematic for the three models](https://amangupta2.github.io/images/icml_schematic_static.jpg)



The models are trained on modern reanalysis ERA5 which resolves part of the atmospheric gravity wave spectrum. Since it does not resolve the mesoscale wave spectrum, the repository provides some functionality to retrain parts of the models M1-M3 trained on ERA5 to be retrained on fluxes obtained from a 1 km global IFS model which resolves the whole mesoscale wave spectrum.

## Getting Started

### Obtaining `nonlocal_gwfluxes`

Clone the latest version of `nonlocal_gwfluxes` using the following command

```bash
git clone git@github.com:DataWaveProject/nonlocal_gwfluxes.git
```

### Install Dependencies

This project uses `poetry` to manage dependencies. To install the dependencies we first need poetry.

```bash
cd nonlocal_gwfluxes
python -m venv .nlgw
```

This will create a Python virtual environment inside our repository. Before we use this environment we must first `activate` it.

```bash
source .nlgw/bin/activate
```

> [!NOTE]
> Your shell prompt should update. Depending on your setup, you may have something like this:
> ```
> (.nlgw) demo@mypc:~/nonlocal_gwfluxes$
> ```

Now we can install `poetry`

```bash
pip install poetry
```

The following command installs all of the necessary dependencies for `nonlocal_gwfluxes`.

```bash
poetry install --no-root
```

>[!IMPORTANT]
>If you plan to develop `nonlocal_gwfluxes`, please add the optional `develop` dependencies and complete the further setup actions detailed below.

```bash
poetry install --no-root --with develop
```

Installation of the `pre-commit` hooks is completed with the following instruction.

```bash
pre-commit install
```

The `pre-commit` hooks can be run manually at any time with the following instruction.

```bash
pre-commit run --all-files
```
>[!NOTE]
>This repository enforces a consistent code style with `ruff` through `pre-commit` hooks and `github` actions. The rules `ruff` applies are detailed in the `tool.ruff.lint` section of the `pyroject.toml`.

## Usage

### Dependencies

To use these scripts, users will first need to install the following dependencies:

* `pytorch`
* `netCDF4`
* `numpy`
* `xarray`

Please follow the steps installation steps above.

### Training
The code to train M1 and M2 is contained in the `ann_cnn_training` directory. The code is split into multiple files but the main is invoked in `training.py`. The model training can be submitted as a single GPU task using the `batch.sh` script using the command:
```bash
    python training.py <horizontal_domain> <vertical_domain> <features> <stencil>
```
*horizontal domain:* `regional` or `global`

*vertical domain:* `global`, `stratosphere_only`, or `stratosphere_update`

*features:* `uvtheta`, `uvthetaw`, or `uvw` for `global` and `stratosphere_update` vertical domain, and `uvtheta`, `uvthetaw`, `uvw`, `uvthetaN2`, or `uvthetawN2` for `stratosphere_only` vertical domain (due to data storage considerations)

*stencil:* 1 for single column (M1), 3 for 3x3 regional nonlocality (M2), 5 for 5x5 regional nonlocality and so on 

Likewise, the code to train M3 is contained in the `attention_unet` directory. The main function is invoked in `training_attention_unet.py`, and the training can be submitted using the `batch.sh` script in the attention_unet directory using the command:
```bash
    python training_attention_unet.py <vertical_domain> <features>
```
Here, the horizontal domain is assumed to be global and the stencil argument is not relevant. Same set of possible values for vertical domain as the ANNs



### Inference
By default, the models are trained on three years of ERA5 data, and a fourth year is used for validation. Inference scripts, `inference.py`, are provided in the respective directories, and can be used as:

```bash
    python inference.py <horizontal_domain> <vertical_domain> <features> <epoch_number> <month> <stencil>
```

for the ANNs, and

```bash
    python inference.py <vertical_domain> <features> <epoch> <month>

```

for the attention unet models.



## References
[1] Gupta, Aman*, Aditi Sheshadri, Sujit Roy*, Vishal Gaur, Manil Maskey, Rahul Ramachandran: "Machine Learning Global Simulation of Nonlocal Gravity Wave Propagation", International Conference on Machine Learning 2024, ML4ESM Workshop, https://arxiv.org/abs/2406.14775
