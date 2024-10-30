# Nonlocal Gravity Wave Flux Prediction

The code in this repository implements three classes of deep learning models to predict atmospheric gravity wave momentum fluxes, provided the atmospheric background conditions as input.

## Models description
The three kinds of models are:
1. **M1:** A single column ANN which takes single column atmospheric variables as input and predicts the gravity wave momentum fluxes in the single column (pixel-to-pixel regression)
2. **M2:** A regionally nonlocal CNN+ANN which takes a (2N+1)x(2N+1) horizontal columns of atmospheric variables as input and predicts the fluxes in the central single column (image-to-pixel regression)
3. **M3:** A global attention UNet model which taked global horizontal maps of variables as input to predict fluxes over the whole horizontal domain (image-to-image regression)
The three models are schematically depicted in the animation below



![Schematic for the three models](https://amangupta2.github.io/images/icml_schematic.gif)



The models are trained on modern reanalysis ERA5 which resolves part of the atmospheric gravity wave spectrum. Since it does not resolve the mesoscale wave spectrum, the repository provides some functionality to retrain parts of the models M1-M3 trained on ERA5 to be retrained on fluxes obtained from a 1 km global IFS model which resolves the whole mesoscale wave spectrum.


## Usage

### Training
The code to train M1 and M2 is contained in the ann_cnn_training directory. The code is split into multiple files but the main is invoked in training.py. The model training can be submitted as a single GPU task using the batch.sh script using the command:
```bash
    python training.py <horizontal_domain> <vertical_domain> <features> <stencil>
```
*horizontal domain:* 'regional' or 'global'

*vertical domain:* 'global' or 'stratosphere_only'

*features:* 'uvtheta', 'uvthetaw', or 'uvw' for 'global' vertical domain, and 'uvtheta', 'uvthetaw', 'uvw', 'uvthetaN2', or 'uvthetawN2' for 'stratosphere_only' vertical domain

*stencil:* 1 for single column (M1), 3 for 3x3 regional nonlocality (M2), 5 for 5x5 regional nonlocality and so on 

Likewise, the code to train M3 is contained in the attention_unet directory. The main function is invoked in training_attention_unet.py, and the training can be submitted using the batch.sh script in the attention_unet directory using the command:
```bash
    python training_attention_unet.py <vertical_domain> <features>
```
Here, the horizontal domain is assumed to be global and the stencil argument is not relevant. Same set of possible values for vertical domain as the ANNs



### Inference
By default, the models are trained on three years of ERA5 data, and a fourth year is used for validation. Inference scripts, inference.py, are provided in the respective directories, and can be used as:

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

