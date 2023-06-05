# Deep Image Prior Amplitude SAR Image Anonymization
This is the official code repository for the paper *Deep Image Prior Amplitude SAR Image Anonymization*, under submission
to MDPI Remote Sensing.
The repository is currently **under maintenance**, the first release is coming soon.
![](assets/GA.png)
![](assets/dip_gif_animation/DIP_iteration_progress.gif)

# Getting started

## Prerequisites
In order to run our code, you need to:
1. install [conda](https://docs.conda.io/en/latest/miniconda.html)
2. create the `sar-dip-anonymization` environment using the *environment.yml* file
```bash
conda env create -f envinroment.yml
conda activate sar-dip-anonymization
```
3. download and unzip the [Sen12MS dataset](https://mediatum.ub.tum.de/1474000).

## Proof of concept
You can use the notebook [proof-of-concept.ipnyb](dip_sar_inpainting/proof-of-concept.ipynb) to have a quick glance on the functioning of the proposed method.  
The notebook will inpaint a Sentinel-1 GRD SAR image (VV polarization) (provided in `dip_sar_inpainting/data/` and create the GIF you see above.

## Anonymizing Sen12MS samples
The scripts [SEN12MS_inpainting_tensorboard.py](dip_sar_inpainting/SEN12MS_inpainting_tensorboard.py) and [SEN12MS_inpainting_wandb.py](dip_sar_inpainting/SEN12MS_inpainting_wandb.py)
allow to perform the anonymization/inpainting procedure on Sen12MS samples. The two scripts differ only for the backend used for logging (`tensorboard` or `wandb`, the last providing some nice [features](https://docs.wandb.ai/guides] like) hyperparameter search (see below)).  
Both scripts accepts different arguments (you can check all of them with `python SEN12MS_inpainting_tensorbooard.py --help`).  
Basically, the scripts take `--samples_per_class` tiles of the land-cover classes specified with `--inp_classes`, make sure that each sample possesses at least `--perc_area_cov` percentage of its surface covered by a single land-cover class, and inpaint them with the DIP.
The only required parameters are:
- `--output_dir`, path where storing the results of the experiments;
- `--SEN12MS_root`, path to the directory containing the unzipped Sen12MS dataset.  

We also provide in `data`:
1. the `data/tiles_info.csv` Pandas DataFrame, containing information on all the tiles of the dataset;
2. the `data/histogram_norm_DFC_2020_scheme.csv` Pandas DataFrame, containing information on the land-cover content of the considered tiles.  

**If you want to double check how we compute these information**, be sure to run the [Sen12MS_preprocessing.ipynb](dip_sar_inpainting/Sen12MS_preprocessing.ipynb) notebook.

## Hyperparameter search
We used WandB [Sweeps](https://docs.wandb.ai/guides/sweeps) to perform a grid search over the DIP hyperparameters.  
In the `./dip_sar_inpainting/wandb_sweep_search` folder we provide `.yml` files for all the configurations used on the different land-cover classes.  
We launched a separate sweep for each class, in order to maximize parallel computation on GPUs (see [here](https://wandb.ai/site/articles/multi-gpu-sweeps)).

