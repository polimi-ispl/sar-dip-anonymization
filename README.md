# Deep Image Prior Amplitude SAR Image Anonymization
<div align="center">
  
<!-- **Authors:** -->

**_¹ [Edoardo Daniele Cannas](linkedin.com/in/edoardo-daniele-cannas-9a7355146/), ¹ [Sara Mandelli](https://www.linkedin.com/in/saramandelli/), ¹ [Paolo Bestagini](https://www.linkedin.com/in/paolo-bestagini-390b461b4/)_**

**_¹ [Stefano Tubaro](https://www.linkedin.com/in/stefano-tubaro-73aa9916/), ² [Edward J. Delp](https://www.linkedin.com/in/ejdelp/)_**


<!-- **Affiliations:** -->

¹ [Image and Sound Processing Laboratory](http://ispl.deib.polimi.it/), ² [Video and Image Processing Laboratory](https://engineering.purdue.edu/~ips/index.html)
</div>

This is the official code repository for the paper *Deep Image Prior Amplitude SAR Image Anonymization*, under submission
to MDPI Remote Sensing.  
The repository is currently **under development**, so feel free to open an issue if you encounter any problem.
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
The notebook will inpaint a Sentinel-1 GRD SAR image (VV polarization, provided in `dip_sar_inpainting/data/`) and create the GIF you see above.

## Anonymizing SEN12MS samples
The scripts [SEN12MS_inpainting_tensorboard.py](dip_sar_inpainting/SEN12MS_inpainting_tensorboard.py) and [SEN12MS_inpainting_wandb.py](dip_sar_inpainting/SEN12MS_inpainting_wandb.py)
allow to perform the anonymization/inpainting procedure on Sen12MS samples. The two scripts differ only for the backend used for logging (`tensorboard` or `wandb`, the last providing some nice [features](https://docs.wandb.ai/guides) like hyperparameter search [see below]).  
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

## Generating the anonymized SAR images dataset
To generate the dataset analyzed in the paper, you can simply run the `./dip_sar_inpainting/create_anonymized_dataset.sh` bash script.  
The script will perform a separate inpainting process for each land-cover class, and save them in the `OUTPUT_DIR` folder.  
You can then use them with the [quality analysis notebook](quality_analysis/DIP_anonymized_images_quality_analysis_COMPLETE.ipynb) (see below) to perform the experiments of the paper.

## Land-cover classification
In the `./landcover_classification` folder we reported a modified training script to train the models used for the semantic evaluation of the DIP anonymized samples.  
It is a modified version of the original [SEN12MS Toolbox](https://github.com/schmitt-muc/SEN12MS) script to follow the train/val/test splits used in the paper.

## DIP anonymized images quality analysis
In the `./quality_analysis` folder this [notebook](quality_analysis/DIP_anonymized_images_quality_analysis_COMPLETE.ipynb) will compute the results and figures shown in Section 4 of the paper.  
We also provide for convenience the features extracted by the 3 CNNs used for the semantic metrics.

