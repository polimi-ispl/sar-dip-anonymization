"""
Data loading utilities.
Some of them have been extended from the Sen12MS toolbox (https://github.com/schmitt-muc/SEN12MS/blob/).

Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
"""
# Libraries import #
import torch
import glob
import os
from typing import Callable, Any, List, Dict, Tuple
import cv2
import numpy as np
import rasterio
import pandas as pd
from skimage.draw import random_shapes
from torchvision.transforms import ToTensor

# Helpers variables

# Statistics for SEN12MS "multi_label" datasets
BANDS = {'s1': [1], 's2': [2, 3, 4]}  # s1: 1 = VV, 2 = VH; s2: 2 = R; 3 = G; 4 = B;

BANDS_MEAN = {'s1_mean': [-11.76858, -18.294598],
              's2_mean': [1226.4215, 1137.3799, 1139.6792, 1350.9973, 1932.9058,
                          2211.1584, 2154.9846, 2409.1128, 2001.8622, 1356.0801]}

BANDS_MAD = {'s1_mad': [0.04405262693762779, 0.010192389832809567]}

BANDS_MED = {'s1_med': [0.0632559210062027, 0.012550059705972672]}

BANDS_STD = {'s1_std': [4.525339, 4.3586307],
             's2_std': [741.6254, 740.883, 960.1045, 946.76056, 985.52747,
                        1082.4341, 1057.7628, 1136.1942, 1132.7898, 991.48016]}


# Helpers functions

def load_raster(path: str, sensor: str, tile_transform: bool=False) -> np.array:
    """
    Simple function to load and normalize SEN12MS raster data with rasterio
    :param path: str, path to the .tif file
    :param sensor: str, either s1 or s2 sentinel
    :param tile_transform: bool, wheter there is another transformation for the raster data
    :return np.array, the array of raster data
    """
    with rasterio.open(path, 'r') as src:
        rdata = src.read()

    if sensor == 's1':
        rdata = rdata.astype(np.float32)
        rdata = np.nan_to_num(rdata)
        rdata = np.clip(rdata, -25, 0)
        if not tile_transform:
            rdata /= 25
            rdata += 1
        rdata = rdata.astype(np.float32)
    elif sensor == 's2':
        rdata = rdata.astype(np.float32)
        if not tile_transform:
            rdata = np.clip(rdata, 0, 10000)
            rdata /= 10000
        rdata = rdata.astype(np.float32)

    return rdata


def load_SEN12MS_s1_raster(path: str, tile_transform: bool=False) -> torch.tensor:
    """
    Load S1 raster data from SEN12MS dataset.
    :param path: str, path to the tile to load
    :param tile_transform: bool, whether the image has to be normalized
    :return: np.array, the loaded and normalized tile
    """

    # Load raster
    with rasterio.open(path, 'r') as src:
        rdata = src.read()

    # Normalize
    rdata = rdata.astype(np.float32)
    rdata = np.nan_to_num(rdata)
    if not tile_transform:
        rdata = np.clip(rdata, -25, 0)
        rdata /= 25
        rdata += 1
    rdata = rdata.astype(np.float32)

    return  torch.from_numpy(rdata)


def mz_score_norm(img: torch.Tensor, img_mad: float, img_med: float) -> torch.Tensor:
    """
    Perform the modified Z-score normalization
    suggested in “DeepInSAR—A Deep Learning Framework for SAR Interferometric Phase Restoration and Coherence Estimation”
    then reports the converted img in a 0-1 range.
    Code by Gianluca Murdaca (gianluca.murdaca@polimi.it)

    :param img: torch.Tensor, one of the S1 polarization bands used as input
    :param img_mad: float, the Mean Absolute Deviation from the median of the considered dataset
    :param img_med: float, the median computed on a significant # of samples of the considered dataset
    :return:
    """
    shape = img.shape
    img = img.flatten()
    mz = 0.6745 * ((img - img_med / img_mad))  # 0.6475 is the 75th quartile of the normal distribution, removes the outlier
    mz = (torch.nn.functional.tanh(mz / 7) + 1) / 2  # apply tanh to remove outliers
    mz = (mz - mz.min()) / (mz.max() - mz.min())  # normalize between 0 and 1
    img = mz.reshape(shape)
    return img


# Classes

class Normalize(object):
    """
    Normalization class derived from the SEN12MS toolbox.
    Original code available here https://github.com/schmitt-muc/SEN12MS/blob/master/classification/dataset.py
    """
    def __init__(self, bands_mean: Dict[str, List]=BANDS_MEAN, bands_std: Dict[str, List]=BANDS_STD):
        """
        Class constructor
        :param bands_mean: Dict, mean for bands of both S1 and S2 tiles
        :param bands_std: Dict, STDs for bands of both S1 and S2 tiles
        """
        self.bands_s1_mean = bands_mean['s1_mean']
        self.bands_s1_std = bands_std['s1_std']

        self.bands_s2_mean = bands_mean['s2_mean']
        self.bands_s2_std = bands_std['s2_std']

        self.bands_RGB_mean = bands_mean['s2_mean'][0:3]
        self.bands_RGB_std = bands_std['s2_std'][0:3]

        self.bands_all_mean = self.bands_s2_mean + self.bands_s1_mean
        self.bands_all_std = self.bands_s2_std + self.bands_s1_std

    def __call__(self, tile: torch.Tensor) -> torch.Tensor:
        """
        Call function. Sensor type is inferred by the number of bands
        :param tile: torch.Tensor, the tile to normalize
        :return: torch.Tensor, the normalized tile
        """

        # different input channels
        if tile.size()[0] == 12:
            for t, m, s in zip(tile, self.bands_all_mean, self.bands_all_std):
                t.sub_(m).div_(s)
        elif tile.size()[0] == 10:
            for t, m, s in zip(tile, self.bands_s2_mean, self.bands_s2_std):
                t.sub_(m).div_(s)
        elif tile.size()[0] == 5:
            for t, m, s in zip(tile,
                               self.bands_RGB_mean + self.bands_s1_mean,
                               self.bands_RGB_std + self.bands_s1_std):
                t.sub_(m).div_(s)
        elif tile.size()[0] == 3:
            for t, m, s in zip(tile, self.bands_RGB_mean, self.bands_RGB_std):
                t.sub_(m).div_(s)
        else:
            for t, m, s in zip(tile, self.bands_s1_mean, self.bands_s1_std):
                t.sub_(m).div_(s)

        return tile


class S1Normalize(object):
    """
    Normalization class derived from the SEN12MS toolbox specifically designed for Sentinel-1 data.
    Original code available here https://github.com/schmitt-muc/SEN12MS/blob/master/classification/dataset.py
    """
    def __init__(self, bands_mean: Dict[str, List]=BANDS_MEAN, bands_std: Dict[str, List]=BANDS_STD,
                 bands_mad: Dict[str, List]=BANDS_MAD, bands_med: Dict[str, List]=BANDS_MED, linear: bool=False,
                 mz_norm: bool=False, mean_std: bool=True):
        """
        Class constructor
        :param bands_mean: Dict, mean for bands of both S1 and S2 tiles
        :param bands_std: Dict, STDs for bands of both S1 and S2 tiles
        :param bands_mad: Dict, MAD for S1 bands
        :param bands_med: Dict, median for S1 bands
        :param mz_norm: bool, whether to perform the MZ-score normalization or not
        """
        self.bands_s1_mean = bands_mean['s1_mean']
        self.bands_s1_std = bands_std['s1_std']
        self.bands_s1_mad = bands_mad['s1_mad']
        self.bands_s1_med = bands_med['s1_med']
        self.linear = linear
        self.mean_std = mean_std
        self.mz_norm = mz_norm

    def __call__(self, tile: torch.Tensor) -> torch.Tensor:
        """
        Call function. Sensor type is inferred by the number of bands
        :param tile: torch.Tensor, the tile to normalize
        :return: torch.Tensor, the normalized tile
        """

        if self.mz_norm:
            # Convert from dB to linear and then perform mZ score normalization
            tile = torch.cat([mz_score_norm(torch.pow(10, torch.div(tile[band], 10)), self.bands_s1_mad[band],
                                            self.bands_s1_med[band]).unsqueeze(0)
                              for band in range(tile.shape[0])])
        # normalize each polarization channel by /std and - mean
        else:
            if self.mean_std:
                for t, m, s in zip(tile, self.bands_s1_mean, self.bands_s1_std):
                        t.sub_(m).div_(s)
            if self.linear:
                tile = torch.pow(10, torch.div(tile, 10))

        return tile


class InpaintingDatasetFolder(torch.utils.data.Dataset):
    """
    Simple dataset to load the images and relative object removal masks inside a folder
    """
    def __init__(self, root: str, img_extension: str = '.tiff', mask_extension: str = 'png', loader: Callable[[str], Any] = None):
        self.root = root
        self.mask_ext = mask_extension
        self.img_ext = img_extension
        self.samples = sorted(glob.glob(os.path.join(self.root, '*.{}'.format(self.img_ext))))
        self.masks = sorted(glob.glob(os.path.join(self.root, '*.{}'.format(self.mask_ext))))
        assert len(self.samples) == len(self.masks), 'Masks and samples are not in equal number in {}!'.format(self.root)
        self.loader = loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        img = self.loader(self.samples[item])
        img_name = self.samples[item].split('/')[-1].split('.')[0]
        mask = cv2.imread(self.masks[item], cv2.IMREAD_UNCHANGED)
        if mask.max() > 1:
            mask = mask.astype(np.float32)
            mask -= mask.min()
            mask /= mask.max()
        img.shape = (1, ) + img.shape
        mask.shape = (1, ) + mask.shape
        return img_name, img, mask


class SEN12MSS1InpaintingDatasetFolder(torch.utils.data.Dataset):
    """
    Simple dataset to load SEN12MS S1 tiles and create a random object removal masks
    """
    def __init__(self, data_root: str, df: pd.DataFrame, loader: Callable[[str], Any] = load_SEN12MS_s1_raster,
                 transforms: Callable[[str], Any] = None, inp_size: Tuple[int, int]=(64, 64),
                 pol_bands: str='VV'):
        """
        Object constructor
        :param data_root: str, path to the Sen12MS dataset folder
        :param df: pd.DataFrame, DataFrame containing the tiles to be loaded
        :param loader: Callable, function for loading the data. Default load_raster_data
        :param transforms: torchvision.Transforms, transform to apply on the the data
        :param inp_size: List[int, int], area of inpainting
        :param pol_bands: str, polarization bands to choose. If not VV or VH, both are taken
        """
        self.data_root = data_root
        self.df = df
        self.loader = loader
        self.transforms = transforms
        self.inp_size = inp_size
        self.bands = pol_bands

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item) -> List[torch.Tensor]:
        """
        Load the tile, create a random inpainting mask.
        :param item: int, idx for loading
        :return: List, tile tensor and mask tensor
        """
        # Load and normalize tile
        item = self.df.iloc[item]
        tile = self.loader(os.path.join(self.data_root, item['path_s1']), self.transforms)
        if self.transforms:
            tile = self.transforms(tile)
        # Take polarization band
        if self.bands == 'VV':
            tile = tile[0]
            tile = torch.unsqueeze(tile, 0)
        elif self.bands == 'VH':
            tile = tile[1]
            tile = torch.unsqueeze(tile, 0)
        # Generate random inpainting mask
        mask, _ = random_shapes((256, 256), max_shapes=1, min_shapes=1, min_size=self.inp_size[0], max_size=self.inp_size[1],
                                multichannel=False, num_channels=1, shape='rectangle', random_seed=42)  # fix random seed for having a specific position in the image
        if mask.max() > 1:
            mask = mask.astype(np.float32)
            mask -= mask.min()
            mask /= mask.max()
        mask = torch.from_numpy(mask)
        mask = torch.unsqueeze(mask, 0)
        return tile, mask
