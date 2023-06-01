"""
Some common utils functions and other related to the management of tensors and plotting
Authors:
Francesco Picetti - francesco.picetti@polimi.it
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
Sara Mandelli - sara.mandelli@polimi.it
Paolo Bestagini - paolo.bestagini@polimi.it
"""

# --- Libraries import
import numpy as np
import torch
import rasterio
import os
from socket import gethostname
from PIL import Image
from .pytorch_ssim import SSIM


# --- Functions
def tiff_to_float32(path: str, scale_16bit: int = 100) -> np.array:
    """
    Load and normalize a TIFF image either in 8 or 16-bits depth to float32 in range 0-1
    :param path: path for loading the TIFF image
    :param scale_16bit: int, scale for moving the dynamic of the 16bit image
    :return img_float: 0-1 normalized float32 version of the SAR image
    """
    # Open the TIFF image
    print('Opening the TIFF image...')
    with rasterio.open(path) as src:
        profile = src.profile
        img = np.squeeze(src.read())

    # Converting based on the bits depth
    if profile['dtype'] == 'uint16':
        # Convert to float, change the dynamic and clip it
        img_float = np.asarray(img, dtype=np.float32) / (
                2 ** 16 - 1)  # normalize the range [0, 2**16-1] to [0, 1]
        img_float *= scale_16bit  # scale values as the maximum is still too small
        img_float = np.clip(img_float, a_min=0, a_max=1)  # clip values in range [0, 1]
    elif profile['dtype'] == 'uint8':
        img_float = np.asarray(img).astype(np.float32)
        img_float -= img_float.min()
        img_float /= img_float.max()
    del img  # delete image to save space

    return img_float


def to_float32(img: np.array, scale_16bit: int = 100) -> np.array:
    """
    Simply normalize a image either in 8 or 16-bits depth to float32 in range 0-1
    :param img: numpy array of the already loaded TIFF image
    :param scale_16bit: int, scale for moving the dynamic of the 16bit image
    :return img_float: 0-1 normalized float32 version of the SAR image
    """
    if img.dtype == np.uint8:
        img_float = np.asarray(img).astype(np.float32)
        img_float -= img_float.min()
        img_float /= img_float.max()
    elif img.dtype == np.uint16:
        img_float = np.asarray(img, dtype=np.float32) / (2 ** 16 - 1)  # normalize the range [0, 2**16-1] to [0, 1]
        img_float *= scale_16bit  # scale values as the maximum is still too small
        img_float = np.clip(img_float, a_min=0, a_max=1)  # clip values in range [0, 1]

    return img_float


def to16(img: np.array, scale: int=100) -> np.array:
    """
    Reconvert a SAR float image back to 16 bit depth
    :param img: the float32 0-1 normalized SAR image
    :param scale: the scale used to move the dynamic of the original 16-bit SAR image
    :return: img_16, the image in 16 bit depth
    """
    img /= scale
    img *= (2 ** 16 - 1)
    img_16 = img.astype(np.uint16)

    return img_16


def to8(img: np.array) -> np.array:
    """
    Convert a 0-1 normalized float32 image to a uint8
    :param img: 0-1 normalized float32 input
    :return: uint8 reconverted image
    """
    return (img*255).astype(np.uint8)


def make_dir_tag(tag_params: dict, debug: bool, suffix: str = None) -> str:
    tag = 'debug_' if debug else ''
    tag += '_'.join(['-'.join([key, str(tag_params[key])]) for key in tag_params])
    if suffix:
        tag += '_{}'.format(suffix)
    return tag


def log10plot(in_content):
    return np.log10(np.asarray(in_content) / in_content[0])


def ten_digit(number):
    return int(np.floor(np.log10(number)) + 1)


def int2str(in_content, digit_number):
    in_content = int(in_content)
    return str(in_content).zfill(ten_digit(digit_number))


def machine_name():
    return gethostname()


def idle_cpu_count(mincpu=1):
    # the load is computed over the last 1 minute
    idle = int(os.cpu_count() - np.floor(os.getloadavg()[0]))
    return max(mincpu, idle)


def plot2pgf(temp, filename, folder='./'):
    """
    :param temp:        list of equally-long data
    :param filename:    filename without extension nor path
    :param folder:      folder where to save
    """
    if len(temp) == 1:  # if used as plt.plot(y) without the abscissa axis
        temp = [list(range(len(temp[0]))), temp[0]]

    if not os.path.exists(folder):
        os.makedirs(folder)
    np.savetxt(os.path.join(folder, filename + '.txt'), np.asarray(temp).T,
               fmt="%f", encoding='ascii')


def clim(in_content, ratio, zero_mean=True):
    """
    Compute the lower-bound and upper-bound `clim` tuple as a percentage
    of the content dynamic range.

    :param in_content:  np.ndarray
    :param ratio:       float, percentage for the dynamic range (default 1.)
    :param zero_mean:   bool, use symmetric bounds (default True)
    :return: clim tuple (as required by matplotlib.pyplot.imshow)
    """
    if zero_mean:
        max_abs_value = np.max(np.abs(in_content))
        return -ratio * max_abs_value, ratio * max_abs_value
    else:
        return ratio * in_content.min(), ratio * in_content.max()


def save_image(in_content, filename, clim=(None, None), folder='./'):
    """
    Save a gray-scale PNG image of the 2D content

    :param in_content:  2D np.ndarray
    :param filename:    name of the output file (without extension)
    :param clim:        tuple for color clipping (as done in matplotlib.pyplot.imshow)
    :param folder:      output directory
    :return:
    """
    if clim[0] and clim[1] is not None:
        in_content = np.clip(in_content, clim[0], clim[1])
        in_content = normalize(in_content, in_min=clim[0], in_max=clim[1])[0]
    else:
        in_content = normalize(in_content)[0]
    out = Image.fromarray(((in_content + 1) / 2 * 255).astype(np.uint8))

    if not os.path.exists(folder):
        os.makedirs(folder)
    out.save(os.path.join(folder, filename + '.png'))


def sec2time(seconds):
    s = seconds % 60
    m = (seconds // 60) % 60
    h = seconds // 3600
    timestamp = '%dh:%dm:%ds' % (h, m, s)
    return timestamp


def clip_normalize_power(x, mymin, mymax, p):
    """
    Preprocessing function to be applied to migrated images in the C2F scenario

    :param x:       data to be processed
    :param mymin:   min value for clipping
    :param mymax:   max value for clipping
    :param p:       exponent for the power function
    :return:
    """
    x = np.clip(x, a_min=mymin, a_max=mymax)
    x, _, _ = normalize(x)
    x = np.sign(x) * np.power(np.abs(x), p)
    return x


def clip_normalize_power_inverse(x, mymin, mymax, p):
    """
    Inverse preprocessing function to be applied to output images in the C2F scenario
    :param x: data to be processed
    :param mymin: min value used for clipping
    :param mymax: max value used for clipping
    :param p: exponent for the power function (to be inverted)
    :return:
    """
    x = np.sign(x) * np.power(np.abs(x), 1 / p)
    x = denormalize(x, mymin, mymax)
    return x


def normalize(x, in_min=None, in_max=None, zero_mean=True):
    if in_min is None and in_max is None:
        in_min = np.min(x)
        in_max = np.max(x)
    x = (x - in_min) / (in_max - in_min)
    if zero_mean:
        x = x * 2 - 1
    return x, in_min, in_max


def denormalize(x, in_min, in_max):
    """
    Denormalize data.
    :param x: ndarray, normalized data
    :param in_min: float, the minimum value
    :param in_max: float, the maximum value
    :return: denormalized data in [in_min, in_max]
    """
    if x.min() == 0.:
        return x * (in_max - in_min) + in_min
    else:
        return (x + 1) * (in_max - in_min) / 2 + in_min


def nextpow2(x):
    return int(2 ** np.ceil(np.log2(x)))


def torch2numpy(in_content):
    assert isinstance(in_content, torch.Tensor), "ERROR! in_content has to be a torch.Tensor object"
    return in_content.cpu().detach().numpy()


def numpy2torch(in_content, dtype=torch.cuda.FloatTensor):
    assert isinstance(in_content, np.ndarray), "ERROR! in_content has to be a numpy.ndarray object"
    return torch.from_numpy(in_content).type(dtype)


def float2png(in_content):
    return np.clip((255 * in_content), 0, 255).astype(np.uint8)


def png2float(in_content):
    return in_content.astype(np.float32) / 255.


# Functions to convert SAR data from decibel units to linear units and back again
def decibel_to_linear(band: np.array) -> np.array:
    """
    Convert to linear units

    :param band: np.array, the dB SAR tile to convert
    """
    return np.power(10, np.array(band) / 10)


def linear_to_decibel(band):
    """
    Convert to dB

    :param band: np.array, the SAR tile to convert into dB
    """
    return 10 * np.log10(band)


def select_loss(loss_name: str, dtype: torch.dtype) -> torch.nn.Module:
    if loss_name == 'mse':
        loss_f = torch.nn.MSELoss().type(dtype)
    elif loss_name == 'mae':
        loss_f = torch.nn.L1Loss().type(dtype)
    elif loss_name == 'ssim':
        loss_f = SSIM()

    return loss_f