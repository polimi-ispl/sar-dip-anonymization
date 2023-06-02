"""
SEN12MS inpaiting
Main script for the inpainting of SAR tiles using the Deep Image Prior.
Code takes something from the original DIP repo ((https://dmitryulyanov.github.io/deep_image_prior))
and DIPPAS (https://github.com/polimi-ispl/dip_prnu_anonymizer/blob/master/main_1_dip.py).
The code is designed to work on the SEN12MS dataset, and allows to perform inpainting based on the
land-cover content of the tile.
The content is described using a simplified label scheme derived from the MODIS IGBP. This simplified scheme is the same
adopted in the IEEE GRSS DFC 2020 (https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest).

"""

# Libraries import
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import time
import os
import glob
import wandb
import argparse
from models import create_network
import tqdm
import pandas as pd
from utils.data import SEN12MSS1InpaintingDatasetFolder, load_SEN12MS_s1_raster, S1Normalize
from torchvision.transforms import ToTensor, Compose
from utils.isplutils import tiff_to_float32, to8, make_dir_tag, select_loss
from utils.pytorch_ssim import SSIM
from utils.common_utils import get_noise, get_params, torch_to_np

# Helpers functions and classes

class OptimizationRoutine:
    """
    Class for the optimization routine (needed to be called in the optimize function)
    """
    def __init__(self, net: nn.Module, net_input_saved: torch.Tensor, reg_noise_std: float,
                 noise: torch.Tensor, mask_var: torch.Tensor, img_var: torch.Tensor, loss_1: nn.Module,
                 loss_2: nn.Module, loss_balance: float, param_noise: bool, log_int: int = 0, num_iter: int = 100):
        self.net = net
        self.net_input_saved = net_input_saved
        self.reg_noise_std = reg_noise_std
        self.noise = noise
        self.mask_var = mask_var
        self.img_var = img_var
        self.loss_1 = loss_1
        self.loss_2 = loss_2
        self.loss_balance = loss_balance
        self.param_noise = param_noise
        self.log_int = log_int
        self.num_iter = num_iter

    def closure(self, it: int):

        if self.param_noise:
            for n in [x for x in self.net.parameters() if len(x.size()) == 4]:
                n = n + n.detach().clone().normal_() * n.std() / 50

        net_input = self.net_input_saved
        if self.reg_noise_std > 0:
            net_input = self.net_input_saved + (self.noise.normal_() * self.reg_noise_std)

        out = self.net(net_input)

        # Watch out for the signs of the losses!
        # During the optimization routine, we want to minimize -SSIM, and therefore we positively sum the MSE!
        # For the sweep, we want to maximize the final loss (SSIM), therefore we negatively sum the MSE!
        if self.loss_1.__class__ == SSIM:
            # total loss = -SSIM + MSE
            total_loss = -self.loss_1(out * self.mask_var, self.img_var * self.mask_var)*self.loss_balance \
                         + (1-self.loss_balance)*self.loss_2(out * self.mask_var, self.img_var * self.mask_var)
        elif self.loss_2.__class__ == SSIM:
            # total loss = MSE - SSIM
            total_loss = self.loss_1(out * self.mask_var, self.img_var * self.mask_var) * self.loss_balance \
                         - (1 - self.loss_balance) * self.loss_2(out * self.mask_var, self.img_var * self.mask_var)
        else:
            total_loss = self.loss_1(out * self.mask_var, self.img_var * self.mask_var)*self.loss_balance + \
                         self.loss_2(out * self.mask_var, self.img_var * self.mask_var)*(1 - self.loss_balance)
        total_loss.backward()

        # Logging
        print('Iteration %05d    Loss %f' % (it, total_loss.item()), '\r', end='')
        it_images = []
        if (self.log_int and it % self.log_int == 0) or it == self.num_iter:
            # Plot with wandb
            if self.img_var.shape[1] > 1:  # double polarization plot
                # Convert float32 tensor to uint8 numpy array (avoids ugly normalization by WandB and bad plots)
                orig_plot = torch_to_np(self.img_var)
                orig_plot = np.concatenate([np.expand_dims((pol - pol.min()) / (pol.max() - pol.min()), axis=0)
                                for pol in orig_plot])
                orig_plot = (orig_plot*255).astype(np.uint8)
                inp_plot = torch_to_np(out)
                inp_plot = np.concatenate([np.expand_dims((pol - pol.min()) / (pol.max() - pol.min()), axis=0)
                                           for pol in inp_plot])
                inp_plot = (inp_plot * 255).astype(np.uint8)
                # Log them
                it_images.append(wandb.Image(orig_plot[0], caption='Original VV polarization'))
                it_images.append(wandb.Image(orig_plot[1], caption='Original VH polarization'))
                it_images.append(wandb.Image(self.mask_var, caption='Mask variable'))
                it_images.append(wandb.Image(inp_plot[0], caption='Reconstructed image final model VV polarization'))
                it_images.append(wandb.Image(inp_plot[1], caption='Reconstructed image final model VH polarization'))
            else:
                # Convert float32 tensor to uint8 numpy array (avoids ugly normalization by WandB and bad plots)
                orig_plot = torch_to_np(self.img_var)
                orig_plot = (orig_plot-orig_plot.min())/(orig_plot.max()-orig_plot.min())  # convert between 0 and 1
                orig_plot = (orig_plot * 255).astype(np.uint8)  # cast a uint8 matrix
                inp_plot = torch_to_np(out)
                inp_plot = (inp_plot-inp_plot.min())/(inp_plot.max()-inp_plot.min())  # convert between 0 and 1
                inp_plot = (inp_plot * 255).astype(np.uint8)  # cast a uint8 matrix
                # Log them
                it_images.append(wandb.Image(orig_plot, caption='Original'))
                it_images.append(wandb.Image(self.mask_var, caption='Mask variable'))
                it_images.append(wandb.Image(inp_plot,
                                             caption='Reconstructed image final model'))

        return total_loss, it_images


def optimize(optimizer_type, parameters, routine: OptimizationRoutine, lr: float, num_iter: int,
             schedule_lr_patience: int = None, schedule_lr_factor: float = None, img_name: str =None):
    """
    Runs optimization loop.

    Args:
        optimizer_type: 'adam' or 'sgd'
        parameters: list of Tensors to optimize over
        routine: OptimizationRoutine instance, with the closure function that returns loss variable
        lr: learning rate
        num_iter: number of iterations
        schedule_lr_patience: LR scheduler patience
        schedule_lr_factor: LR scheduler factor
        img_name: name of the image, used for logging a different section for each sample
    """

    if optimizer_type == 'adam':
        # Instantiate the optimizer and scheduler
        optimizer = torch.optim.Adam(parameters, lr=lr)
        if schedule_lr_patience:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=schedule_lr_factor,
                                                                   patience=schedule_lr_patience, min_lr=1e-8)

        # Optimization loop #
        tag = '{}/'.format(img_name) if img_name is not None else ''  # set for logging
        for j in range(num_iter):
            optimizer.zero_grad(set_to_none=True)
            loss, it_images = routine.closure(j)
            optimizer.step()
            if schedule_lr_patience:
                scheduler.step(loss)
            # Logging
            if len(it_images):
                wandb.log({'{}Iteration loss'.format(tag): loss.item(),
                           '{}Network output'.format(tag): it_images,
                           '{}Learning rate'.format(tag): optimizer.param_groups[0]['lr'],
                           'It': j})
            else:
                wandb.log({'{}Iteration loss'.format(tag): loss.item(),
                           '{}Learning rate'.format(tag): optimizer.param_groups[0]['lr'],
                           'It': j})

    elif optimizer_type == 'sgd':
        # Instantiate optimization and scheduler
        optimizer = torch.optim.SGD(parameters, lr=lr,
                                    momentum=0, dampening=0, weight_decay=0, nesterov=False)
        if schedule_lr_patience:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=schedule_lr_factor,
                                                                   patience=schedule_lr_patience, min_lr=1e-8)
        # Optimization loop #
        tag = '{}/'.format(img_name) if img_name is not None else ''  # set for logging
        for j in range(num_iter):
            # optimizer.zero_grad(set_to_none=True)
            optimizer.zero_grad()
            loss, it_images = routine.closure(j)
            optimizer.step()
            if schedule_lr_patience:
                scheduler.step(loss)
            # Logging
            if len(it_images):
                wandb.log({'{}Iteration loss'.format(tag): loss.item(),
                           '{}Network output'.format(tag): it_images,
                           '{}Learning rate'.format(tag): optimizer.param_groups[0]['lr'],
                           'It': j})
            else:
                wandb.log({'{}Iteration loss'.format(tag): loss.item(),
                           '{}Learning rate'.format(tag): optimizer.param_groups[0]['lr'],
                           'It': j})
    else:
        assert False


def train(args: argparse.Namespace):

    # Torch configuration #
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    dtype = torch.cuda.FloatTensor

    # set the engine to be
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpu)

    # Load the SEN12MS dataframes #
    tiles_df = pd.read_csv(args.SEN12MS_df, index_col=0)
    classes_df = pd.read_csv(args.classes_df, index_col=0)

    # Samples filtering #
    # Parse arguments
    inp_class = args.inp_classes
    perc_area_cov = args.perc_area_cov
    n_samples_per_class = args.samples_per_class
    pol_bands = args.pol_bands
    # Filter classes
    classes_df = classes_df[['path', 'seed', 'season', 'region', 'sensor', 'tile',]+[inp_class]]
    # Select n samples per class
    filt_df = classes_df.loc[classes_df[inp_class]>=perc_area_cov].sample(n=n_samples_per_class, random_state=42)
    filt_df = filt_df.drop_duplicates().reset_index(drop=True)  # concatenate everything
    # Leave only s1 data
    data_df = filt_df.merge(tiles_df, 'left', left_on=['seed', 'season', 'region', 'tile'],
                            right_on=['seed', 'season', 'region', 'tile'], suffixes=['_lc', '_s1'])
    data_df = data_df.loc[data_df['sensor_s1']=='s1'].reset_index(drop=True)

    # Instantiate torchvision.Transforms and the DataLoader
    trans = Compose([S1Normalize(mz_norm=args.mz_score_norm)])
    dataset = SEN12MSS1InpaintingDatasetFolder(df=data_df, loader=load_SEN12MS_s1_raster, transforms=trans,
                                               inp_size=(args.inp_size, args.inp_size), pol_bands=pol_bands)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Create the config dictionary
    config_defaults = {
        # Insert all the parameter you want WandB to sweep over
        'net': args.net,
        'input_depth': args.input_depth,
        'inp_classes': 'Urban' if args.inp_classes=='Urban/Built-up' else args.inp_classes,
        'perc_area_cov': perc_area_cov,
        'inp_size': args.inp_size,
        'pad': args.pad,
        'upsample': args.upsample,
        'activation': args.activation,
        'loss_1': args.loss_1,
        'loss_2': args.loss_2,
        'loss_balance': args.loss_balance,
        'pol_bands': pol_bands,
        'mz_norm': args.mz_score_norm,
        'need_sigmoid': args.need_sigmoid
    }
    # config_defaults['experiment'] = make_dir_tag(config_defaults, args.debug, args.suffix)
    wandb.init(project=args.project_name, config=config_defaults, name=make_dir_tag(config_defaults, args.debug, args.suffix),
               settings=wandb.Settings(start_method='thread'), reinit=True)

    # Create the output dir
    save_dir = os.path.join(args.output_dir, make_dir_tag(dict(wandb.config.items()), args.debug, args.suffix))
    os.makedirs(save_dir, exist_ok=True)

    # MAIN LOOP #
    final_loss = []  # final loss array
    inp_df = data_df.copy()
    inp_df['Final_inpainted_sample_path'] = ''
    inp_df['Final_generated_sample_path'] = ''
    inp_df['Inpainting_mask_path'] = ''
    inp_df['Polarization_bands'] = pol_bands
    inp_df = inp_df.drop(labels=['sensor_lc'], axis=1)
    for img_idx, (img, mask) in enumerate(tqdm.tqdm(dataloader, desc='Processed images')):
        # Configure wandb for logging a section for each sample
        # wandb.init(project=args.project_name, config=config_defaults, tags=[img_name[0]], reinit=True)  # need to work it out
        # Create the network input
        net_input = get_noise(input_depth=wandb.config.input_depth, method='noise',
                              spatial_size=img.shape[2:], noise_type=args.noise_dist,
                              var=args.noise_std, noise_range=args.noise_range).type(dtype)
        net_input_saved = net_input.detach().clone()
        noise = net_input.detach().clone()
        # Convert the image and mask
        img = img.type(dtype)
        mask = mask.type(dtype)
        # Create the network
        net = create_network(net_name=wandb.config.net, input_depth=wandb.config.input_depth, pad=wandb.config.pad,
                             upsample=wandb.config.upsample, activation=wandb.config.activation,
                             need_sigmoid=args.need_sigmoid, num_channels=img.shape[1], dtype=dtype)
        # wandb.watch(net, log='gradients', idx=img_idx)  # log gradients of the model (don't care for now)
        # Create the optimization routine

        # Instatiate the loss functions and weight
        if wandb.config.loss_1 != wandb.config.loss_2:
            loss_1 = select_loss(wandb.config.loss_1, dtype)
            loss_2 = select_loss(wandb.config.loss_2, dtype)
        else:
            loss_1 = loss_2 = select_loss(wandb.config.loss_1, dtype)
            print('Watch out! The two losses are equal! Going to use a single one.')
        loss_balance = wandb.config.loss_balance

        # Instantiate the optimization routine
        routine = OptimizationRoutine(net=net, net_input_saved=net_input_saved, reg_noise_std=args.noise_std,
                                      noise=noise, mask_var=mask, img_var=img, loss_1=loss_1, loss_2=loss_2,
                                      loss_balance=wandb.config.loss_balance,
                                      param_noise=args.param_noise, log_int=args.log_int, num_iter=args.max_iter)
        # OPTIMIZATION LOOP #
        p = get_params('net', routine.net, net_input)
        tic = time.time()  # log time for optimization
        optimize(optimizer_type=args.optimizer, parameters=p, routine=routine, lr=args.lr,
                 num_iter=args.max_iter, schedule_lr_patience=args.lr_patience, schedule_lr_factor=args.lr_factor,
                 img_name=f'Sample {img_idx}')
        toc = time.time()
        wandb.log({'Optimization time': toc-tic, 'Img': img_idx})  # save optimization time
        # Compute PSNR as an additional metric
        out_img = torch_to_np(net(net_input))
        mse = np.mean((torch_to_np(img)-out_img)**2)
        wandb.log({'Final MSE': mse, 'Img': img_idx})
        wandb.log({'Final Peak-SNR': 10 * np.log10((img.max().detach().cpu().numpy() - img.min().detach().cpu().numpy()) ** 2 / mse),
                   'Img': img_idx})
        # Save the obtained images
        # WATCH OUT: already does it Wandb locally, but we might want to clean up after our mess and still preserving the images
        out_np = np.squeeze(torch_to_np(net(net_input)))
        inp_np = np.squeeze(torch_to_np(img).copy())
        if pol_bands == 'VVVH':
            inp_np[:, np.squeeze(mask.detach().cpu().numpy())==0] = out_np[:, np.squeeze(mask.detach().cpu().numpy())==0].copy()
        else:
            inp_np[np.squeeze(mask.detach().cpu().numpy())==0] = out_np[np.squeeze(mask.detach().cpu().numpy())==0].copy()
        np.save(os.path.join(save_dir, '{}_dip.npy'.format(img_idx)), out_np)
        np.save(os.path.join(save_dir, '{}_inp.npy'.format(img_idx)), inp_np)
        np.save(os.path.join(save_dir, '{}_mask.npy'.format(img_idx)), (np.squeeze(mask.detach().cpu().numpy())*255).astype(np.uint8))
        # Update the inpainted images DataFrame
        inp_df.iloc[img_idx, -4] = os.path.join(save_dir, '{}_inp.npy'.format(img_idx))
        inp_df.iloc[img_idx, -3] = os.path.join(save_dir, '{}_dip.npy'.format(img_idx))
        inp_df.iloc[img_idx, -2] = os.path.join(save_dir, '{}_mask.npy'.format(img_idx))
        # Log final loss
        with torch.no_grad():
            # Watch out for the signs of the losses!
            # During the optimization routine, we want to minimize -SSIM, and therefore we positively sum the MSE!
            # For the sweep, we want to maximize the final loss (SSIM), therefore we negatively sum the MSE!
            if loss_1.__class__ == SSIM:
                # final loss = SSIM - MSE
                final_loss.append(loss_1(net(net_input) * mask, img * mask).item() * loss_balance \
                                  - (1 - loss_balance) * loss_2(net(net_input) * mask, img * mask).item())
            elif loss_2.__class__ == SSIM:
                # final loss = -MSE + SSIM
                final_loss.append(-loss_1(net(net_input) * mask, img * mask).item() * loss_balance \
                                  + (1 - loss_balance) * loss_2(net(net_input) * mask, img * mask).item())
            else:
                final_loss.append(loss_1(net(net_input) * mask, img * mask).item() * loss_balance \
                                  + (1 - loss_balance) * loss_2(net(net_input) * mask, img * mask).item())
        wandb.log({'Average final dataset loss': np.mean(final_loss), 'Img': img_idx})

    # Save final inpainting DataFrame
    inp_df.to_pickle(os.path.join(save_dir, 'inpainting_df.pkl'))


def main():

    # Parser arguments #
    parser = argparse.ArgumentParser()
    # Execution params
    parser.add_argument('--gpu', type=int, default=None, help='GPU on which execute the code')
    parser.add_argument('--project_name', type=str, help='Name of the project for the WandB logging',
                        default='SEN12MS-SAR-DIP-inpainting')
    parser.add_argument('--output_dir', type=str, help='Directory for storing the results of the experiments',
                        required=True)
    parser.add_argument('--log_int', type=int, help='Logging interval for wandb', default=500)
    parser.add_argument('--debug', action='store_true', help='Debug flag')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix to add to the experiment tag')
    parser.add_argument('--SEN12MS_df', type=str,
                        help='DataFrame containing info on the whole SEN12MS (the all triplets)',
                        default='data/tiles_info_df.csv',)
    parser.add_argument('--classes_df', type=str,
                        help='DataFrame containing info on the classes of the SEN12MS triplets. Beware: this DataFrame'
                             'has been filtered to avoid working on U.S. regions',
                        default='data/histogram_norm_DFC_2020_scheme.csv',)
    parser.add_argument('--inp_classes', type=str, help='Classes to use for performing inpainting',
                        choices=["Forest", "Shrubland", "Savanna", "Grassland", "Wetlands",
                                 "Croplands", "Urban/Built-up", "Snow/Ice", "Barren", "Water"],
                        default='Urban/Built-up')
    parser.add_argument('--perc_area_cov', type=int,
                        help='Percentiles of area covered by the classes to be inpainted. This parameters allows to '
                             'filter the samples depending on the type of mixture of morphologies we want in the samples.',
                        choices=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90], default=90)
    parser.add_argument('--pol_bands', type=str, help='Polarization band to consider. For SEN12MS, VV and VH are available',
                        choices=['VV', 'VH', 'VVVH'], default='VV')
    parser.add_argument('--samples_per_class', type=int, help='Number of samples per morphology class to use',
                        default=10)
    parser.add_argument('--inp_size', type=int, help='Inpainting size. For the moment, the inpainted area is a square,'
                                                     'only 1 parameter needed',
                        default=64)
    parser.add_argument('--mz_score_norm', action='store_true', help='Whether to apply the modified Z-score normalization'
                                                                     'or not')
    # Training hyperparams
    parser.add_argument('--max_iter', type=int, help='Number of iteration on the sample', default=3001)
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-3)
    parser.add_argument('--optimizer', type=str, required=False, default='adam', choices=['adam', 'sgd'],
                        help='Optimizer to be used')
    parser.add_argument('--lr_patience', type=int, help='Patience for scheduling the learning rate on plateau',
                        default=250)
    parser.add_argument('--lr_factor', type=float, help='Factor for scheduling the learning rate on plateau',
                        default=1e-1)
    parser.add_argument('--loss_1', type=str, help='First loss function to use',
                        default='mse', choices=['mse', 'mae', 'ssim'])
    parser.add_argument('--loss_2', type=str, help='Second loss function to use',
                        default='mse', choices=['mse', 'mae', 'ssim'])
    parser.add_argument('--loss_balance', type=float, help='Weight to use for the losses', default=0.5)
    # Network hyperparams
    parser.add_argument('--net', type=str, help='Model to optimize',
                        choices=['skip_depth6', 'skip_depth4', 'skip_depth2',
                                 'Unet', 'ResNet', 'MultiResUnet'], required=True)
    parser.add_argument('--input_depth', type=int, help='Depth of the input noise tensor', default=1)
    parser.add_argument('--pad', type=str, required=False, default='zero', choices=['zero', 'reflection'],
                        help='Padding strategy for the network')
    parser.add_argument('--upsample', type=str, required=False, default='nearest',
                        choices=['nearest', 'bilinear', 'trilinear', 'deconv', 'bicubic'],
                        help='Upgoing deconvolution strategy for the network')
    parser.add_argument('--activation', type=str, default='LeakyReLU', required=False,
                        choices=['ReLU', 'Tanh', 'LeakyReLU'],
                        help='Activation function to be used in the convolution block [ReLU, Tanh, LeakyReLU]')
    parser.add_argument('--need_sigmoid', action='store_true',
                        help='Apply a sigmoid activation to the network output')
    # DIP strategies
    parser.add_argument('--reg_noise_std', type=float, help='Noise STD to add to net params each iteration', default=0)
    parser.add_argument('--param_noise', action='store_true', help='Add normal noise to the net params at'
                                                                   'each iteration')
    parser.add_argument('--noise_dist', type=str, default='normal', choices=['normal', 'uniform', 'rayleigh'],
                        help='Type of noise for the input tensor: normal, uniform')
    parser.add_argument('--noise_std', type=float, default=.1, required=False,
                        help='Standard deviation of the noise for the input tensor')
    parser.add_argument('--noise_range', type=str, default='-4:4', choices=['0:1', '-1:1', '-4:4', None],
                        help='Provides the range of normalization for the noise input')

    args = parser.parse_args()

    # Call main
    print('Starting DIP on SEN12MS tiles...')
    try:
        train(args)
    except Exception as e:
        print('Something happened! Error is {}'.format(e))
    print('Done! Bye!')


if __name__ == '__main__':
    main()
