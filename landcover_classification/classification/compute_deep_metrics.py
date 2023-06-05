"""
Script for computing the IS, FID and KID scores using the Pytorch Image Quality library
We will use the features extracted by three different classification networks:
1. an InceptionV3;
2. a ResNet50;
3. an EfficietNetB0.

Features must be precomputed!!!
"""

# --- Libraries import --- #
import torch
import numpy as np
import piq
import pickle
import pandas as pd
import argparse
import os
from tqdm import tqdm
import sys
sys.path.append('../..')
from ispl_utils.slack import ISPLSlack

# --- Helpers functions --- #

def main(args: argparse.Namespace):

    # Parse arguments
    model_names = args.models
    feats_dir = args.feats_dir
    batch_size = args.batch_size
    gpu = args.gpu

    # Set GPU
    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    print(f'Available devices are: {torch.cuda.device_count()}')
    print(f'Selected device is {device}')

    # Load features and store them in a dictionary
    feats_or, feats_gen = {model_name: dict() for model_name in model_names}, {model_name: dict() for model_name in model_names}

    # -- Original
    for model_name in model_names:
        path = os.path.join(feats_dir, f'{model_name}_original_features.pkl')
        assert os.path.exists(path)
        with open(path, 'rb') as src:
            feats_or[model_name] = pickle.load(src)

    # -- Generated
    for model_name in model_names:
        path = os.path.join(feats_dir, f'{model_name}_DIP_generated_features.pkl')
        assert os.path.exists(path)
        with open(path, 'rb') as src:
            feats_gen[model_name] = pickle.load(src)

    # Compute metrics in batch and in total
    stop = 400
    if args.debug:
        # Consider just a batch_size sample for each class
        stop = batch_size*2
        for model_name in feats_or.keys():
            for lc_class in feats_or[model_name].keys():
                feats_or[model_name][lc_class]['feats'] = feats_or[model_name][lc_class]['feats'][:batch_size*2]
                feats_gen[model_name][lc_class]['feats'] = feats_gen[model_name][lc_class]['feats'][:batch_size * 2]
                feats_or[model_name][lc_class]['pre_logits'] = feats_or[model_name][lc_class]['pre_logits'][:batch_size * 2]
                feats_gen[model_name][lc_class]['pre_logits'] = feats_gen[model_name][lc_class]['pre_logits'][:batch_size * 2]

    # -- Batch metrics
    deep_metrics_batches = pd.DataFrame(index=pd.MultiIndex.from_product([feats_or.keys(), feats_or[model_name].keys()]),
                                        columns=['IS', 'FID', 'KID'])
    for model_name in feats_or.keys():
        print(f'Processing {model_name} extracted features...')
        for lc_class in feats_or[model_name].keys():
            print(f'Processing {lc_class} samples...')

            # Prepare the DataFrame
            deep_metrics_batches.loc[(model_name, lc_class), 'IS'] = []
            deep_metrics_batches.loc[(model_name, lc_class), 'FID'] = []
            deep_metrics_batches.loc[(model_name, lc_class), 'KID'] = []

            for batch_idx0 in tqdm(np.arange(start=0, stop=stop, step=batch_size), desc='Images processed'):

                with torch.no_grad():
                    # Send features to GPU if available
                    if torch.cuda.is_available():
                        # Prepare features
                        feats_orig = torch.tensor(feats_or[model_name][lc_class]['feats'][batch_idx0:(batch_idx0 + 1) * batch_size]).cuda().to(device)
                        feats_orig = feats_orig.view(feats_orig.size(0), -1)  # reshape if not already reshaped
                        feats_generated = torch.tensor(
                            feats_gen[model_name][lc_class]['feats'][batch_idx0:(batch_idx0 + 1) * batch_size]).cuda().to(device)
                        feats_generated = feats_generated.view(feats_generated.size(0), -1)  # reshape if not already reshaped
                        # Prepare raw logits
                        rlogits_orig = torch.tensor(
                            feats_or[model_name][lc_class]['pre_logits'][batch_idx0:(batch_idx0 + 1) * batch_size]).cuda().to(device)
                        rlogits_orig = rlogits_orig.view(rlogits_orig.size(0), -1)  # reshape if not already reshaped
                        rlogits_gen = torch.tensor(
                            feats_gen[model_name][lc_class]['pre_logits'][batch_idx0:(batch_idx0 + 1) * batch_size]).cuda().to(device)
                        rlogits_gen = rlogits_gen.view(rlogits_gen.size(0), -1)  # reshape if not already reshaped


                        # Compute IS for the class considered
                        deep_metrics_batches.loc[(model_name, lc_class), 'IS'].append(
                            piq.IS(distance='l1')(rlogits_gen, rlogits_orig).detach().item())
                        # Compute FID for the class considered
                        deep_metrics_batches.loc[(model_name, lc_class), 'FID'].append(
                            piq.FID()(feats_generated, feats_orig).detach().item())
                        # Compute KID for the class considered
                        deep_metrics_batches.loc[(model_name, lc_class), 'KID'].append(
                            piq.KID()(feats_generated, feats_orig).detach().item())

            # clear GPU memory
            torch.cuda.empty_cache()

    # Save the computed metrics
    save_name = 'debug_metrics_batches_all_models.pkl' if args.debug else 'deep_metrics_batches_all_models.pkl'
    deep_metrics_batches.to_pickle(os.path.join(feats_dir, save_name))

    # -- All samples distribution
    deep_metrics_all = pd.DataFrame(
        index=pd.MultiIndex.from_product([feats_or.keys(), feats_or[model_name].keys()]),
        columns=['IS', 'FID', 'KID'])
    for model_name in feats_or.keys():
        print(f'Processing {model_name} extracted features...')
        for lc_class in feats_or[model_name].keys():
            print(f'Processing {lc_class} samples...')

            with torch.no_grad():
                # Send features to GPU if available
                if torch.cuda.is_available():
                    # Prepare features
                    feats_orig = torch.tensor(feats_or[model_name][lc_class]['feats']).cuda().to(device)
                    feats_orig = feats_orig.view(feats_orig.size(0), -1)  # reshape if not already reshaped
                    feats_generated = torch.tensor(feats_gen[model_name][lc_class]['feats']).cuda().to(device)
                    feats_generated = feats_generated.view(feats_generated.size(0), -1)  # reshape if not already reshaped
                    # Prepare raw logits
                    rlogits_orig = torch.tensor(feats_or[model_name][lc_class]['pre_logits']).cuda().to(device)
                    rlogits_orig = rlogits_orig.view(rlogits_orig.size(0), -1)  # reshape if not already reshaped
                    rlogits_gen = torch.tensor(feats_gen[model_name][lc_class]['pre_logits']).cuda().to(device)
                    rlogits_gen = rlogits_gen.view(rlogits_gen.size(0), -1)  # reshape if not already reshaped

                    # Compute IS for the class considered
                    deep_metrics_all.loc[(model_name, lc_class), 'IS'] = \
                        piq.IS(distance='l1')(rlogits_gen, rlogits_orig).detach().item()
                    # Compute FID for the class considered
                    deep_metrics_all.loc[(model_name, lc_class), 'FID'] = \
                        piq.FID()(feats_generated, feats_orig).detach().item()
                    # Compute KID for the class considered
                    deep_metrics_all.loc[(model_name, lc_class), 'KID'] = \
                        piq.KID()(feats_generated, feats_orig).detach().item()

        # clear GPU memory
        torch.cuda.empty_cache()

    # Save the computed metrics
    save_name = 'debug_metrics_all_samples_all_models.pkl' if args.debug else 'deep_metrics_all_samples_all_models.pkl'
    deep_metrics_all.to_pickle(os.path.join(feats_dir, save_name))

    return 1


# --- MAIN CALL --- #
if __name__ == '__main__':

    # Create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str, nargs='+', help='Models to evaluate when computing the metrics',
                        default='InceptionV3')
    parser.add_argument('--feats_dir', type=str, help='Directory where to find the extracted features',
                        default='/nas/public/exchange/sar_inpainting/code/dip_sar_inpainting/notebook')
    parser.add_argument('--batch_size', type=int, help='Batch size to consider when computing metrics', default=8)
    parser.add_argument('--gpu', type=int, help='GPU to consider for executing the computations', default=1)
    parser.add_argument('--slack_user', type=str, help='User in the ISPL Slack workspace to warn about the script'
                                                       'execution', default='edo.cannas')
    parser.add_argument('--debug', action='store_true', help='Whether to execute the code in debug mode')
    args = parser.parse_args()

    # call main
    print('Starting analysis on test tiles...')
    if args.slack_user is not None:
        slack_m = ISPLSlack()
        slack_m.to_user(recipient=args.slack_user, message=f'Starting computation of deep metrics on GPU {args.gpu}...')
    # try:
    #     main(args)
    # except Exception as e:
    #     print('Something happened! Error is {}'.format(e))
    #     if args.slack_user is not None:
    #         slack_m.to_user(recipient=args.slack_user, message='Something happened! Error is {}'.format(e))
    # print('Done! Bye!')
    main(args)
    if args.slack_user is not None:
        slack_m.to_user(recipient=args.slack_user, message='Computation of deep metrics done!')