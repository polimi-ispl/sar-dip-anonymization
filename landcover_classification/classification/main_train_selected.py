"""
Main train selected
Train script on a selected number of regions and tiles.
Modified starting from the original main_train.py script

Author: Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
"""


import os
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm
import pandas as pd

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

import shutil
import sys
from dataset import SEN12MS_selected, ToTensor, Normalize
from models.VGG import VGG16, VGG19
from models.ResNet import ResNet50, ResNet101, ResNet152
from models.DenseNet import DenseNet121, DenseNet121FullDropout, DenseNet161, DenseNet169, DenseNet201
from models.EfficientNet import EfficientNetGen
from models.InceptionNet import ISPLInceptionV3
from metrics import MetricTracker, Precision_score, Recall_score, F1_score, \
    F2_score, Hamming_loss, Subset_accuracy, Accuracy_score, One_error, \
    Coverage_error, Ranking_loss, LabelAvgPrec_score

# sec.2 (done)

model_choices = ['VGG16', 'VGG19',
                 'ResNet50', 'ResNet101', 'ResNet152',
                 'DenseNet121', 'DenseNet121FullDropout', 'DenseNet161', 'DenseNet169', 'DenseNet201',
                 'EfficientNetB0', 'EfficientNetB4', 'InceptionV3']
label_choices = ['multi_label', 'single_label']


# ----------------------------- saving files ---------------------------------
def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in vars(args).items():
            f.write('%s: %s\n' % (key, str(value)))


def save_checkpoint(state, is_best, name):
    filename = os.path.join(checkpoint_dir, name + '_checkpoint.pth')

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(checkpoint_dir, name +
                                               '_model_best.pth'))


# -------------------------------- Main Program -------------------------------
def main(args: argparse.Namespace):

    # save configuration to file
    sv_name = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
    print('saving file name is ', sv_name)

    write_arguments_to_file(args, os.path.join(logs_dir, sv_name + '_arguments.txt'))

    # ----------------------------------- data
    # define mean/std of the training set (for data normalization)
    label_type = args.label_type

    bands_mean = {'s1_mean': [-11.76858, -18.294598],
                  's2_mean': [1226.4215, 1137.3799, 1139.6792, 1350.9973, 1932.9058,
                              2211.1584, 2154.9846, 2409.1128, 2001.8622, 1356.0801]}

    bands_std = {'s1_std': [4.525339, 4.3586307],
                 's2_std': [741.6254, 740.883, 960.1045, 946.76056, 985.52747,
                            1082.4341, 1057.7628, 1136.1942, 1132.7898, 991.48016]}

    # load datasets

    # Load the SEN12MS dataframes #
    tiles_df = pd.read_csv(args.SEN12MS_df, index_col=0)
    classes_df = pd.read_csv(args.classes_df, index_col=0)

    # Create a copy of the DataFrame of all tiles, re-organize the index structure to match the original code #
    tiles_df['id'] = tiles_df['path'].apply(lambda x:
                                            x.split('/')[-1].replace('_s1_', '_s2_').
                                            replace('_lc_', '_s2_'))  # add ID to all tiles
    ro_tiles_df = tiles_df.loc[tiles_df['sensor'] != 'lc'].set_index(['id', 'sensor'],
                                                                     drop=True, verify_integrity=True)  # set new index

    # Samples filtering #
    # Parse arguments
    inp_classes = args.inp_classes
    perc_area_cov = 90
    n_samples_per_class = 400
    # Filter classes
    classes_df = classes_df[['path', 'seed', 'season', 'region', 'sensor', 'tile', ] + inp_classes]
    # Select n samples per class
    inp_df = []
    for inp_class in inp_classes:
        inp_df.append(classes_df.loc[classes_df[inp_class] >= perc_area_cov].sample(n=n_samples_per_class, random_state=42))
    inp_df = pd.concat(inp_df)  # concatenate everything

    # Remove the samples used for creating the synthetic DIP tiles
    filt_ids = tiles_df.loc[tiles_df['path'].isin(inp_df['path'])]['id']  # select all samples not used for inpainting
    filt_df = ro_tiles_df.loc[~ro_tiles_df.index.get_level_values(0).isin(filt_ids)]

    imgTransform = transforms.Compose([ToTensor(), Normalize(bands_mean, bands_std)])

    train_dataGen = SEN12MS_selected(args.label_split_dir,
                            imgTransform=imgTransform,
                            label_type=label_type, threshold=args.threshold,
                            use_s1=args.use_s1, use_s2=args.use_s2, use_RGB=args.use_RGB,
                            IGBP_s=args.IGBP_simple, data_df=filt_df, data_root=args.data_dir)

    val_dataGen = SEN12MS_selected(args.label_split_dir,
                          imgTransform=imgTransform,
                          label_type=label_type, threshold=args.threshold,
                          use_s1=args.use_s1, use_s2=args.use_s2, use_RGB=args.use_RGB,
                          IGBP_s=args.IGBP_simple, data_df=filt_df, data_root=args.data_dir)

    # number of input channels
    n_inputs = train_dataGen.n_inputs
    print('input channels =', n_inputs)

    # Split in training and validation
    dataset_idxs = list(range(len(filt_df.index.get_level_values(0).unique())))
    np.random.seed(args.split_seed)  # setting the seed for training-val split
    np.random.shuffle(dataset_idxs)
    test_split_index = int(np.floor((1 - args.perc_train_test) * len(filt_df.index.get_level_values(0).unique())))
    train_val_idxs, test_idxs = dataset_idxs[test_split_index:], dataset_idxs[:test_split_index]
    val_split_index = int(np.floor((1 - args.perc_train_val) * len(train_val_idxs)))
    train_idx, val_idx = train_val_idxs[val_split_index:], train_val_idxs[:val_split_index]
    if debug:
        train_idx = train_idx[:int(len(train_idx) / 10)]
        val_idx = val_idx[:int(len(val_idx) / 10)]

    # Create Samplers #
    train_sampler = SubsetRandomSampler(train_idx, generator=torch.Generator().manual_seed(args.split_seed))
    val_sampler = SubsetRandomSampler(val_idx, generator=torch.Generator().manual_seed(args.split_seed))

    # set up dataloaders
    train_data_loader = DataLoader(train_dataGen,
                                   batch_size=args.batch_size,
                                   num_workers=args.num_workers,
                                   shuffle=False,
                                   pin_memory=True, sampler=train_sampler, drop_last=True)
    val_data_loader = DataLoader(val_dataGen,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=False,
                                 pin_memory=True, sampler=val_sampler, drop_last=True)

    # -------------------------------- ML setup
    # cuda
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.backends.cudnn.enabled = True
        cudnn.benchmark = True

    # define number of classes
    if args.IGBP_simple:
        numCls = 10
    else:
        numCls = 17

    print('num_class: ', numCls)

    # define model

    # --- VGGs --- #
    if args.model == 'VGG16':
        model = VGG16(n_inputs, numCls)
    elif args.model == 'VGG19':
        model = VGG19(n_inputs, numCls)
    # --- ResNets --- #
    elif args.model == 'ResNet50':
        model = ResNet50(n_inputs, numCls)
    elif args.model == 'ResNet101':
        model = ResNet101(n_inputs, numCls)
    elif args.model == 'ResNet152':
        model = ResNet152(n_inputs, numCls)
    # --- DenseNets --- #
    elif args.model == 'DenseNet121':
        model = DenseNet121(n_inputs, numCls)
    elif args.model == 'DenseNet121FullDropout':
        model = DenseNet121FullDropout(n_inputs, numCls)
    elif args.model == 'DenseNet161':
        model = DenseNet161(n_inputs, numCls)
    elif args.model == 'DenseNet169':
        model = DenseNet169(n_inputs, numCls)
    elif args.model == 'DenseNet201':
        model = DenseNet201(n_inputs, numCls)
    # --- EfficietNets --- #
    elif args.model == 'EfficientNetB0':
        model = EfficientNetGen(model='efficientnet-b0', n_classes=numCls,
                                in_channels=n_inputs, pretrained=True)
    elif args.model == 'EfficientNetB4':
        model = EfficientNetGen(model='efficientnet-b4', n_classes=numCls,
                                in_channels=n_inputs, pretrained=True)
    # --- InceptionV3 --- #
    elif args.model == 'InceptionV3':
        model = ISPLInceptionV3(num_classes=numCls, in_channels=n_inputs, pretrained=True)
    else:
        raise NameError("no model")

    # move model to GPU if is available
    if use_cuda:
        model = model.cuda()

        # define loss function
    if label_type == 'multi_label':
        lossfunc = torch.nn.BCEWithLogitsLoss()
    else:
        lossfunc = torch.nn.CrossEntropyLoss()

    # set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    best_acc = 0
    start_epoch = 0
    epochs = 2 if debug else args.epochs
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            checkpoint_nm = os.path.basename(args.resume)
            sv_name = checkpoint_nm.split('_')[0] + '_' + checkpoint_nm.split('_')[1]
            print('saving file name is ', sv_name)

            if checkpoint['epoch'] > start_epoch:
                start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_prec']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # set up tensorboard logging
    train_writer = SummaryWriter(os.path.join(logs_dir, 'runs', sv_name, 'training'))
    val_writer = SummaryWriter(os.path.join(logs_dir, 'runs', sv_name, 'val'))

    # ----------------------------- executing Train/Val.
    # train network
    for epoch in range(start_epoch, epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        train(train_data_loader, model, optimizer, lossfunc, label_type, epoch, use_cuda, train_writer)
        micro_f1 = val(val_data_loader, model, optimizer, label_type, epoch, use_cuda, val_writer)

        is_best_acc = micro_f1 > best_acc
        best_acc = max(best_acc, micro_f1)

        save_checkpoint({
            'epoch': epoch,
            'arch': args.model,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_prec': best_acc
        }, is_best_acc, sv_name)


def train(trainloader, model, optimizer, lossfunc, label_type, epoch, use_cuda, train_writer):
    lossTracker = MetricTracker()

    # set model to train mode
    model.train()

    # main training loop
    for idx, data in enumerate(tqdm(trainloader, desc="training")):
    # for idx, data in enumerate(trainloader):

        numSample = data["image"].size(0)

        # unpack sample
        bands = data["image"]
        if label_type == 'multi_label':
            labels = data["label"]
        else:
            labels = (torch.max(data["label"], 1)[1]).type(torch.long)

            # move data to gpu if model is on gpu
        if use_cuda:
            bands = bands.to(torch.device("cuda"))
            labels = labels.to(torch.device("cuda"))

        # reset gradients
        optimizer.zero_grad()

        # forward pass
        logits = model(bands)
        loss = lossfunc(logits, labels)

        # backward pass
        loss.backward()
        optimizer.step()

        #
        lossTracker.update(loss.item(), numSample)

    train_writer.add_scalar("loss", lossTracker.avg, epoch)

    print('Train loss: {:.6f}'.format(lossTracker.avg))


def val(valloader, model, optimizer, label_type, epoch, use_cuda, val_writer):
    prec_score_ = Precision_score()
    recal_score_ = Recall_score()
    f1_score_ = F1_score()
    f2_score_ = F2_score()
    hamming_loss_ = Hamming_loss()
    subset_acc_ = Subset_accuracy()
    acc_score_ = Accuracy_score()
    one_err_ = One_error()
    coverage_err_ = Coverage_error()
    rank_loss_ = Ranking_loss()
    labelAvgPrec_score_ = LabelAvgPrec_score()

    # set model to evaluation mode
    model.eval()

    # main validation loop
    y_true = []
    predicted_probs = []

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(valloader, desc="validation")):

            # unpack sample
            bands = data["image"]
            labels = data["label"]

            # move data to gpu if model is on gpu
            if use_cuda:
                bands = bands.to(torch.device("cuda"))
                # labels = labels.to(torch.device("cuda"))

            # forward pass
            logits = model(bands)

            # convert logits to probabilies
            if label_type == 'multi_label':
                probs = torch.sigmoid(logits).cpu().numpy()
            else:
                sm = torch.nn.Softmax(dim=1)
                probs = sm(logits).cpu().numpy()

            labels = labels.cpu().numpy()  # keep true & pred label at same loc.
            predicted_probs += list(probs)
            y_true += list(labels)

    predicted_probs = np.asarray(predicted_probs)
    # convert predicted probabilities into one/multi-hot labels
    if label_type == 'multi_label':
        y_predicted = (predicted_probs >= 0.5).astype(np.float32)
    else:
        loc = np.argmax(predicted_probs, axis=-1)
        y_predicted = np.zeros_like(predicted_probs).astype(np.float32)
        for i in range(len(loc)):
            y_predicted[i, loc[i]] = 1

    y_true = np.asarray(y_true)

    macro_f1, micro_f1, sample_f1 = f1_score_(y_predicted, y_true)
    macro_f2, micro_f2, sample_f2 = f2_score_(y_predicted, y_true)
    macro_prec, micro_prec, sample_prec = prec_score_(y_predicted, y_true)
    macro_rec, micro_rec, sample_rec = recal_score_(y_predicted, y_true)
    hamming_loss = hamming_loss_(y_predicted, y_true)
    subset_acc = subset_acc_(y_predicted, y_true)
    macro_acc, micro_acc, sample_acc = acc_score_(y_predicted, y_true)

    # Note that below 4 ranking-based metrics are not applicable to single-label
    # (multi-class) classification, but they will still show the scores during
    # validation on tensorboard
    one_error = one_err_(predicted_probs, y_true)
    coverage_error = coverage_err_(predicted_probs, y_true)
    rank_loss = rank_loss_(predicted_probs, y_true)
    labelAvgPrec = labelAvgPrec_score_(predicted_probs, y_true)

    info = {
        "macroPrec": macro_prec,
        "microPrec": micro_prec,
        "samplePrec": sample_prec,
        "macroRec": macro_rec,
        "microRec": micro_rec,
        "sampleRec": sample_rec,
        "macroF1": macro_f1,
        "microF1": micro_f1,
        "sampleF1": sample_f1,
        "macroF2": macro_f2,
        "microF2": micro_f2,
        "sampleF2": sample_f2,
        "HammingLoss": hamming_loss,
        "subsetAcc": subset_acc,
        "macroAcc": macro_acc,
        "microAcc": micro_acc,
        "sampleAcc": sample_acc,
        "oneError": one_error,
        "coverageError": coverage_error,
        "rankLoss": rank_loss,
        "labelAvgPrec": labelAvgPrec
    }

    for tag, value in info.items():
        val_writer.add_scalar(tag, value, epoch)

    print('Validation microPrec: {:.6f} microF1: {:.6f} sampleF1: {:.6f} microF2: {:.6f} sampleF2: {:.6f}'.format(
        micro_prec,
        micro_f1,
        sample_f1,
        micro_f2,
        sample_f2
    ))
    return micro_f1


if __name__ == "__main__":
    # ----------------------- define and parse arguments --------------------------
    parser = argparse.ArgumentParser()

    # experiment name
    parser.add_argument('--exp_dir', type=str,
                        default='/nas/public/exchange/sar_inpainting/experiments/lc_classification',
                        help='Directory where to store the training results')
    parser.add_argument('--exp_name', type=str, default=None,
                        help="experiment name. will be used in the path names \
                             for log- and savefiles. If no input experiment name, \
                             path would be set to model name.")

    # data directory
    parser.add_argument('--data_dir', type=str, required=True,
                        help='path to SEN12MS dataset')
    parser.add_argument('--label_split_dir', type=str, required=True,
                        help="path to label data and split list")
    parser.add_argument('--SEN12MS_df', type=str,
                        help='DataFrame containing info on the whole SEN12MS (the all triplets)',
                        default='../data/tiles_info_df.csv', )
    parser.add_argument('--classes_df', type=str,
                        help='DataFrame containing info on the classes of the SEN12MS triplets. Beware: this DataFrame'
                             'has been filtered to avoid working on U.S. regions',
                        default='../data/histogram_norm_DFC_2020_scheme.csv', )
    parser.add_argument('--inp_classes', type=str, help='Classes to use for performing inpainting',
                        choices=["Forest", "Shrubland", "Savanna", "Grassland", "Wetlands",
                                 "Croplands", "Urban/Built-up", "Snow/Ice", "Barren", "Water"],
                        default=["Forest", "Shrubland", "Grassland",
                                 "Croplands", "Urban/Built-up", "Barren", "Water"], nargs='+')
    parser.add_argument('--perc_train_test', type=float, default=0.75, help='Train-test split percentage')
    parser.add_argument('--perc_train_val', type=float, default=0.75, help='Train-val split percentage')
    parser.add_argument('--split_seed', type=int, default=42, help='Seed use for splitting the dataset randomly')
    parser.add_argument('--debug', action='store_true', help='Whether to execute the code in debugging mode')

    # input/output
    parser.add_argument('--use_s2', action='store_true', default=False,
                        help='use sentinel-2 bands')
    parser.add_argument('--use_s1', action='store_true', default=False,
                        help='use sentinel-1 data')
    parser.add_argument('--use_RGB', action='store_true', default=False,
                        help='use sentinel-2 RGB bands')
    parser.add_argument('--IGBP_simple', action='store_true', default=True,
                        help='use IGBP simplified scheme; otherwise: IGBP original scheme')
    parser.add_argument('--label_type', type=str, choices=label_choices,
                        default='multi_label',
                        help="label-type (default: multi_label)")
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='threshold to convert probability-labels to multi-hot \
                        labels, mean/std for normalizatin would not be accurate \
                        if the threshold is larger than 0.22. \
                        for single_label threshold would be ignored')

    # network
    parser.add_argument('--model', type=str, choices=model_choices,
                        default='ResNet50',
                        help="network architecture (default: ResNet50)")

    # training hyperparameters
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--decay', type=float, default=1e-5,
                        help='decay rate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='mini-batch size (default: 64)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num_workers for data loading in pytorch')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs (default: 100)')
    parser.add_argument('--resume', '-r', type=str, default=None,
                        help='path to the pretrained weights file', )

    args = parser.parse_args()

    # -------------------- set directory for saving files -------------------------
    debug = args.debug
    exp_name = '' if not args.exp_name else args.exp_name
    exp_name = f'debug_{exp_name}' if debug else exp_name
    exp_name += f'_model-{args.model}_perc_train_test-{args.perc_train_test}_perc_train_val-{args.perc_train_val}_' \
                f'seed-{args.split_seed}'
    if exp_name:
        checkpoint_dir = os.path.join(args.exp_dir, exp_name, 'checkpoints')
        logs_dir = os.path.join(args.exp_dir, exp_name, 'logs')
    else:
        checkpoint_dir = os.path.join(args.exp_dir, args.model, 'checkpoints')
        logs_dir = os.path.join(args.exp_dir, args.model, 'logs')

    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    if not os.path.isdir(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)

    # --- RUN THE MAIN FUNCTION --- #
    main(args)

