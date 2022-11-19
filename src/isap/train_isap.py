import os
import pickle as pkl
import numpy as np
import argparse
import tqdm.notebook as tq

import sys
sys.path.append("../../") # Add directory containing data/nuscenes to path
print("Current Working Directory " , os.getcwd())

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

from nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.helper import *

from nuscenes.prediction.models.backbone import ResNetBackbone
from nuscenes.prediction.models.mtp import MTP
from post_covernet_decode_double_latent import PostCoverNet, UCELoss
from nuscenes.prediction.models.covernet import CoverNet, ConstantLatticeLoss
from nuscenes.eval.prediction.metrics import *

from dataset_ood_norm_input_decode_utils import *
import model_ood_norm_input_decode_double_latent_utils
from model_ood_norm_input_decode_double_latent_utils import *

def populate_common_params(
        arg_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # random seed
    arg_parser.add_argument("--random_seed", type=int, default=123,
                            help="Set random seed")
    
    # backbone
    arg_parser.add_argument("--backbone", type=str, default="resnet18",
                            help="Choose backbone architecture")
    
    # num_modes
    arg_parser.add_argument("--num_modes", type=int, default=64,
                            help="Set number of trajectory modes (default: 64)")

    # trainer params
    arg_parser.add_argument("--n_epochs", type=int, default=25,
                            help="Number of epochs to train (default: 25)")
    arg_parser.add_argument("--load_from_checkpoint", type=str, default=None,
                            help="""If the parameter is set, model,
                            trainer, and optimizer states are loaded from the
                            checkpoint (default: None)""")

    # dataset
    arg_parser.add_argument("--batch_size", type=int, default=16,
                            help="Input batch size for training (default: 16)")

    # optimizer
    arg_parser.add_argument("--optimizer", type=str, default="sgd",
                            choices=["adam", "sgd", "adagrad"],
                            help="Optimizer to use [adam, sgd, adagrad] (default: sgd)")
    arg_parser.add_argument("--lr", type=float, default=1e-3,
                            help="Learning rate (default: 1e-4)")
    arg_parser.add_argument("--weight_decay", type=float, default=5e-4,
                            help="L2 regularization constant (default: 5e-4)")
    arg_parser.add_argument("--n_density", type=int, default=8,
                            help="Flow length (default: 8)")
    arg_parser.add_argument("--n_hidden_layers", type=int, default=4096, # 512
                            help="Hidden layer size (default: 4096)")
    arg_parser.add_argument("--ood_dist", type=float, default=10.0,
                            help="Hidden layer size (default: 10)")

    return arg_parser

def main(opts):
    # Set-up parameters and datasets.
    model_name = 'post_covernet_decode_double_latent_agent_norm_10'

    # Fix seed.
    pl.seed_everything(opts.random_seed)

    tb_logger = pl_loggers.TensorBoardLogger(
        'logs/',
        name=model_name)

    tb_logger.log_hyperparams(opts, metrics=None)

    # Provide dataset root.
    dataroot = '../../data/nuscenes'
    nuscenes = NuScenes('v1.0-trainval', dataroot=dataroot)
    dataset = NuScenesDataModule(dataroot, nuscenes, batch_size=opts.batch_size, dist=opts.ood_dist)

    dataset.setup(stage='fit')
    dataset.setup(stage='val')
    dataset.setup(stage='test')
    print(len(dataset.train))

    # Update certainty budget.
    N = dataset.train.N
    H = 6
    N = dataset.train.N/torch.sum(dataset.train.N)*np.exp(H)

    # Init and train model.
    model = PredictionModel(N=N, num_modes=opts.num_modes, lr=opts.lr, batch_size=opts.batch_size, optimizer=opts.optimizer, weight_decay=opts.weight_decay, n_density=opts.n_density, backbone_name=opts.backbone, n_hidden_layers=[opts.n_hidden_layers], path_to_epsilon_set="../../data/nuscenes/covernet_traj_set/nuscenes-prediction-challenge-trajectory-sets/epsilon_8.pkl")
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_last=True)

    trainer = pl.Trainer(
        logger=tb_logger,
        max_epochs=opts.n_epochs,
        weights_save_path='checkpoints/',
        weights_summary='top',
        gpus=1 if torch.cuda.is_available() else 0,
        resume_from_checkpoint=opts.load_from_checkpoint,
        deterministic=True,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=1)

    trainer.fit(model, dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    arg_parser = populate_common_params(parser)

    opts = arg_parser.parse_args()

    main(opts)
