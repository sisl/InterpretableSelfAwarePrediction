import os
import pickle as pkl
import numpy as np
import argparse
import tqdm.notebook as tq

import sys
sys.path.append("../../") # Add directory containing src/data to path
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
from nuscenes.prediction.models.covernet import CoverNet, ConstantLatticeLoss
from nuscenes.eval.prediction.metrics import *

from dataset_ood_norm_input_utils import *
from model_ood_norm_input_utils import PredictionModel as CoverNetPredictionModel

import tikzplotlib
from scipy import signal

def populate_common_params(
        arg_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # random seed
    arg_parser.add_argument("--random_seed", type=str, default='123,100,150,0,5,10,20,30,40,50',
                            help="Set random seed")

    # Size of ensemble
    arg_parser.add_argument("--N", type=int, default=10,
                            help="Set ensemble size")
    
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
    random_seed = opts.random_seed.split(',')
    for i in range(len(random_seed)):
        random_seed[i] = int(random_seed[i])
 
    model_name = 'covernet_agent_norm_10'

    # Fix seed.
    pl.seed_everything(random_seed[0])

    # Provide dataset root.
    dataroot = '../../data/nuscenes'
    nuscenes = NuScenes('v1.0-trainval', dataroot=dataroot)
    dataset = NuScenesDataModule(dataroot, nuscenes, batch_size=opts.batch_size, dist=opts.ood_dist)

    dataset.setup(stage='fit')
    dataset.setup(stage='val')
    dataset.setup(stage='test')
    
    # Construct the ensemble.
    PATH_TO_EPSILON_8_SET = "../../data/nuscenes/covernet_traj_set/nuscenes-prediction-challenge-trajectory-sets/epsilon_8.pkl"
    trajectories = pkl.load(open(PATH_TO_EPSILON_8_SET, 'rb'))
    trajectories = torch.Tensor(trajectories).cuda()

    ground_truth_preds_test = pkl.load(open(model_name + "_ground_truth_preds_test_last.pkl", "rb"))
    b = len(ground_truth_preds_test)
    N = opts.N
    classes = 64
    logits_test = torch.Tensor(b,N,1,classes)
    for i in range(N):
        logits_test[:,i] = torch.nn.functional.softmax(torch.Tensor(pkl.load(open(model_name + "_" + str(random_seed[i]) + "_logits_test_last.pkl", "rb"))), dim=-1)
    
    logits_test_mean = torch.mean(logits_test, dim=1)
    
    c = torch.argmax(logits_test_mean, dim=-1, keepdims=True)
    c = c.unsqueeze(1).repeat(1,N,1,1)
    var_test = torch.gather(logits_test, -1, c)
    var_test = 1./torch.var(var_test, dim=1, unbiased=False)
    var_test[var_test > torch.finfo(torch.float32).max/10000] = torch.finfo(torch.float32).max/10000

    covernet_preds_test = trajectories[torch.Tensor(logits_test_mean).argsort(descending=True, dim=-1)[:]].cuda()

    # Evaluate on the trajectory metrics.
    ground_truth_preds_np = np.array(ground_truth_preds_test)
    covernet_preds_np = covernet_preds_test.cpu().data.numpy()

    num_modes = 1
    ade_1 = np.mean(np.amin(np.linalg.norm(covernet_preds_np[:,0,0:num_modes] - ground_truth_preds_np[:,:], axis=-1), axis=(1)))
    num_modes = 5
    ade_5 = np.mean(np.amin(np.linalg.norm(covernet_preds_np[:,0,0:num_modes] - ground_truth_preds_np[:,:], axis=-1), axis=(1)))
    num_modes = 10
    ade_10 = np.mean(np.amin(np.linalg.norm(covernet_preds_np[:,0,0:num_modes] - ground_truth_preds_np[:,:], axis=-1), axis=(1)))
    num_modes = 15
    ade_15 = np.mean(np.amin(np.linalg.norm(covernet_preds_np[:,0,0:num_modes] - ground_truth_preds_np[:,:], axis=-1), axis=(1)))
    fde = np.mean(np.linalg.norm(covernet_preds_np[:,0,0:1,-1] - ground_truth_preds_np[:,:,-1], axis=-1))
    print(ground_truth_preds_np.shape, covernet_preds_np.shape)
    print("ADE_1", ade_1)
    print("ADE_5", ade_5)
    print("ADE_10", ade_10)
    print("ADE_15", ade_15)
    print("FDE", fde)
        
    # Get trajectory classification labels and in-distribution uncertainty metrics.
    ground_truth_preds_test_tensor = torch.Tensor(ground_truth_preds_test).cuda()
    logits_test_in = torch.Tensor(logits_test_mean).cuda()
    var_test_in = torch.Tensor(var_test).cuda()
    labels = []
    for i in range(ground_truth_preds_test_tensor.shape[0]):
        closest_trajectory = mean_pointwise_l2_distance(trajectories, ground_truth_preds_test_tensor[i])
        label = torch.LongTensor([closest_trajectory]).cuda()
        labels.append(nn.functional.one_hot(label, num_classes=trajectories.shape[0]))
    labels = torch.stack(labels)
    
    print("Brier Score", brier_score(torch.argmax(labels[:,0,:],dim=-1), logits_test_in[:,0,:]))
    print("Aleatoric Confidence", confidence(torch.argmax(labels[:,0,:],dim=-1), logits_test_in[:,0,:], uncertainty_type='aleatoric'))
    print("Epistemic Confidence", confidence(torch.argmax(labels[:,0,:],dim=-1), var_test_in[:,0,:], uncertainty_type='epistemic'))

    # Compute out-of-distribution uncertainty metrics.
    b = torch.nn.functional.softmax(torch.Tensor(pkl.load(open(model_name + "_" + str(random_seed[0]) + "_logits_ood_test_last.pkl", "rb"))), dim=-1).shape[0]
    logits_test_ood_sep = torch.Tensor(b,N,1,classes)
    for i in range(N):
        logits_test_ood_sep[:,i] = torch.nn.functional.softmax(torch.Tensor(pkl.load(open(model_name + "_" + str(random_seed[i]) + "_logits_ood_test_last.pkl", "rb"))), dim=-1)
    logits_test_ood = torch.mean(logits_test_ood_sep, dim=1)

    c = torch.argmax(logits_test_ood, dim=-1, keepdims=True)
    c = c.unsqueeze(1).repeat(1,N,1,1)
    var_test_ood = torch.gather(logits_test_ood_sep, -1, c)
    var_test_ood = 1./torch.var(var_test_ood, dim=1, unbiased=False)
    var_test_ood[var_test_ood > torch.finfo(torch.float32).max/10000] = torch.finfo(torch.float32).max/10000

    logits_test_in_tensor = torch.Tensor(logits_test_in).cuda()
    logits_test_ood_tensor = torch.Tensor(logits_test_ood).cuda()
    var_test_in_tensor = torch.Tensor(var_test_in).cuda()
    var_test_ood_tensor = torch.Tensor(var_test_ood).cuda()
    
    print("Aleatoric OOD APR", anomaly_detection(alpha=logits_test_in_tensor[:,0], ood_alpha=logits_test_ood_tensor[:,0], score_type='APR', uncertainty_type='aleatoric', model='ensemble'))
    print("Epistemic OOD APR", anomaly_detection(alpha=var_test_in_tensor[:,0], ood_alpha=var_test_ood_tensor[:,0], score_type='APR', uncertainty_type='epistemic', model='ensemble'))
    print("Aleatoric OOD AUROC", anomaly_detection(alpha=logits_test_in_tensor[:,0], ood_alpha=logits_test_ood_tensor[:,0], score_type='AUROC', uncertainty_type='aleatoric', model='ensemble'))
    print("Epistemic OOD AUROC", anomaly_detection(alpha=var_test_in_tensor[:,0], ood_alpha=var_test_ood_tensor[:,0], score_type='AUROC', uncertainty_type='epistemic', model='ensemble'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    arg_parser = populate_common_params(parser)

    opts = arg_parser.parse_args()

    main(opts)
