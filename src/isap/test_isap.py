import os
import pickle as pkl
import numpy as np
import argparse
import tqdm.notebook as tq

import sys
sys.path.append("../") # Add directory containing src/data to path
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

from posterior_network.src.results_manager.metrics_prior import *
from post_covernet_decode_double_latent import mean_pointwise_l2_distance

import tikzplotlib
from scipy import signal

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

    # Provide dataset root.
    dataroot = '../../data/nuscenes'
    nuscenes = NuScenes('v1.0-trainval', dataroot=dataroot)
    dataset = NuScenesDataModule(dataroot, nuscenes, batch_size=opts.batch_size, dist=opts.ood_dist)

    dataset.setup(stage='fit')
    dataset.setup(stage='val')
    dataset.setup(stage='test')

    dataset_ood = NuScenesDataModule(dataroot, nuscenes, batch_size=opts.batch_size, dist=opts.ood_dist, ood=True)
    dataset_ood.setup(stage='val')
    dataset_ood.setup(stage='test')

    # Evaluation
    # Path to trajectory set downloaded from NuScenes.
    PATH_TO_EPSILON_8_SET = "../../data/nuscenes/covernet_traj_set/nuscenes-prediction-challenge-trajectory-sets/epsilon_8.pkl"
    trajectories = pkl.load(open(PATH_TO_EPSILON_8_SET, 'rb'))
    trajectories = torch.Tensor(trajectories).cuda()

    model_postcovernet_test = PredictionModel.load_from_checkpoint('/epistemic_bp/checkpoints/' + model_name + '/version_0/checkpoints/last.ckpt').cuda()
    model_postcovernet_test.eval()
    
    # Compute all the predictions on the test set.
    covernet_preds_test = []
    ground_truth_preds_test = []
    logits_test_agent = []
    logits_test_hd_map = []
    logits_test_social_context = []
    agent_test = []
    hd_map_test = []
    social_context_test = []
    agent_test_gt = []
    hd_map_test_gt = []
    social_context_test_gt = []
    logits_test = []
    model_postcovernet_test.cuda()
    model_postcovernet_test.eval()

    for center_agent_img, hd_map_img, social_context_img, agent_state_vector, combined_img, ground_truth in tq.tqdm(dataset.test):
        logits_agent, logits_hd_map, logits_social_context, agent_state_decode, hd_map_decode, social_context_decode = model_postcovernet_test.forward((_, _, _, agent_state_vector.cuda(), combined_img.cuda(), _), return_output='all_alpha', decode=True)
        logits_test_agent.append(logits_agent.cpu().data.numpy())
        logits_test_hd_map.append(logits_hd_map.cpu().data.numpy())
        logits_test_social_context.append(logits_social_context.cpu().data.numpy())
        logits = 1./3.*logits_social_context + 1./3.*logits_hd_map + 1./3.*logits_agent
        pred = trajectories[logits.argsort(descending=True, dim=-1)[:]].cuda()
        logits_test.append(logits.cpu().data.numpy())
        ground_truth_preds_test.append(ground_truth)
        covernet_preds_test.append(pred)

    # Save the ground truth and predictions
    pkl.dump(ground_truth_preds_test, open(model_name + "_gt_preds_test_last.pkl", "wb"))
    pkl.dump(logits_test_agent, open(model_name + "_logits_agent_test_last.pkl", "wb"))
    pkl.dump(logits_test_hd_map, open(model_name + "_logits_hd_map_test_last.pkl", "wb"))
    pkl.dump(logits_test_social_context, open(model_name + "_logits_social_context_test_last.pkl", "wb"))
    pkl.dump(logits_test, open(model_name + "_logits_test_last.pkl", "wb"))
    
    # Evaluate on the trajectory metrics.
    ground_truth_preds_test = pkl.load(open(model_name + "_gt_preds_test_last.pkl", "rb"))
    logits_test = pkl.load(open(model_name + "_logits_test_last.pkl", "rb"))
    covernet_preds_test = trajectories[torch.Tensor(logits_test).argsort(descending=True, dim=-1)[:]].cuda()

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
    print("ADE_1", ade_1)
    print("ADE_5", ade_5)
    print("ADE_10", ade_10)
    print("ADE_15", ade_15)
    print("FDE", fde)
    
    # Get trajectory classification labels and in-distribution uncertainty metrics.
    ground_truth_preds_test_tensor = torch.Tensor(ground_truth_preds_test).cuda()
    postcovernet_preds_test_tensor = covernet_preds_test.cuda()
    logit_test_tensor = torch.Tensor(logits_test).cuda()
    labels = []
    for i in range(ground_truth_preds_test_tensor.shape[0]):
        closest_trajectory = mean_pointwise_l2_distance(trajectories, ground_truth_preds_test_tensor[i])
        label = torch.LongTensor([closest_trajectory]).cuda()
        labels.append(nn.functional.one_hot(label, num_classes=trajectories.shape[0]))
    labels = torch.stack(labels)
    
    print("Brier Score", brier_score(torch.argmax(labels[:,0,:],dim=-1), logit_test_tensor[:,0,:]))
    print("Aleatoric Confidence", confidence(torch.argmax(labels[:,0,:],dim=-1), logit_test_tensor[:,0,:], uncertainty_type='aleatoric'))
    print("Epistemic Confidence", confidence(torch.argmax(labels[:,0,:],dim=-1), logit_test_tensor[:,0,:], uncertainty_type='epistemic'))

    # Compute all the predictions on the out-of-distribution test set.
    covernet_preds_test = []
    ground_truth_preds_test = []
    logits_test_agent = []
    logits_test_hd_map = []
    logits_test_social_context = []
    logits_test = []
    model_postcovernet_test.cuda()
    model_postcovernet_test.eval()
    for center_agent_img, hd_map_img, social_context_img, agent_state_vector, combined_img, ground_truth in tq.tqdm(dataset_ood.test):
        logits_agent, logits_hd_map, logits_social_context, agent_state_decode, hd_map_decode, social_context_decode = model_postcovernet_test.forward((_, _, _, agent_state_vector.cuda(), combined_img.cuda(), _),return_output='all_alpha')
        logits_test_agent.append(logits_agent.cpu().data.numpy())
        logits_test_hd_map.append(logits_hd_map.cpu().data.numpy())
        logits_test_social_context.append(logits_social_context.cpu().data.numpy())
        logits = 1./3.*logits_social_context + 1./3.*logits_hd_map + 1./3.*logits_agent
        pred = trajectories[logits.argsort(descending=True, dim=-1)[:]].cuda()
        ground_truth_preds_test.append(ground_truth)
        covernet_preds_test.append(pred)
        logits_test.append(logits.cpu().data.numpy())

    # Save the ground truth and predictions.
    pkl.dump(ground_truth_preds_test, open(model_name + "_ground_truth_preds_oodtest_last.pkl", "wb"))
    pkl.dump(logits_test_agent, open(model_name + "_logits_agent_oodtest_last.pkl", "wb"))
    pkl.dump(logits_test_hd_map, open(model_name + "_logits_hd_map_oodtest_last.pkl", "wb"))
    pkl.dump(logits_test_social_context, open(model_name + "_logits_social_context_oodtest_last.pkl", "wb"))
    pkl.dump(logits_test, open(model_name + "_logits_oodtest_last.pkl", "wb"))
    
    # Compute out-of-distribution uncertainty metrics.
    ground_truth_preds_test_in = pkl.load(open(model_name + "_ground_truth_preds_oodtest.pkl", "rb"))
    logits_test_in = pkl.load(open(model_name + "_logits_test_last.pkl", "rb"))
    logits_test_ood = pkl.load(open(model_name + "_logits_oodtest.pkl", "rb"))

    logit_test_in_tensor = torch.Tensor(logits_test_in).cuda()
    logit_test_ood_tensor = torch.Tensor(logits_test_ood).cuda()

    print("Aleatoric OOD APR", anomaly_detection(alpha=logit_test_in_tensor[:,0], ood_alpha=logit_test_ood_tensor[:,0], score_type='APR', uncertainty_type='aleatoric'))
    print("Epistemic OOD APR", anomaly_detection(alpha=logit_test_in_tensor[:,0], ood_alpha=logit_test_ood_tensor[:,0], score_type='APR', uncertainty_type='epistemic'))
    print("Aleatoric OOD AUROC", anomaly_detection(alpha=logit_test_in_tensor[:,0], ood_alpha=logit_test_ood_tensor[:,0], score_type='AUROC', uncertainty_type='aleatoric'))
    print("Epistemic OOD AUROC", anomaly_detection(alpha=logit_test_in_tensor[:,0], ood_alpha=logit_test_ood_tensor[:,0], score_type='AUROC', uncertainty_type='epistemic'))
    print("Average alpha in", torch.mean(torch.sum(logit_test_in_tensor[:,0], dim=-1)))
    print("Average alpha out", torch.mean(torch.sum(logit_test_ood_tensor[:,0], dim=-1)))
    print("Average alpha ratio", torch.mean(torch.sum(logit_test_ood_tensor[:,0], dim=-1))/torch.mean(torch.sum(logit_test_in_tensor[:,0], dim=-1)))
    print("% in", logit_test_in_tensor.shape[0]/(logit_test_ood_tensor.shape[0] + logit_test_in_tensor.shape[0]))
    print("% ood", logit_test_ood_tensor.shape[0]/(logit_test_ood_tensor.shape[0] + logit_test_in_tensor.shape[0]))

    # Load data
    logits_test_in = torch.Tensor(pkl.load(open(model_name + "_logits_test_last.pkl", "rb")))
    logits_test_ood = torch.Tensor(pkl.load(open(model_name + "_logits_oodtest_last.pkl", "rb")))

    logits_test_agent = torch.Tensor(pkl.load(open(model_name + "_logits_agent_test_last.pkl", "rb")))
    logits_test_hd_map = torch.Tensor(pkl.load(open(model_name + "_logits_hd_map_test_last.pkl", "rb")))
    logits_test_social_context = torch.Tensor(pkl.load(open(model_name + "_logits_social_context_test_last.pkl", "rb")))

    logits_test_ood_agent = torch.Tensor(pkl.load(open(model_name[0:125] + model_name[-50:] + "_logits_agent_oodtest_last.pkl", "rb")))
    logits_test_ood_hd_map = torch.Tensor(pkl.load(open(model_name[0:125] + model_name[-50:] + "_logits_hd_map_oodtest_last.pkl", "rb")))
    logits_test_ood_social_context = torch.Tensor(pkl.load(open(model_name[0:125] + model_name[-50:] + "_logits_social_context_oodtest_last.pkl", "rb")))

    distance = []
    distance_train = []
    distance_ood = []

    for token in tq.tqdm(dataset.test.predictions):
        instance_token, sample_token = token.split("_")
        history = dataset.test.helper.get_past_for_agent(instance_token, sample_token, seconds=1, in_agent_frame=True)
        distance.append(np.linalg.norm(history[-1]))

    for token in tq.tqdm(dataset.train.predictions):
        instance_token, sample_token = token.split("_")
        history = dataset.test.helper.get_past_for_agent(instance_token, sample_token, seconds=1, in_agent_frame=True)
        distance_train.append(np.linalg.norm(history[-1]))

    for token in tq.tqdm(dataset_ood.test.predictions):
        instance_token, sample_token = token.split("_")
        history = dataset.test.helper.get_past_for_agent(instance_token, sample_token, seconds=1, in_agent_frame=True)
        distance_ood.append(np.linalg.norm(history[-1]))

    distance = np.array(distance)
    distance_train = np.array(distance_train)
    distance_ood = np.array(distance_ood)

    plt.figure()
    plt.scatter(distance, torch.sum(logits_test_agent, dim=-1).cpu().data.numpy(), color='green', label='ID', alpha=0.25)
    plt.scatter(distance_ood, torch.sum(logits_test_ood_agent, dim=-1).cpu().data.numpy(), color='orange', label='OOD', alpha=0.25, marker='x')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.xlabel('Distance')
    plt.ylabel('Alpha_0')
    plt.ylim(0,70000)
    plt.legend(by_label.values(), by_label.keys(), loc='best', fancybox=True, edgecolor='black', frameon=True)
    plt.savefig('alpha_0_agent_vs_distance_x_o_radial_decode_double_latent_10_agent_last.png', dpi=600)
    tikzplotlib.save('alpha_0_agent_vs_distance_x_o_radial_decode_double_latent_10_agent_last.tex')
    plt.show()

    plt.figure()
    plt.scatter(distance, torch.sum(logits_test_hd_map, dim=-1).cpu().data.numpy(), color='green', label='ID', alpha=0.25)
    plt.scatter(distance_ood, torch.sum(logits_test_ood_hd_map, dim=-1).cpu().data.numpy(), color='orange', label='OOD', alpha=0.25, marker='x')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.xlabel('Distance')
    plt.ylabel('Alpha_0')
    plt.ylim(0,60000)
    plt.legend(by_label.values(), by_label.keys(), loc='best', fancybox=True, edgecolor='black', frameon=True)
    plt.savefig('alpha_0_hd_map_vs_distance_x_o_radial_decode_double_latent_10_agent_last.png', dpi=600)
    tikzplotlib.save('alpha_0_hd_map_vs_distance_x_o_radial_decode_double_latent_10_agent_last.tex')
    plt.show()

    plt.figure()
    plt.scatter(distance, torch.sum(logits_test_social_context, dim=-1).cpu().data.numpy(), color='green', label='ID', alpha=0.25)
    plt.scatter(distance_ood, torch.sum(logits_test_ood_social_context, dim=-1).cpu().data.numpy(), color='orange', label='OOD', alpha=0.25, marker='x')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.xlabel('Distance')
    plt.ylabel('Alpha_0')
    plt.ylim(0,15000)
    plt.legend(by_label.values(), by_label.keys(), loc='best', fancybox=True, edgecolor='black', frameon=True)
    plt.savefig('alpha_0_social_context_vs_distance_x_o_radial_decode_double_latent_10_agent_last.png', dpi=600)
    tikzplotlib.save('alpha_0_social_context_vs_distance_x_o_radial_decode_double_latent_10_agent_last.tex')
    plt.show()

    plt.figure()
    plt.hist(distance_train, color='green', alpha=1.0, bins=200, range=(0,25), label='ID')
    plt.hist(distance_ood, color='orange', alpha=1.0, bins=200, range=(0,25), label='OOD')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    plt.xlabel('Distance')
    plt.ylabel('Data Count')
    plt.legend(by_label.values(), by_label.keys(), loc='best', fancybox=True, edgecolor='black', frameon=True)
    plt.savefig('data_count_vs_distance_x_o_radial_separate_inputs_train_in_test.png', dpi=600)
    tikzplotlib.save('data_count_vs_distance_x_o_radial_separate_inputs_train_in_test.tex')
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    arg_parser = populate_common_params(parser)

    opts = arg_parser.parse_args()

    main(opts)