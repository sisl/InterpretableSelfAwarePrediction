import os
import pickle as pkl
import numpy as np
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from nuscenes import NuScenes
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction import PredictHelper
from nuscenes.prediction.helper import *

from nuscenes.prediction.models.backbone import ResNetBackbone
from post_covernet_decode_double_latent import PostCoverNet, UCELoss, ConstantLatticeLoss
from nuscenes.eval.prediction.metrics import *

from torchvision import models, transforms
import torchvision

import pdb

class PredictionModel(pl.LightningModule):

    def __init__(self, N, num_modes=64, lr=1e-4, batch_size=16, optimizer='sgd', weight_decay=1e-2, n_density=8, backbone_name='resnet50', n_hidden_layers=None, path_to_epsilon_set = "../../data/nuscenes/covernet_traj_set/nuscenes-prediction-challenge-trajectory-sets/epsilon_8.pkl"):
        super().__init__()
        self.save_hyperparameters()
        self.num_modes = num_modes
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        if backbone_name == 'resnet50':
            backbone = ResNetBackbone('resnet50')
        elif backbone_name == 'resnet18':
            backbone = ResNetBackbone('resnet18')
        elif backbone_name == 'squeezenet':
            backbone = models.squeezenet1_1(pretrained=False)
        elif backbone_name == 'shufflenet':
            backbone = models.shufflenet_v2_x0_5(pretrained=False)
        elif backbone_name == 'mobilenet':
            backbone = models.mobilenet_v2(pretrained=False)
        elif backbone_name == 'mnasnet':
            backbone = models.mnasnet0_5(pretrained=False)
        
        # Note that the value of num_modes depends on the size of the lattice used for CoverNet.
        self.postcovernet = PostCoverNet(backbone, num_modes=self.num_modes, N=N, n_density=n_density, n_hidden_layers=n_hidden_layers)
        
        # Constant trajectory set
        self.trajectories = pkl.load(open(path_to_epsilon_set, 'rb'))
        
        # Saved them as a list of lists
        self.trajectories = torch.Tensor(self.trajectories)
        self.loss = UCELoss(self.trajectories)
        self.val_step = 0

    def forward(self, x, return_output='alpha', decode=True):
        _, _, _, agent_state_vector, combined_tensor, _ = x
        agent_state_vector = torch.squeeze(agent_state_vector, 1)
        combined_tensor = torch.squeeze(combined_tensor, 1)
        output = self.postcovernet(combined_tensor, agent_state_vector[:,0:3], return_output=return_output, decode=decode)
        return output

    def training_step(self, batch, batch_idx):
        agent_tensor_gt, hd_map_tensor_gt, social_context_tensor_gt, agent_state_vector_gt, combined_tensor, labels = batch
        agent_state_vector_gt = torch.squeeze(agent_state_vector_gt, 1)
        hd_map_tensor_gt = torch.squeeze(hd_map_tensor_gt, 1)/255.
        social_context_tensor_gt = torch.squeeze(social_context_tensor_gt, 1)/255.
        combined_tensor = torch.squeeze(combined_tensor, 1)
        logits, agent_state_vector, hd_map_tensor, social_context_tensor = self.postcovernet(combined_tensor, agent_state_vector_gt[:,0:3], decode=True)
        loss_total, loss, loss_agent, loss_hd, loss_social = self.loss(logits, labels, agent_state_vector, agent_state_vector_gt, hd_map_tensor, hd_map_tensor_gt, social_context_tensor, social_context_tensor_gt)
    
        # Logging to TensorBoard by default
        self.log('train_alpha_0', torch.max(torch.sum(logits, dim=-1)))
        self.log('train_loss', loss)
        self.log('train_loss_total', loss_total)
        self.log('train_loss_hd', loss_hd)
        self.log('train_loss_social', loss_social)
        self.log('train_loss_agent', loss_agent)
        pred = self.trajectories[logits.argsort(descending=True, dim=-1)[:]]
        pred = pred.cuda()
        ade = torch.mean(torch.linalg.norm(pred[:,0] - labels[:,0], dim=-1))
        self.log('train_ade', ade)
        
        return loss_total
    
    def validation_step(self, batch, batch_idx):
        agent_tensor_gt, hd_map_tensor_gt, social_context_tensor_gt, agent_state_vector_gt, combined_tensor, labels = batch
        agent_state_vector_gt = torch.squeeze(agent_state_vector_gt, 1)
        hd_map_tensor_gt = torch.squeeze(hd_map_tensor_gt, 1)/255.
        social_context_tensor_gt = torch.squeeze(social_context_tensor_gt, 1)/255.
        combined_tensor = torch.squeeze(combined_tensor, 1)
        logits_agent, logits_hd_map, logits_social_context, agent_state_vector, hd_map_tensor, social_context_tensor = self.postcovernet(combined_tensor, agent_state_vector_gt[:,0:3], return_output='all_alpha', decode=True)
        logits = 1./3.*logits_agent + 1./3.*logits_hd_map + 1./3.*logits_social_context
        loss_total, loss, loss_agent, loss_hd, loss_social = self.loss(logits, labels, agent_state_vector, agent_state_vector_gt, hd_map_tensor, hd_map_tensor_gt, social_context_tensor, social_context_tensor_gt)
    
        # Logging to TensorBoard by default
        self.log('val_alpha_0', torch.max(torch.sum(logits, dim=-1)))
        self.log('val_alpha_0_agent', torch.max(torch.sum(logits_agent, dim=-1)))
        self.log('val_alpha_0_hd_map', torch.max(torch.sum(logits_hd_map, dim=-1)))
        self.log('val_alpha_0_social_context', torch.max(torch.sum(logits_social_context, dim=-1)))
        self.log('val_loss', loss)
        self.log('val_loss_total', loss_total)
        self.log('val_loss_hd', loss_hd)
        self.log('val_loss_social', loss_social)
        self.log('val_loss_agent', loss_agent)
        
        if self.val_step % 10 == 0:
            tensorboard = self.logger.experiment

            grid = torchvision.utils.make_grid(hd_map_tensor,padding=0)
            tensorboard.add_image('val_images/hd_map', grid, self.val_step)
            grid = torchvision.utils.make_grid(social_context_tensor,padding=0)
            tensorboard.add_image('val_images/social_context', grid, self.val_step)
            
            grid = torchvision.utils.make_grid(hd_map_tensor_gt,padding=0)
            tensorboard.add_image('val_images/hd_map_gt', grid, self.val_step)
            grid = torchvision.utils.make_grid(social_context_tensor_gt,padding=0)
            tensorboard.add_image('val_images/social_context_gt', grid, self.val_step)
        
        self.val_step += 1

        pred = self.trajectories[logits.argsort(descending=True, dim=-1)[:]]
        pred = pred.cuda()
        ade = torch.mean(torch.linalg.norm(pred[:,0] - labels[:,0], dim=-1))
        self.log('val_ade', ade)
        
        return loss_total

    def test_step(self, batch, batch_idx):
        agent_tensor_gt, hd_map_tensor_gt, social_context_tensor_gt, agent_state_vector_gt, combined_tensor, labels = batch
        agent_state_vector_gt = torch.squeeze(agent_state_vector_gt, 1)
        hd_map_tensor_gt = torch.squeeze(hd_map_tensor_gt, 1)/255.
        social_context_tensor_gt = torch.squeeze(social_context_tensor_gt, 1)/255.
        combined_tensor = torch.squeeze(combined_tensor, 1)
        logits, agent_state_vector, hd_map_tensor, social_context_tensor = self.postcovernet(combined_tensor, agent_state_vector_gt[:,0:3], decode=True)
        loss_total, loss, loss_agent, loss_hd, loss_social = self.loss(logits, labels, agent_state_vector, agent_state_vector_gt, hd_map_tensor, hd_map_tensor_gt, social_context_tensor, social_context_tensor_gt)
    
        # Logging to TensorBoard by default
        self.log('test_alpha_0', torch.max(torch.sum(logits, dim=-1)))
        self.log('test_loss', loss)
        self.log('test_loss_total', loss_total)
        self.log('test_loss_hd', loss_hd)
        self.log('test_loss_social', loss_social)
        self.log('test_loss_agent', loss_agent)
        
        tensorboard = self.logger.experiment
        
        grid = torchvision.utils.make_grid(hd_map_tensor,padding=0)
        tensorboard.add_image('test_images/hd_map', grid)
        grid = torchvision.utils.make_grid(social_context_tensor,padding=0)
        tensorboard.add_image('test_images/social_context', grid)     

        pred = self.trajectories[logits.argsort(descending=True, dim=-1)[:]]
        pred = pred.cuda()
        ade = torch.mean(torch.linalg.norm(pred[:,0] - labels[:,0], dim=-1))
        self.log('test_ade', ade)
        
        return loss_total

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=self.weight_decay)
        return optimizer