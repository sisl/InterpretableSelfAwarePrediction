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
from nuscenes.prediction.models.covernet import CoverNet, ConstantLatticeLoss
from nuscenes.eval.prediction.metrics import *

from torchvision import models, transforms
import torchvision

class PredictionModel(pl.LightningModule):

    def __init__(self, num_modes=64, lr=1e-4, batch_size=16, optimizer='sgd', weight_decay=1e-2, backbone_name='resnet50', n_hidden_layers=None, path_to_epsilon_set = "nuscenes/covernet_traj_set/nuscenes-prediction-challenge-trajectory-sets/epsilon_8.pkl"):
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
        self.covernet = CoverNet(backbone, num_modes=self.num_modes)
        
        # Constant trajectory set.
        self.trajectories = pkl.load(open(path_to_epsilon_set, 'rb'))
        
        # Saved them as a list of lists.
        self.trajectories = torch.Tensor(self.trajectories)
        self.loss = ConstantLatticeLoss(self.trajectories)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        image_tensor, agent_state_vector = x
        agent_state_vector = torch.squeeze(agent_state_vector, 1)
        image_tensor = torch.squeeze(image_tensor, 1)
        logits = self.covernet(image_tensor, agent_state_vector)
        return logits

    def training_step(self, batch, batch_idx):
        image_tensor, agent_state_vector, labels = batch
        agent_state_vector = torch.squeeze(agent_state_vector, 1)
        image_tensor = torch.squeeze(image_tensor, 1)
        
        logits = self.covernet(image_tensor, agent_state_vector)
        loss = self.loss(logits, labels)
    
        # Logging to TensorBoard by default
        self.log('train_loss', loss)

        pred = self.trajectories[logits.argsort(descending=True, dim=-1)[:]]
        pred = pred.cuda()
        ade = torch.mean(torch.linalg.norm(pred[:,0] - labels[:,0], dim=-1))
        self.log('train_ade', ade)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        image_tensor, agent_state_vector, labels = batch
        agent_state_vector = torch.squeeze(agent_state_vector, 1)
        image_tensor = torch.squeeze(image_tensor, 1)
        
        logits = self.covernet(image_tensor, agent_state_vector)
        loss = self.loss(logits, labels)

        # Logging to TensorBoard by default
        self.log('val_loss', loss)
        
        pred = self.trajectories[logits.argsort(descending=True, dim=-1)[:]]
        pred = pred.cuda()
        ade = torch.mean(torch.linalg.norm(pred[:,0] - labels[:,0], dim=-1))
        self.log('val_ade', ade)

        return loss

    def test_step(self, batch, batch_idx):
        image_tensor, agent_state_vector, labels = batch
        agent_state_vector = torch.squeeze(agent_state_vector, 1)
        image_tensor = torch.squeeze(image_tensor, 1)
        
        logits = self.covernet(image_tensor, agent_state_vector)
        loss = self.loss(logits, labels)

        # Logging to TensorBoard by default
        self.log('test_loss', loss)
        
        pred = self.trajectories[logits.argsort(descending=True, dim=-1)[:]]
        pred = pred.cuda()
        ade = torch.mean(torch.linalg.norm(pred[:,0] - labels[:,0], dim=-1))
        self.log('test_ade', ade)
        
        return loss

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=self.weight_decay)
        return optimizer
