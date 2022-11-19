import os
import pickle as pkl
import numpy as np
import argparse
import tqdm.notebook as tq

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

from nuscenes.prediction.models.covernet import mean_pointwise_l2_distance

from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.combinators import Rasterizer
from nuscenes.eval.prediction.metrics import *
from input_represenation_decode import *
import pdb

class NuScenesDataset(Dataset):
    
    def __init__(self, dataroot, nuscenes, mode='train', seconds_of_history=1, dist=10., path_to_epsilon_set = "../../data/nuscenes/covernet_traj_set/nuscenes-prediction-challenge-trajectory-sets/epsilon_8.pkl"):
        self.dataroot = dataroot
        self.seconds_of_history = seconds_of_history
        self.nuscenes = nuscenes
            
        self.helper = PredictHelper(self.nuscenes)
        self.static_layer_rasterizer = StaticLayerRasterizer(self.helper)
        self.agent_rasterizer = AgentBoxesWithFadedHistory(self.helper, seconds_of_history=self.seconds_of_history)
        self.input_representation = InputRepresentation(self.static_layer_rasterizer, self.agent_rasterizer, Rasterizer())
        
        # Constant trajectory set
        self.trajectories = pkl.load(open(path_to_epsilon_set, 'rb'))

        # Saved them as a list of lists
        self.trajectories = torch.Tensor(self.trajectories)    
        
        check_train = 0
        check_val = 0
        check_test = 0
        
        if mode == 'train':
            self.predictions = get_prediction_challenge_split("train", dataroot=self.dataroot)
            
            # Compute class counts in training set
            self.N = torch.zeros(self.trajectories.shape[0])
            for i in tq.tqdm(reversed(range(len(self.predictions)))):
                instance_token, sample_token = self.predictions[i].split("_")
                history = self.helper.get_past_for_agent(instance_token, sample_token, seconds=self.seconds_of_history, in_agent_frame=True)[::-1]
                sample_record = nuscenes.get('sample', sample_token)
                scene_record = nuscenes.get('scene', sample_record['scene_token'])
                location = nuscenes.get('log', scene_record['log_token'])['location']
                if not ('big street' in scene_record['description'] or 'roundabout' in scene_record['description']) and (location == 'singapore-hollandvillage' or location == 'singapore-queenstown') and len(history) == self.seconds_of_history*2:
                    label = torch.unsqueeze(torch.Tensor(self.helper.get_future_for_agent(instance_token, sample_token, 6, in_agent_frame=True)), dim=0)
                    closest_lattice_trajectory_in_dist = mean_pointwise_l2_distance(self.trajectories, label)
                    label = torch.LongTensor([closest_lattice_trajectory_in_dist])
                    self.N[label] += 1
                else:
                    self.predictions.pop(i)
                
        elif mode == 'val':
            self.predictions = get_prediction_challenge_split("train_val", dataroot=self.dataroot)
            for i in tq.tqdm(reversed(range(len(self.predictions)))):
                instance_token, sample_token = self.predictions[i].split("_")
                history = self.helper.get_past_for_agent(instance_token, sample_token, seconds=self.seconds_of_history, in_agent_frame=True)[::-1]
                sample_record = nuscenes.get('sample', sample_token)
                scene_record = nuscenes.get('scene', sample_record['scene_token'])
                location = nuscenes.get('log', scene_record['log_token'])['location']
                if not ('big street' in scene_record['description'] or 'roundabout' in scene_record['description']) and (location == 'singapore-hollandvillage' or location == 'singapore-queenstown') and len(history) == self.seconds_of_history*2:
                    label = torch.unsqueeze(torch.Tensor(self.helper.get_future_for_agent(instance_token, sample_token, 6, in_agent_frame=True)), dim=0)
                    closest_lattice_trajectory_in_dist = mean_pointwise_l2_distance(self.trajectories, label)
                    label = torch.LongTensor([closest_lattice_trajectory_in_dist])
                else:
                    self.predictions.pop(i)
                    
        elif mode == 'test':
            self.predictions = get_prediction_challenge_split("val", dataroot=self.dataroot)
            for i in tq.tqdm(reversed(range(len(self.predictions)))):
                instance_token, sample_token = self.predictions[i].split("_")
                history = self.helper.get_past_for_agent(instance_token, sample_token, seconds=self.seconds_of_history, in_agent_frame=True)[::-1]
                sample_record = nuscenes.get('sample', sample_token)
                scene_record = nuscenes.get('scene', sample_record['scene_token'])
                location = nuscenes.get('log', scene_record['log_token'])['location']
                if not ('big street' in scene_record['description'] or 'roundabout' in scene_record['description']) and (location == 'singapore-hollandvillage' or location == 'singapore-queenstown') and len(history) == self.seconds_of_history*2:
                    label = torch.unsqueeze(torch.Tensor(self.helper.get_future_for_agent(instance_token, sample_token, 6, in_agent_frame=True)), dim=0)
                    closest_lattice_trajectory_in_dist = mean_pointwise_l2_distance(self.trajectories, label)
                    label = torch.LongTensor([closest_lattice_trajectory_in_dist])
                else:
                    self.predictions.pop(i)
                
        else:
            raise ValueError('Incorrect mode was provided. mode can be one of the following values: {train, val}')
                    
    def __len__(self):
        return len(self.predictions)
    
    def __getitem__(self, idx):
        # Get and load image
        instance_token, sample_token = self.predictions[idx].split("_")
        img = self.input_representation.make_input_representation(instance_token, sample_token)
        img = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)
      
        agent_state_vector = torch.Tensor([[self.helper.get_velocity_for_agent(instance_token, sample_token),
                                    self.helper.get_acceleration_for_agent(instance_token, sample_token),
                                    self.helper.get_heading_change_rate_for_agent(instance_token, sample_token)]])

        agent_state_vector[torch.isnan(agent_state_vector)] = -1
        label = self.helper.get_future_for_agent(instance_token, sample_token, 6, in_agent_frame=True)
        label = np.expand_dims(label, 0)
        
        return img, agent_state_vector, label
    
class NuScenesDatasetOOD(Dataset):
    
    def __init__(self, dataroot, nuscenes, mode='train', seconds_of_history=1, dist=10., path_to_epsilon_set = "../../data/nuscenes/covernet_traj_set/nuscenes-prediction-challenge-trajectory-sets/epsilon_8.pkl"):
        self.dataroot = dataroot
        self.seconds_of_history = seconds_of_history
        self.nuscenes = nuscenes
            
        self.helper = PredictHelper(self.nuscenes)
        self.static_layer_rasterizer = StaticLayerRasterizer(self.helper)
        self.agent_rasterizer = AgentBoxesWithFadedHistory(self.helper, seconds_of_history=self.seconds_of_history)
        self.input_representation = InputRepresentation(self.static_layer_rasterizer, self.agent_rasterizer, Rasterizer())
        
        # Constant trajectory set
        self.trajectories = pkl.load(open(path_to_epsilon_set, 'rb'))

        # Saved them as a list of lists
        self.trajectories = torch.Tensor(self.trajectories)
        self.predictions_train = get_prediction_challenge_split("train", dataroot=self.dataroot)
        self.predictions_add_val = self.predictions_train[0:int(len(self.predictions_train)/2)]
        self.predictions_add_test = self.predictions_train[int(len(self.predictions_train)/2):]
        
        if mode == 'val':
            self.predictions = get_prediction_challenge_split("train_val", dataroot=self.dataroot)
            self.predictions += self.predictions_add_val
            for i in tq.tqdm(reversed(range(len(self.predictions)))):
                instance_token, sample_token = self.predictions[i].split("_")
                history = self.helper.get_past_for_agent(instance_token, sample_token, seconds=self.seconds_of_history, in_agent_frame=True)[::-1]
                sample_record = nuscenes.get('sample', sample_token)
                scene_record = nuscenes.get('scene', sample_record['scene_token'])
                location = nuscenes.get('log', scene_record['log_token'])['location']
                if ('roundabout' in scene_record['description'].lower()) and (location[:3] == 'bos') and len(history) == self.seconds_of_history*2:
                    label = torch.unsqueeze(torch.Tensor(self.helper.get_future_for_agent(instance_token, sample_token, 6, in_agent_frame=True)), dim=0)
                    closest_lattice_trajectory_ood_dist = mean_pointwise_l2_distance(self.trajectories, label)
                    label = torch.LongTensor([closest_lattice_trajectory_ood_dist])
                else:
                    self.predictions.pop(i)

        elif mode == 'test':
            self.predictions = get_prediction_challenge_split("val", dataroot=self.dataroot)
            self.predictions += self.predictions_add_test
            for i in tq.tqdm(reversed(range(len(self.predictions)))):
                instance_token, sample_token = self.predictions[i].split("_")
                history = self.helper.get_past_for_agent(instance_token, sample_token, seconds=self.seconds_of_history, in_agent_frame=True)[::-1]
                sample_record = nuscenes.get('sample', sample_token)
                scene_record = nuscenes.get('scene', sample_record['scene_token'])
                location = nuscenes.get('log', scene_record['log_token'])['location']
                if ('roundabout' in scene_record['description'].lower()) and (location[:3] == 'bos') and len(history) == self.seconds_of_history*2:
                    label = torch.unsqueeze(torch.Tensor(self.helper.get_future_for_agent(instance_token, sample_token, 6, in_agent_frame=True)), dim=0)
                    closest_lattice_trajectory_ood_dist = mean_pointwise_l2_distance(self.trajectories, label)
                    label = torch.LongTensor([closest_lattice_trajectory_ood_dist])
                else:
                    self.predictions.pop(i)
        else:
            raise ValueError('Incorrect mode was provided. mode can be one of the following values: {train, val}')
                    
    def __len__(self):
        return len(self.predictions)
    
    def __getitem__(self, idx):
        # Get and load image
        instance_token, sample_token = self.predictions[idx].split("_")
        img = self.input_representation.make_input_representation(instance_token, sample_token)
        img = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0)
      
        agent_state_vector = torch.Tensor([[self.helper.get_velocity_for_agent(instance_token, sample_token),
                                    self.helper.get_acceleration_for_agent(instance_token, sample_token),
                                    self.helper.get_heading_change_rate_for_agent(instance_token, sample_token)]])

        agent_state_vector[torch.isnan(agent_state_vector)] = -1
        label = self.helper.get_future_for_agent(instance_token, sample_token, 6, in_agent_frame=True)
        label = np.expand_dims(label, 0)
        
        return img, agent_state_vector, label
    
class NuScenesDataModule(pl.LightningDataModule):

      def __init__(self, dataroot, nuscenes, batch_size=32, seconds_of_history=1, dist=10.0, ood=False):
          super().__init__()
          self.batch_size = batch_size
          self.dataroot = dataroot
          self.nuscenes = nuscenes
          self.seconds_of_history = seconds_of_history
          self.dist = dist
          self.ood = ood

      # OPTIONAL, called for every GPU/machine (assigning state is OK)
      def setup(self, stage = None):
          
          # split dataset
          if stage == 'fit':
                 if not self.ood:
                     self.train = NuScenesDataset(self.dataroot, self.nuscenes, mode='train', seconds_of_history=self.seconds_of_history, dist=self.dist)
                 print(len(self.train))
          elif stage == 'val':
                 if not self.ood:
                     self.val = NuScenesDataset(self.dataroot, self.nuscenes, mode='val', seconds_of_history=self.seconds_of_history, dist=self.dist) 
                 else:
                     self.val = NuScenesDatasetOOD(self.dataroot, self.nuscenes, mode='val', seconds_of_history=self.seconds_of_history, dist=self.dist) 
                 print(len(self.val))
          if stage == 'test':
                 if not self.ood:
                    self.test = NuScenesDataset(self.dataroot, self.nuscenes, mode='test', seconds_of_history=self.seconds_of_history, dist=self.dist) 
                 else:
                    self.test = NuScenesDatasetOOD(self.dataroot, self.nuscenes, mode='test', seconds_of_history=self.seconds_of_history, dist=self.dist) 
                 print(len(self.test))

      # return the dataloader for each split
      def train_dataloader(self):
          train = DataLoader(self.train, batch_size=self.batch_size, num_workers=16, shuffle=True, pin_memory=True)
          return train

      def val_dataloader(self):
          val = DataLoader(self.val, batch_size=self.batch_size, num_workers=16, shuffle=False, pin_memory=True)
          return val

      def test_dataloader(self):
          test = DataLoader(self.test, batch_size=self.batch_size, num_workers=16, shuffle=False, pin_memory=True)
          return test
