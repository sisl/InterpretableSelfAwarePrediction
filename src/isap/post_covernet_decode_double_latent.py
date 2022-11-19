from typing import List, Tuple, Callable, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as f
from torch.distributions.dirichlet import Dirichlet

import os
import sys
sys.path.append("../../") # Add directory containing src/data to path

from nuscenes.prediction.models.backbone import calculate_backbone_feature_dim

# Modules from: https://github.com/sharpenb/Posterior-Network
from posterior_network.src.posterior_networks.NormalizingFlowDensity import NormalizingFlowDensity
from posterior_network.src.posterior_networks.BatchedNormalizingFlowDensity import BatchedNormalizingFlowDensity
from posterior_network.src.posterior_networks.MixtureDensity import MixtureDensity

# Number of entries in Agent State Vector
ASV_DIM = 7

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(False),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(False),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class PostCoverNet(nn.Module):

    def __init__(self, backbone: nn.Module,
                 num_modes: int,
                 N: torch.Tensor, 
                 n_hidden_layers: List[int] = None,
                 latent_dim = 4, # 4, # 6, # 10
                 input_shape: Tuple[int, int, int] = (3, 500, 500),
                 density_type= 'radial_flow', #'normal_mixture', #'radial_flow',
                 budget_function='id',
                 n_density=8,
                 seconds=1,
                 dim=4):
        """
        Inits Covernet.
        :param backbone: Backbone model. Typically ResNetBackBone or MobileNetBackbone
        :param num_modes: Number of modes in the lattice
        :param n_hidden_layers: List of dimensions in the fully connected layers after the backbones.
            If None, set to [4096]
        :param input_shape: Shape of image input. Used to determine the dimensionality of the feature
            vector after the CNN backbone.
        """

        if n_hidden_layers and not isinstance(n_hidden_layers, list):
            raise ValueError(f"Param n_hidden_layers must be a list. Received {type(n_hidden_layers)}")

        super().__init__()

        if not n_hidden_layers:
            n_hidden_layers = [4096]

        self.backbone = backbone
        self.num_modes = num_modes
        self.latent_dim = latent_dim
        self.density_type = density_type
        self.N = N
        self.budget_function = budget_function
        self.dim = dim

        backbone_feature_dim = calculate_backbone_feature_dim(backbone, input_shape)
        n_hidden_layers = [backbone_feature_dim + ASV_DIM - 4] + n_hidden_layers + [latent_dim] # + [num_modes]
        
        print("Layer numbers: ", backbone_feature_dim, n_hidden_layers)

        # Splitting layers
        linear_layers_agent = []
        for in_dim, out_dim in zip(n_hidden_layers[:-1], n_hidden_layers[1:]):
            linear_layers_agent.append(nn.Linear(in_dim, out_dim))
            if out_dim != n_hidden_layers[-1]:
                linear_layers_agent.append(nn.ReLU())

        self.head_agent = nn.ModuleList(linear_layers_agent)
        
        self.batch_norm_agent = nn.BatchNorm1d(num_features=n_hidden_layers[-1])
        
        linear_layers_hd_map = []
        for in_dim, out_dim in zip(n_hidden_layers[:-1], n_hidden_layers[1:]):
            linear_layers_hd_map.append(nn.Linear(in_dim, out_dim))
            if out_dim != n_hidden_layers[-1]:
                linear_layers_hd_map.append(nn.ReLU())
        
        self.head_hd_map = nn.ModuleList(linear_layers_hd_map)

        self.batch_norm_hd_map = nn.BatchNorm1d(num_features=n_hidden_layers[-1])
        
        linear_layers_social_context = []
        for in_dim, out_dim in zip(n_hidden_layers[:-1], n_hidden_layers[1:]):
            linear_layers_social_context.append(nn.Linear(in_dim, out_dim))
            if out_dim != n_hidden_layers[-1]:
                linear_layers_social_context.append(nn.ReLU())

        self.head_social_context = nn.ModuleList(linear_layers_social_context)
        
        self.batch_norm_social_context = nn.BatchNorm1d(num_features=n_hidden_layers[-1])

        # Normalizing Flow -- Normalized density on latent space
        if self.density_type == 'planar_flow':
            self.density_estimation_agent = nn.ModuleList([NormalizingFlowDensity(dim=n_hidden_layers[-1], flow_length=n_density, flow_type=self.density_type) for c in range(self.num_modes)])
            self.density_estimation_hd_map = nn.ModuleList([NormalizingFlowDensity(dim=n_hidden_layers[-1], flow_length=n_density, flow_type=self.density_type) for c in range(self.num_modes)])
            self.density_estimation_social_context = nn.ModuleList([NormalizingFlowDensity(dim=n_hidden_layers[-1], flow_length=n_density, flow_type=self.density_type) for c in range(self.num_modes)])
        elif self.density_type == 'radial_flow':
            self.density_estimation_agent = nn.ModuleList([NormalizingFlowDensity(dim=n_hidden_layers[-1], flow_length=n_density, flow_type=self.density_type) for c in range(self.num_modes)])
            self.density_estimation_hd_map = nn.ModuleList([NormalizingFlowDensity(dim=n_hidden_layers[-1], flow_length=n_density, flow_type=self.density_type) for c in range(self.num_modes)])
            self.density_estimation_social_context = nn.ModuleList([NormalizingFlowDensity(dim=n_hidden_layers[-1], flow_length=n_density, flow_type=self.density_type) for c in range(self.num_modes)])
        elif self.density_type == 'batched_radial_flow':
            self.density_estimation_agent = BatchedNormalizingFlowDensity(c=self.num_modes, dim=n_hidden_layers[-1], flow_length=n_density, flow_type=self.density_type.replace('batched_', ''))
            self.density_estimation_hd_map = BatchedNormalizingFlowDensity(c=self.num_modes, dim=n_hidden_layers[-1], flow_length=n_density, flow_type=self.density_type.replace('batched_', ''))
            self.density_estimation_social_context = BatchedNormalizingFlowDensity(c=self.num_modes, dim=n_hidden_layers[-1], flow_length=n_density, flow_type=self.density_type.replace('batched_', ''))
        elif self.density_type == 'iaf_flow':
            self.density_estimation_agent = nn.ModuleList([NormalizingFlowDensity(dim=n_hidden_layers[-1], flow_length=n_density, flow_type=self.density_type) for c in range(self.num_modes)])
            self.density_estimation_hd_map = nn.ModuleList([NormalizingFlowDensity(dim=n_hidden_layers[-1], flow_length=n_density, flow_type=self.density_type) for c in range(self.num_modes)])
            self.density_estimation_social_context = nn.ModuleList([NormalizingFlowDensity(dim=n_hidden_layers[-1], flow_length=n_density, flow_type=self.density_type) for c in range(self.num_modes)])
        elif self.density_type == 'normal_mixture':
            self.density_estimation_agent = nn.ModuleList([MixtureDensity(dim=n_hidden_layers[-1], n_components=n_density, mixture_type=self.density_type) for c in range(self.num_modes)])
            self.density_estimation_hd_map = nn.ModuleList([MixtureDensity(dim=n_hidden_layers[-1], n_components=n_density, mixture_type=self.density_type) for c in range(self.num_modes)])
            self.density_estimation_social_context = nn.ModuleList([MixtureDensity(dim=n_hidden_layers[-1], n_components=n_density, mixture_type=self.density_type) for c in range(self.num_modes)])
        else:
            raise NotImplementedError
            
        # Decoding layers
        n_hidden_layers_agent_decode = [latent_dim] + [latent_dim] + [ASV_DIM]
        linear_layers_agent_decode = []
        for in_dim, out_dim in zip(n_hidden_layers_agent_decode[:-1], n_hidden_layers_agent_decode[1:]):
            linear_layers_agent_decode.append(nn.Linear(in_dim, out_dim))
            if out_dim != n_hidden_layers_agent_decode[-1]:
                linear_layers_agent_decode.append(nn.Tanh())

        self.decode_agent = nn.ModuleList(linear_layers_agent_decode)
        self.VQVAEBlock_hd_map = nn.Sequential(
            ResBlock(self.dim),
            ResBlock(self.dim),
            nn.ReLU(False),
            nn.ConvTranspose2d(self.dim, self.dim, 3, 2, 1),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(False),
            nn.ConvTranspose2d(self.dim, self.dim, 4, 2, 1),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(False),
            nn.ConvTranspose2d(self.dim, self.dim, 3, 2, 1),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(False),
            nn.ConvTranspose2d(self.dim, 3, 4, 2, 2),
            nn.Sigmoid()
        )
        
        self.VQVAEBlock_social_context = nn.Sequential(
            ResBlock(self.dim),
            ResBlock(self.dim),
            nn.ReLU(False),
            nn.ConvTranspose2d(self.dim, self.dim, 3, 2, 1),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(False),
            nn.ConvTranspose2d(self.dim, self.dim, 4, 2, 1),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(False),
            nn.ConvTranspose2d(self.dim, self.dim, 3, 2, 1),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(False),
            nn.ConvTranspose2d(self.dim, 3, 4, 2, 2),
            nn.Sigmoid()
        )

    def forward(self, image_tensor: torch.Tensor, agent_state_vector: torch.Tensor, return_output='alpha', compute_loss=True, decode=True) -> torch.Tensor:
        """
        :param image_tensor: Tensor of images in the batch.
        :param agent_state_vector: Tensor of agent state vectors in the batch
        :return: Logits for the batch.
        """

        backbone_features = self.backbone(image_tensor)
        
        features = torch.cat([backbone_features, agent_state_vector], dim=1)

        hd_map_features_latent = features.clone()
        social_context_features_latent = features.clone()
        agent_state_features = features.clone()
        
        for linear in self.head_hd_map[:-1]:
            hd_map_features_latent = linear(hd_map_features_latent)
            
        hd_map_features = self.head_hd_map[-1](hd_map_features_latent)
            
        for linear in self.head_social_context[:-1]:
            social_context_features_latent = linear(social_context_features_latent)
            
        social_context_features = self.head_social_context[-1](social_context_features_latent)
        
        for linear in self.head_agent:
            agent_state_features = linear(agent_state_features)

        batch_size = agent_state_features.size(0)

        if self.N.device != agent_state_features.device:
            self.N = self.N.to(agent_state_features.device)

        if self.budget_function == 'parametrized':
            N = self.N / self.N.sum()
        else:
            N = self.N

        # Forward agent
        zk_agent = self.batch_norm_agent(agent_state_features)
        log_q_zk_agent = torch.zeros((batch_size, self.num_modes)).to(zk_agent.device.type)
        alpha_agent = torch.zeros((batch_size, self.num_modes)).to(zk_agent.device.type)

        if isinstance(self.density_estimation_agent, nn.ModuleList):
            for c in range(self.num_modes):
                log_p_agent = self.density_estimation_agent[c].log_prob(zk_agent)
                log_q_zk_agent[:, c] = log_p_agent
                alpha_agent[:, c] = 1. + (N[c] * torch.exp(log_q_zk_agent[:, c]))
        else:
            log_q_zk_agent = self.density_estimation_agent.log_prob(zk_agent)
            alpha_agent = 1. + (N[:, None] * torch.exp(log_q_zk_agent)).permute(1, 0)

        soft_output_pred_agent = torch.nn.functional.normalize(alpha_agent, p=1)
        output_pred_agent = torch.max(soft_output_pred_agent, dim=-1)[1]
        
        # Forward hd map
        zk_hd_map = self.batch_norm_hd_map(hd_map_features)
        log_q_zk_hd_map = torch.zeros((batch_size, self.num_modes)).to(zk_hd_map.device.type)
        alpha_hd_map = torch.zeros((batch_size, self.num_modes)).to(zk_hd_map.device.type)

        if isinstance(self.density_estimation_hd_map, nn.ModuleList):
            for c in range(self.num_modes):
                log_p_hd_map = self.density_estimation_hd_map[c].log_prob(zk_hd_map)
                log_q_zk_hd_map[:, c] = log_p_hd_map
                alpha_hd_map[:, c] = 1. + (N[c] * torch.exp(log_q_zk_hd_map[:, c]))
        else:
            log_q_zk_hd_map = self.density_estimation_hd_map.log_prob(zk_hd_map)
            alpha_hd_map = 1. + (N[:, None] * torch.exp(log_q_zk_hd_map)).permute(1, 0)

        soft_output_pred_hd_map = torch.nn.functional.normalize(alpha_hd_map, p=1)
        output_pred_hd_map = torch.max(soft_output_pred_hd_map, dim=-1)[1]
        
        # Forward social context
        zk_social_context = self.batch_norm_social_context(social_context_features)
        log_q_zk_social_context = torch.zeros((batch_size, self.num_modes)).to(zk_social_context.device.type)
        alpha_social_context = torch.zeros((batch_size, self.num_modes)).to(zk_social_context.device.type)

        if isinstance(self.density_estimation_social_context, nn.ModuleList):
            for c in range(self.num_modes):
                log_p_social_context = self.density_estimation_social_context[c].log_prob(zk_social_context)
                log_q_zk_social_context[:, c] = log_p_social_context
                alpha_social_context[:, c] = 1. + (N[c] * torch.exp(log_q_zk_social_context[:, c]))
        else:
            log_q_zk_social_context = self.density_estimation.log_prob(zk_social_context)
            alpha_social_context = 1. + (N[:, None] * torch.exp(log_q_zk_social_context)).permute(1, 0)

        soft_output_pred_social_context = torch.nn.functional.normalize(alpha_social_context, p=1)
        output_pred_social_context = torch.max(soft_output_pred_social_context, dim=-1)[1]
        
        alpha = 1./3.*alpha_agent + 1./3.*alpha_hd_map + 1./3.*alpha_social_context
        soft_output_pred = torch.nn.functional.normalize(alpha, p=1)
        output_pred = torch.max(soft_output_pred, dim=-1)[1]
        
        # Decode the latent variables
        agent_state_decode = agent_state_features.clone()
        hd_map_decode = hd_map_features_latent.clone()
        social_context_decode = social_context_features_latent.clone()
        
        for linear in self.decode_agent:
            agent_state_decode = linear(agent_state_decode)
            
        hd_map_decode = hd_map_decode.view(-1, self.dim, 32, 32)
        social_context_decode = social_context_decode.view(-1, self.dim, 32, 32)
        
        hd_map_decode = self.VQVAEBlock_hd_map(hd_map_decode)
        social_context_decode = self.VQVAEBlock_social_context(social_context_decode)

        if decode:
            if return_output == 'hard':
                return output_pred, agent_state_decode, hd_map_decode, social_context_decode
            elif return_output == 'all_hard':
                return output_pred_agent, output_pred_hd_map, output_pred_social_context, agent_state_decode, hd_map_decode, social_context_decode
            elif return_output == 'soft':
                return soft_output_pred, agent_state_decode, hd_map_decode, social_context_decode
            elif return_output == 'all_soft':
                return soft_output_pred_agent, soft_output_pred_hd_map, soft_output_pred_social_context, agent_state_decode, hd_map_decode, social_context_decode
            elif return_output == 'alpha':
                return alpha, agent_state_decode, hd_map_decode, social_context_decode
            elif return_output == 'all_alpha':
                return alpha_agent, alpha_hd_map, alpha_social_context, agent_state_decode, hd_map_decode, social_context_decode
            elif return_output == 'latent':
                return zk_agent, zk_hd_map, zk_social_context, agent_state_decode, hd_map_decode, social_context_decode
            else:
                raise AssertionError
        else:
            if return_output == 'hard':
                return output_pred
            elif return_output == 'all_hard':
                return output_pred_agent, output_pred_hd_map, output_pred_social_context
            elif return_output == 'soft':
                return soft_output_pred
            elif return_output == 'all_soft':
                return soft_output_pred_agent, soft_output_pred_hd_map, soft_output_pred_social_context
            elif return_output == 'alpha':
                return alpha
            elif return_output == 'all_alpha':
                return alpha_agent, alpha_hd_map, alpha_social_context
            elif return_output == 'latent':
                return zk_agent, zk_hd_map, zk_social_context
            else:
                raise AssertionError

def mean_pointwise_l2_distance(lattice: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    """
    Computes the index of the closest trajectory in the lattice as measured by l1 distance.
    :param lattice: Lattice of pre-generated trajectories. Shape [num_modes, n_timesteps, state_dim]
    :param ground_truth: Ground truth trajectory of agent. Shape [1, n_timesteps, state_dim].
    :return: Index of closest mode in the lattice.
    """
    stacked_ground_truth = ground_truth.repeat(lattice.shape[0], 1, 1)
    return torch.pow(lattice - stacked_ground_truth, 2).sum(dim=2).sqrt().mean(dim=1).argmin()

# ConstantLatticeLoss from: https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/prediction/models/covernet.py.
class ConstantLatticeLoss:
    """
    Computes the loss for a constant lattice CoverNet model.
    """

    def __init__(self, lattice: Union[np.ndarray, torch.Tensor],
                 similarity_function: Callable[[torch.Tensor, torch.Tensor], int] = mean_pointwise_l2_distance):
        """
        Inits the loss.
        :param lattice: numpy array of shape [n_modes, n_timesteps, state_dim]
        :param similarity_function: Function that computes the index of the closest trajectory in the lattice
            to the actual ground truth trajectory of the agent.
        """

        self.lattice = torch.Tensor(lattice)
        self.similarity_func = similarity_function

    def __call__(self, batch_logits: torch.Tensor, batch_ground_truth_trajectory: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss on a batch.
        :param batch_logits: Tensor of shape [batch_size, n_modes]. Output of a linear layer since this class
            uses nn.functional.cross_entropy.
        :param batch_ground_truth_trajectory: Tensor of shape [batch_size, 1, n_timesteps, state_dim]
        :return: Average element-wise loss on the batch.
        """

        if self.lattice.device != batch_logits.device:
            self.lattice = self.lattice.to(batch_logits.device)

        batch_losses = torch.Tensor().requires_grad_(True).to(batch_logits.device)

        for logit, ground_truth in zip(batch_logits, batch_ground_truth_trajectory):

            closest_lattice_trajectory = self.similarity_func(self.lattice, ground_truth)
            label = torch.LongTensor([closest_lattice_trajectory]).to(batch_logits.device)
            classification_loss = f.cross_entropy(logit.unsqueeze(0), label)

            batch_losses = torch.cat((batch_losses, classification_loss.unsqueeze(0)), 0)

        return batch_losses.mean()

# UCELoss is adapted from: https://github.com/sharpenb/Posterior-Network to the interpetable paradigm. UCELoss is equivalent to the ELBO loss for our purposes since we use a uninformative prior.
class UCELoss:
    """
    Computes the loss for a constant lattice CoverNet model using UCE loss definition (PostNet).
    """

    def __init__(self, lattice: Union[np.ndarray, torch.Tensor],
                 similarity_function: Callable[[torch.Tensor, torch.Tensor], int] = mean_pointwise_l2_distance,
                 regr: float = 1e-5, agent_coeff: float = 1.0, hd_map_coeff: float = 1.0, social_context_coeff: float = 10.0):
        """
        Inits the loss.
        :param lattice: numpy array of shape [n_modes, n_timesteps, state_dim]
        :param similarity_function: Function that computes the index of the closest trajectory in the lattice
            to the actual ground truth trajectory of the agent.
        """

        self.lattice = torch.Tensor(lattice)
        self.similarity_func = similarity_function
        self.num_modes = self.lattice.shape[0]
        self.regr = regr
        self.agent_coeff = agent_coeff
        self.hd_map_coeff = hd_map_coeff
        self.social_context_coeff = social_context_coeff
        
        self.loss_agent = nn.MSELoss(reduction='sum')
        self.loss_hd_map = nn.MSELoss(reduction='sum')
        self.loss_social_context = nn.MSELoss(reduction='sum')

    def __call__(self, batch_alpha: torch.Tensor, batch_ground_truth_trajectory: torch.Tensor, batch_agent: torch.Tensor, batch_ground_truth_agent: torch.Tensor, batch_hd_map: torch.Tensor, batch_ground_truth_hd_map: torch.Tensor, batch_social_context: torch.Tensor, batch_ground_truth_social_context: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss on a batch.
        :param batch_logits: Tensor of shape [batch_size, n_modes]. Output of a linear layer since this class
            uses nn.functional.cross_entropy.
        :param batch_ground_truth_trajectory: Tensor of shape [batch_size, 1, n_timesteps, state_dim]
        :return: Average element-wise loss on the batch.
        """

        if self.lattice.device != batch_alpha.device:
            self.lattice = self.lattice.to(batch_alpha.device)

        batch_labels = torch.Tensor(batch_alpha.shape[0], self.num_modes).to(batch_alpha.device)

        for i in range(batch_ground_truth_trajectory.shape[0]):
            ground_truth = batch_ground_truth_trajectory[i]
            closest_lattice_trajectory = self.similarity_func(self.lattice, ground_truth)
            label = torch.LongTensor([closest_lattice_trajectory]).to(batch_alpha.device)

            batch_labels[i] = nn.functional.one_hot(label, num_classes=self.num_modes)

        alpha_0 = batch_alpha.sum(1).unsqueeze(-1).repeat(1, self.num_modes)
        entropy_reg = Dirichlet(batch_alpha).entropy()
        UCE_loss = (torch.sum(batch_labels * (torch.digamma(alpha_0) - torch.digamma(batch_alpha))) - self.regr * torch.sum(entropy_reg)).requires_grad_(True).to(batch_alpha.device)
        
        batch_agent[batch_ground_truth_agent == -1] = -1
        agent_loss = self.loss_agent(batch_agent, batch_ground_truth_agent)
        hd_map_loss = self.loss_hd_map(batch_hd_map, batch_ground_truth_hd_map)
        social_context_loss = self.loss_agent(batch_social_context, batch_ground_truth_social_context)
        
        total_loss = UCE_loss + self.agent_coeff * agent_loss + self.hd_map_coeff * hd_map_loss + self.social_context_coeff * social_context_loss

        return total_loss, UCE_loss, agent_loss, hd_map_loss, social_context_loss