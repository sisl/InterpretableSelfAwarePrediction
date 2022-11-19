from typing import List, Tuple, Callable, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as f
from torch.distributions.dirichlet import Dirichlet

from nuscenes.prediction.models.backbone import calculate_backbone_feature_dim

# Modules from: https://github.com/sharpenb/Posterior-Network
from posterior_network.src.posterior_networks.NormalizingFlowDensity import NormalizingFlowDensity
from posterior_network.src.posterior_networks.BatchedNormalizingFlowDensity import BatchedNormalizingFlowDensity
from posterior_network.src.posterior_networks.MixtureDensity import MixtureDensity

# Number of entries in Agent State Vector
ASV_DIM = 3

class PostCoverNet(nn.Module):

    def __init__(self, backbone: nn.Module,
                 num_modes: int,
                 N: torch.Tensor, 
                 n_hidden_layers: List[int] = None,
                 latent_dim = 4,
                 input_shape: Tuple[int, int, int] = (3, 500, 500),
                 density_type='radial_flow',
                 budget_function='id',
                 n_density=8):
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

        backbone_feature_dim = calculate_backbone_feature_dim(backbone, input_shape)
        n_hidden_layers = [backbone_feature_dim + ASV_DIM] + n_hidden_layers + [latent_dim] # + [num_modes]
        
        print("Layer numbers: ", backbone_feature_dim, n_hidden_layers)

        linear_layers = [nn.Linear(in_dim, out_dim)
                         for in_dim, out_dim in zip(n_hidden_layers[:-1], n_hidden_layers[1:])]

        self.batch_norm = nn.BatchNorm1d(num_features=n_hidden_layers[-1])

        self.head = nn.ModuleList(linear_layers)

        # Normalizing Flow -- Normalized density on latent space
        if self.density_type == 'planar_flow':
            self.density_estimation = nn.ModuleList([NormalizingFlowDensity(dim=n_hidden_layers[-1], flow_length=n_density, flow_type=self.density_type) for c in range(self.num_modes)])
        elif self.density_type == 'radial_flow':
            self.density_estimation = nn.ModuleList([NormalizingFlowDensity(dim=n_hidden_layers[-1], flow_length=n_density, flow_type=self.density_type) for c in range(self.num_modes)])
        elif self.density_type == 'batched_radial_flow':
            self.density_estimation = BatchedNormalizingFlowDensity(c=self.num_modes, dim=n_hidden_layers[-1], flow_length=n_density, flow_type=self.density_type.replace('batched_', ''))
        elif self.density_type == 'iaf_flow':
            self.density_estimation = nn.ModuleList([NormalizingFlowDensity(dim=n_hidden_layers[-1], flow_length=n_density, flow_type=self.density_type) for c in range(self.num_modes)])
        elif self.density_type == 'normal_mixture':
            self.density_estimation = nn.ModuleList([MixtureDensity(dim=n_hidden_layers[-1], n_components=n_density, mixture_type=self.density_type) for c in range(self.num_modes)])
        else:
            raise NotImplementedError

    def forward(self, image_tensor: torch.Tensor, agent_state_vector: torch.Tensor, return_output='alpha', compute_loss=True, decode=True) -> torch.Tensor:
        """
        :param image_tensor: Tensor of images in the batch.
        :param agent_state_vector: Tensor of agent state vectors in the batch
        :return: Logits for the batch.
        """

        backbone_features = self.backbone(image_tensor)

        logits = torch.cat([backbone_features, agent_state_vector], dim=1)

        for linear in self.head:
            logits = linear(logits)

        batch_size = image_tensor.size(0)

        if self.N.device != image_tensor.device:
            self.N = self.N.to(image_tensor.device)

        if self.budget_function == 'parametrized':
            N = self.N / self.N.sum()
        else:
            N = self.N

        # # Forward
        zk = self.batch_norm(logits)
        log_q_zk = torch.zeros((batch_size, self.num_modes)).to(zk.device.type)
        alpha = torch.zeros((batch_size, self.num_modes)).to(zk.device.type)

        if isinstance(self.density_estimation, nn.ModuleList):
            for c in range(self.num_modes):
                log_p = self.density_estimation[c].log_prob(zk)
                log_q_zk[:, c] = log_p
                alpha[:, c] = 1. + (N[c] * torch.exp(log_q_zk[:, c]))
        else:
            log_q_zk = self.density_estimation.log_prob(zk)
            alpha = 1. + (N[:, None] * torch.exp(log_q_zk)).permute(1, 0)

        # print(torch.sum(alpha != 1))

        soft_output_pred = torch.nn.functional.normalize(alpha, p=1)
        output_pred = torch.max(soft_output_pred, dim=-1)[1]

        if return_output == 'hard':
            return output_pred
        elif return_output == 'soft':
            return soft_output_pred
        elif return_output == 'alpha':
            return alpha
        elif return_output == 'latent':
            return zk
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
                 regr: float = 1e-5):
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

    def __call__(self, batch_alpha: torch.Tensor, batch_ground_truth_trajectory: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss on a batch.
        :param batch_logits: Tensor of shape [batch_size, n_modes]. Output of a linear layer since this class
            uses nn.functional.cross_entropy.
        :param batch_ground_truth_trajectory: Tensor of shape [batch_size, 1, n_timesteps, state_dim]
        :return: Average element-wise loss on the batch.
        """

        # If using GPU, need to copy the lattice to the GPU if haven't done so already
        # This ensures we only copy it once
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

        return UCE_loss