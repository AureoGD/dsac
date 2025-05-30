# dsac/sac_cas/networks.py
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import gymnasium as gym

from ..core.base_networks import init_layer, MLP, BaseEncoder, SimpleMLPEncoder, IdentityEncoder
# from .cnn_lstm_head import FeatureHead


class FeatureEncoderForContinuousSAC(BaseEncoder):

    def __init__(self,
                 obs_shapes_or_space,
                 use_encoder_for_dict_obs=True,
                 encoder_mlp_hidden_dims_for_flat=[64]):  # Added param for MLP encoder
        super().__init__()
        self.use_encoder_for_dict_obs = use_encoder_for_dict_obs
        self._is_dict_obs_type = isinstance(obs_shapes_or_space, (dict, gym.spaces.Dict))
        self.internal_encoder_module = None  # Will hold the actual encoder (Heads or SimpleMLP/Identity)

        if self.use_encoder_for_dict_obs and self._is_dict_obs_type:
            current_obs_shapes_dict = {}
            if isinstance(obs_shapes_or_space, gym.spaces.Dict):
                for key, space in obs_shapes_or_space.spaces.items():
                    if not isinstance(space, gym.spaces.Box):
                        raise ValueError(
                            f"Encoder input for key '{key}' must be Box for gym.spaces.Dict, got {type(space)}")
                    current_obs_shapes_dict[key] = space.shape
            elif isinstance(obs_shapes_or_space, dict):
                current_obs_shapes_dict = obs_shapes_or_space
            else:
                raise ValueError(
                    "obs_shapes_or_space must be a dict or gym.spaces.Dict if use_encoder_for_dict is True")

            self.heads = nn.ModuleDict()
            total_out_dim = 0
            self.heads_obs_shapes_info = {}

            for key, shape in current_obs_shapes_dict.items():
                if not shape or np.prod(shape) == 0:
                    self.heads_obs_shapes_info[key] = {
                        "channels": 0,
                        "time_steps": 0,
                        "original_shape": shape,
                        "is_empty": True
                    }
                    continue
                self.heads_obs_shapes_info[key] = {"original_shape": shape, "is_empty": False}
                if len(shape) == 1: channels, time_steps = 1, shape[0]
                elif len(shape) == 2: channels, time_steps = shape[0], shape[1]
                elif len(shape) == 3: channels, time_steps = shape[0], np.prod(shape[1:])
                else: raise ValueError(f"Unsupported shape {shape} for key '{key}'.")
                self.heads_obs_shapes_info[key]["channels"] = channels
                self.heads_obs_shapes_info[key]["time_steps"] = time_steps
                if channels > 0 and time_steps > 0:
                    head = FeatureHead(channels=channels, time_steps=time_steps, use_lstm=False)
                    self.heads[key] = head
                    total_out_dim += head.out_dim
            self._output_dim = total_out_dim
            self.internal_encoder_module = self.heads  # Use ModuleDict as the encoder
        else:
            obs_dim_for_flat_encoder = 0
            if isinstance(obs_shapes_or_space, gym.spaces.Box):
                obs_dim_for_flat_encoder = obs_shapes_or_space.shape[0]
            elif isinstance(obs_shapes_or_space, tuple):
                obs_dim_for_flat_encoder = obs_shapes_or_space[0]
            else:
                raise ValueError(f"Unsupported obs_shapes_or_space type for flat encoder: {type(obs_shapes_or_space)}")

            if not encoder_mlp_hidden_dims_for_flat:  # No hidden layers, use Identity
                self.internal_encoder_module = IdentityEncoder(obs_dim=obs_dim_for_flat_encoder)
            else:
                self.internal_encoder_module = SimpleMLPEncoder(obs_dim=obs_dim_for_flat_encoder,
                                                                hidden_dims_encoder=encoder_mlp_hidden_dims_for_flat)
            self._output_dim = self.internal_encoder_module.output_dim

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, obs):
        if self.use_encoder_for_dict_obs and self._is_dict_obs_type:
            encoded_parts = []
            active_device = None
            for key_obs in obs.values():
                if isinstance(key_obs, T.Tensor):
                    active_device = key_obs.device
                    break
            if active_device is None: active_device = "cpu"

            for key, head_module in self.internal_encoder_module.items():  # Iterate through ModuleDict (self.heads)
                if self.heads_obs_shapes_info[key]["is_empty"]: continue
                head_input = obs[key]
                shape_info = self.heads_obs_shapes_info[key]
                original_shape_len = len(shape_info["original_shape"])
                if original_shape_len == 1 and head_input.ndim == 2: head_input = head_input.unsqueeze(1)
                elif original_shape_len == 3 and head_input.ndim == 4:
                    batch_size = head_input.shape[0]
                    head_input = head_input.reshape(batch_size, shape_info["channels"], -1)
                encoded_parts.append(head_module(head_input))

            if not encoded_parts:
                return T.empty(obs[list(obs.keys())[0]].shape[0] if obs and obs.keys() else 1, 0, device=active_device)
            return T.cat(encoded_parts, dim=1)
        else:
            return self.internal_encoder_module(obs)  # Use SimpleMLPEncoder or IdentityEncoder


class CriticNetworkContinuous(nn.Module):

    def __init__(self,
                 critic_lr,
                 n_actions,
                 encoder: BaseEncoder,
                 hidden_dims_body=[256, 256],
                 name="critic_continuous",
                 chkpt_dir="tmp/sac_cas"):
        super().__init__()
        self.chkpt_dir = chkpt_dir
        self.name = name
        os.makedirs(self.chkpt_dir, exist_ok=True)
        self.checkpoint_file_base = os.path.join(self.chkpt_dir, self.name + "_sac_cas")
        self.encoder = encoder
        input_dim_to_body = self.encoder.output_dim
        self.q_network = MLP(input_dim=input_dim_to_body + n_actions, output_dim=1, hidden_dims=hidden_dims_body)
        self.optimizer = T.optim.Adam(self.parameters(), lr=critic_lr)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state, action):
        encoded_state = self.encoder(state)
        if action.ndim == 1 and encoded_state.ndim == 2:
            action = action.unsqueeze(0) if encoded_state.shape[0] == 1 else action.repeat(encoded_state.shape[0], 1)
        elif action.ndim == 2 and encoded_state.ndim == 2 and action.shape[0] == 1 and encoded_state.shape[0] > 1:
            action = action.repeat(encoded_state.shape[0], 1)
        x = T.cat([encoded_state, action], dim=1)
        return self.q_network(x)

    def save_checkpoint(self, suffix=""):
        T.save(self.state_dict(), self.checkpoint_file_base + suffix + ".pth")

    def load_checkpoint(self, suffix=""):
        self.load_state_dict(T.load(self.checkpoint_file_base + suffix + ".pth", map_location=self.device))


class ActorNetworkContinuous(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(self,
                 actor_lr,
                 n_actions,
                 max_action_scalar,
                 encoder: BaseEncoder,
                 hidden_dims_body=[256, 256],
                 name="actor_continuous",
                 chkpt_dir="tmp/sac_cas"):
        super().__init__()
        self.chkpt_dir = chkpt_dir
        self.name = name
        os.makedirs(self.chkpt_dir, exist_ok=True)
        self.checkpoint_file_base = os.path.join(self.chkpt_dir, self.name + "_sac_cas")
        self.encoder = encoder
        input_dim_to_body = self.encoder.output_dim

        if input_dim_to_body == 0:
            self.actor_body = nn.Identity()
            final_body_dim = 0
        elif hidden_dims_body:
            self.actor_body = MLP(input_dim=input_dim_to_body,
                                  output_dim=hidden_dims_body[-1],
                                  hidden_dims=hidden_dims_body[:-1],
                                  activation=nn.ReLU)
            final_body_dim = hidden_dims_body[-1]
        else:
            self.actor_body = nn.Identity()
            final_body_dim = input_dim_to_body

        self.mu_layer = init_layer(nn.Linear(max(1, final_body_dim), n_actions), std=0.01)
        self.log_std_layer = init_layer(nn.Linear(max(1, final_body_dim), n_actions), std=0.01)
        self.max_action_val = float(max_action_scalar)
        self.reparam_noise = 1e-6
        self.optimizer = T.optim.Adam(self.parameters(), lr=actor_lr)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        features = self.encoder(state)
        if features.ndim > 0 and features.shape[
                1] == 0 and self.encoder.output_dim != 0:  # Check if features is not an empty tensor with no second dimension
            if self.mu_layer.in_features > 0:
                batch_size = state[list(state.keys())[0]].shape[0] if isinstance(state, dict) else state.shape[0]
                features = T.zeros((batch_size, self.mu_layer.in_features), device=self.device)
        body_output = self.actor_body(features)
        mu = self.mu_layer(body_output)
        log_std = self.log_std_layer(body_output)
        log_std = T.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        sigma = T.exp(log_std)
        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)
        actions_gaussian = probabilities.rsample() if reparameterize else probabilities.sample()
        actions_tanh = T.tanh(actions_gaussian)
        log_probs_gaussian = probabilities.log_prob(actions_gaussian)
        log_probs_tanh = log_probs_gaussian - T.log(1.0 - actions_tanh.pow(2) + self.reparam_noise)
        if log_probs_tanh.ndim > 1 and log_probs_tanh.shape[1] > 0:
            log_probs_tanh = log_probs_tanh.sum(dim=1, keepdim=True)
        elif log_probs_tanh.ndim == 1:
            log_probs_tanh = log_probs_tanh.unsqueeze(1)
        scaled_tanh_action = actions_tanh * self.max_action_val
        return scaled_tanh_action, log_probs_tanh

    def save_checkpoint(self, suffix=""):
        T.save(self.state_dict(), self.checkpoint_file_base + suffix + ".pth")

    def load_checkpoint(self, suffix=""):
        self.load_state_dict(T.load(self.checkpoint_file_base + suffix + ".pth", map_location=self.device))
