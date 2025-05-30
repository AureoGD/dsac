# dsac/sac_das/networks.py
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..core.base_networks import init_layer, MLP, BaseEncoder


class ActorNetworkDiscrete(nn.Module):

    def __init__(self,
                 actor_lr: float,
                 n_discrete_actions: int,
                 encoder: BaseEncoder,
                 hidden_dims_body: list = [256, 256],
                 name: str = "actor_discrete",
                 chkpt_dir: str = "tmp/dsac_discrete"):
        super().__init__()
        self.chkpt_dir = chkpt_dir
        self.name = name
        os.makedirs(self.chkpt_dir, exist_ok=True)
        self.checkpoint_file_base = os.path.join(self.chkpt_dir, self.name + "_sac_discrete")

        self.encoder = encoder
        input_dim_to_body = self.encoder.output_dim

        if input_dim_to_body == 0 and obs_shapes_or_space != (0, ):  # Check against a potential dummy empty obs shape
            print(
                f"Warning: ActorNetworkDiscrete input from encoder is 0 for {name}. Actor output will be based on biases only."
            )
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

        self.action_logits_head = init_layer(nn.Linear(max(1, final_body_dim), n_discrete_actions), std=0.01)

        self.optimizer = T.optim.Adam(self.parameters(), lr=actor_lr)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, obs):
        features = self.encoder(obs)

        # Handle case where features might be empty from encoder
        if features.ndim > 0 and features.shape[1] == 0 and self.encoder.output_dim != 0:
            if self.action_logits_head.in_features > 0:
                # Attempt to get batch_size from the original observation dictionary if it's a dict
                # or from the flat tensor. This part assumes obs is the raw observation.
                if isinstance(obs, dict) and obs:
                    batch_size = obs[list(obs.keys())[0]].shape[0]
                elif isinstance(obs, T.Tensor):
                    batch_size = obs.shape[0]
                else:  # Fallback, though should not happen if obs is valid
                    batch_size = 1
                features = T.zeros((batch_size, self.action_logits_head.in_features), device=self.device)

        body_out = self.actor_body(features)
        action_logits = self.action_logits_head(body_out)
        return action_logits

    def save_checkpoint(self, suffix=""):
        filepath = self.checkpoint_file_base + suffix + ".pth"
        T.save(self.state_dict(), filepath)

    def load_checkpoint(self, suffix=""):
        filepath = self.checkpoint_file_base + suffix + ".pth"
        self.load_state_dict(T.load(filepath, map_location=self.device))


class CriticNetworkDiscrete(nn.Module):

    def __init__(self,
                 critic_lr: float,
                 n_discrete_actions: int,
                 encoder: BaseEncoder,
                 hidden_dims_body: list = [256, 256],
                 name: str = "critic_discrete",
                 chkpt_dir: str = "tmp/dsac_discrete"):
        super().__init__()
        self.chkpt_dir = chkpt_dir
        self.name = name
        os.makedirs(self.chkpt_dir, exist_ok=True)
        self.checkpoint_file_base = os.path.join(self.chkpt_dir, self.name + "_sac_discrete")

        self.encoder = encoder
        input_dim_to_body = self.encoder.output_dim

        if input_dim_to_body == 0 and obs_shapes_or_space != (0, ):
            print(
                f"Warning: CriticNetworkDiscrete input from encoder is 0 for {name}. Critic output will be based on biases only."
            )
            self.q_network_body = nn.Identity()
            final_body_dim = 0
        elif hidden_dims_body:
            self.q_network_body = MLP(input_dim=input_dim_to_body,
                                      output_dim=hidden_dims_body[-1],
                                      hidden_dims=hidden_dims_body[:-1],
                                      activation=nn.ReLU)
            final_body_dim = hidden_dims_body[-1]
        else:
            self.q_network_body = nn.Identity()
            final_body_dim = input_dim_to_body

        self.q_values_head = init_layer(nn.Linear(max(1, final_body_dim), n_discrete_actions), std=1.0)

        self.optimizer = T.optim.Adam(self.parameters(), lr=critic_lr)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, obs):
        features = self.encoder(obs)
        if features.ndim > 0 and features.shape[1] == 0 and self.encoder.output_dim != 0:
            if self.q_values_head.in_features > 0:
                if isinstance(obs, dict) and obs:
                    batch_size = obs[list(obs.keys())[0]].shape[0]
                elif isinstance(obs, T.Tensor):
                    batch_size = obs.shape[0]
                else:
                    batch_size = 1
                features = T.zeros((batch_size, self.q_values_head.in_features), device=self.device)
        body_out = self.q_network_body(features)
        q_all_actions = self.q_values_head(body_out)
        return q_all_actions

    def save_checkpoint(self, suffix=""):
        filepath = self.checkpoint_file_base + suffix + ".pth"
        T.save(self.state_dict(), filepath)

    def load_checkpoint(self, suffix=""):
        filepath = self.checkpoint_file_base + suffix + ".pth"
        self.load_state_dict(T.load(filepath, map_location=self.device))
