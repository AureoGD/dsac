import os
import numpy as np
import torch as T
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym

# Adjust relative imports based on the project structure:
# If dsac is the top-level package in PYTHONPATH:
# from dsac.core.base_agent import BaseAgent
# from dsac.sac_das.networks import ActorNetworkDiscrete, CriticNetworkDiscrete
# from dsac.core.base_networks import SimpleMLPEncoder, IdentityEncoder, BaseEncoder
# If running with `python -m dsac.run_example` and this file is part of the dsac package:
from ...core.base_agent import BaseAgent
from ..networks import ActorNetworkDiscrete, CriticNetworkDiscrete
from ...core.base_networks import SimpleMLPEncoder, IdentityEncoder, BaseEncoder


class DSAC(BaseAgent):

    def __init__(self,
                 env: gym.Env,
                 obs_shapes_or_space,
                 use_encoder: bool = False,
                 encoder_mlp_hidden_dims: list = None,
                 hidden_dims_actor_body: list = None,
                 hidden_dims_critic_body: list = None,
                 alpha_init: any = "auto",
                 critic_lr: float = 3e-4,
                 actor_lr: float = 3e-4,
                 alpha_lr: float = 3e-4,
                 reward_scale: float = 1.0,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 replay_buffer_size: int = 1_000_000,
                 batch_size: int = 256,
                 learning_starts: int = 1000,
                 gradient_steps: int = 1,
                 policy_delay: int = 1,
                 max_grad_norm: float = None,
                 aux_data_specs: dict = None,
                 chkpt_dir: str = "tmp/dsac_christodoulou"):

        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("DSAC (Christodoulou) expects a gymnasium.spaces.Discrete action space.")

        self.n_discrete_actions = env.action_space.n
        action_shape_for_buffer = ()
        action_dtype_for_buffer = np.int64

        self._agent_obs_shapes_or_space = obs_shapes_or_space
        self._agent_use_encoder = use_encoder
        self._encoder_mlp_hidden_dims = encoder_mlp_hidden_dims if encoder_mlp_hidden_dims is not None else []
        self._hidden_dims_actor_body = hidden_dims_actor_body if hidden_dims_actor_body is not None else [256, 256]
        self._hidden_dims_critic_body = hidden_dims_critic_body if hidden_dims_critic_body is not None else [256, 256]

        super().__init__(env=env,
                         gamma=gamma,
                         tau=tau,
                         replay_buffer_size=replay_buffer_size,
                         batch_size=batch_size,
                         learning_starts=learning_starts,
                         gradient_steps=gradient_steps,
                         policy_delay=policy_delay,
                         max_grad_norm=max_grad_norm,
                         chkpt_dir=chkpt_dir,
                         action_shape=action_shape_for_buffer,
                         action_dtype=action_dtype_for_buffer,
                         aux_data_specs=aux_data_specs)

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.alpha_init_val = alpha_init
        self.reward_scale = reward_scale

        self.hparams.update({
            'agent_variant': 'DSAC_Christodoulou',
            'obs_shapes_config': str(self._agent_obs_shapes_or_space),
            'use_encoder_config': self._agent_use_encoder,
            'encoder_mlp_hidden_dims_cfg': str(self._encoder_mlp_hidden_dims),
            'hidden_dims_actor_body': str(self._hidden_dims_actor_body),
            'hidden_dims_critic_body': str(self._hidden_dims_critic_body),
            'actor_lr': self.actor_lr,
            'critic_lr': self.critic_lr,
            'alpha_lr': self.alpha_lr,
            'alpha_init': self.alpha_init_val,
            'reward_scale': self.reward_scale,
            'n_discrete_actions': self.n_discrete_actions
        })

        # self.writer is set by Trainer
        self._setup_networks()

    def _get_obs_shapes_for_buffer(self):
        if self._agent_use_encoder and isinstance(self._agent_obs_shapes_or_space, (dict, gym.spaces.Dict)):
            if isinstance(self._agent_obs_shapes_or_space, gym.spaces.Dict):
                return {k: tuple(v.shape) for k, v in self._agent_obs_shapes_or_space.spaces.items()}
            return {k: tuple(v_s) for k, v_s in self._agent_obs_shapes_or_space.items()}
        elif isinstance(self._agent_obs_shapes_or_space, gym.spaces.Box):
            return {"obs": self._agent_obs_shapes_or_space.shape}
        elif isinstance(self._agent_obs_shapes_or_space, tuple):
            return {"obs": self._agent_obs_shapes_or_space}
        else:
            raise ValueError(f"Unsupported obs_shapes_or_space for buffer: {self._agent_obs_shapes_or_space}")

    def _setup_networks(self):
        obs_dim_for_simple_encoder = 0
        encoder_to_use: BaseEncoder  # Type hint

        # Determine input dimension for SimpleMLPEncoder or IdentityEncoder if not using custom dict encoder
        if isinstance(self._agent_obs_shapes_or_space, gym.spaces.Box):
            obs_dim_for_simple_encoder = self._agent_obs_shapes_or_space.shape[0]
        elif isinstance(self._agent_obs_shapes_or_space, tuple):
            obs_dim_for_simple_encoder = self._agent_obs_shapes_or_space[0]
        # If it's a dict and use_encoder is True, it's handled next.

        if self._agent_use_encoder and isinstance(self._agent_obs_shapes_or_space, (dict, gym.spaces.Dict)):
            # TODO: Instantiate your MultiHeadFeatureExtractor or similar dict encoder here
            # This encoder should inherit from BaseEncoder
            # from ....custom_feature_extractors.multihead_feature_extractor import MultiHeadFeatureExtractor
            # encoder_to_use = MultiHeadFeatureExtractor(observation_space=self._agent_obs_shapes_or_space, ...)
            # For now, this path needs a concrete dictionary encoder implementation from you.
            raise NotImplementedError(
                "DSAC: Dictionary observation encoder (e.g., MultiHeadFeatureExtractor) needs to be provided and instantiated here when use_encoder=True for dict observations."
            )
        elif not self._encoder_mlp_hidden_dims:  # If empty list for MLP hidden dims, use IdentityEncoder
            encoder_to_use = IdentityEncoder(obs_dim=obs_dim_for_simple_encoder).to(self.device)
        else:  # Use SimpleMLPEncoder for flat observations with specified hidden layers
            encoder_to_use = SimpleMLPEncoder(obs_dim=obs_dim_for_simple_encoder,
                                              hidden_dims_encoder=self._encoder_mlp_hidden_dims).to(self.device)

        self.actor = ActorNetworkDiscrete(actor_lr=self.actor_lr,
                                          n_discrete_actions=self.n_discrete_actions,
                                          encoder=encoder_to_use,
                                          hidden_dims_body=self._hidden_dims_actor_body,
                                          chkpt_dir=self.chkpt_dir,
                                          name="actor_dsac")
        critic_params = {
            "critic_lr": self.critic_lr,
            "n_discrete_actions": self.n_discrete_actions,
            "encoder": encoder_to_use,
            "hidden_dims_body": self._hidden_dims_critic_body,
            "chkpt_dir": self.chkpt_dir
        }
        self.critic_1 = CriticNetworkDiscrete(**critic_params, name="critic_1_dsac")
        self.critic_2 = CriticNetworkDiscrete(**critic_params, name="critic_2_dsac")
        self.target_critic_1 = CriticNetworkDiscrete(**critic_params, name="target_critic_1_dsac")
        self.target_critic_2 = CriticNetworkDiscrete(**critic_params, name="target_critic_2_dsac")

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        for p in self.target_critic_1.parameters():
            p.requires_grad = False
        for p in self.target_critic_2.parameters():
            p.requires_grad = False

        if isinstance(self.alpha_init_val, str) and self.alpha_init_val.lower() == "auto":
            self.target_entropy = 0.98 * np.log(float(self.n_discrete_actions))
            self.log_alpha = T.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = T.optim.Adam([self.log_alpha], lr=self.alpha_lr)
            self.entropy_tuning = True
            self.alpha = self.log_alpha.exp().detach()
        else:
            self.entropy_tuning = False
            self.alpha = T.tensor(float(self.alpha_init_val), device=self.device)
            self.alpha_optimizer = None
            self.log_alpha = None
        self.hparams[
            'target_entropy'] = self.target_entropy if self.entropy_tuning else f"Fixed_Alpha ({self.alpha_init_val})"

    def choose_action(self, observation, evaluate=False):
        self.actor.eval()

        obs_for_net = {}  # Will be populated if dict observation
        if self._agent_use_encoder and isinstance(self._agent_obs_shapes_or_space, (dict, gym.spaces.Dict)):
            # Assuming observation is already a dict from env.reset() or env.step()
            obs_tensor = {
                k: T.tensor(np.array(v), dtype=T.float32).unsqueeze(0).to(self.device)
                for k, v in observation.items()
            }
        else:
            obs_tensor = T.tensor(np.array(observation), dtype=T.float32).unsqueeze(0).to(self.device)

        with T.no_grad():
            action_logits = self.actor.forward(obs_tensor)
            action_probs = F.softmax(action_logits, dim=-1)
            if evaluate:
                action = T.argmax(action_probs, dim=-1).item()
            else:
                dist = Categorical(probs=action_probs)
                action = dist.sample().item()
        self.actor.train()
        return action

    def _update_critic(self, batch_data_tuple):
        states_dict, actions_idx, rewards, next_states_dict, dones, _ = batch_data_tuple

        # Prepare observation tensors based on whether a dict encoder is used
        if self._agent_use_encoder and isinstance(self._agent_obs_shapes_or_space, (dict, gym.spaces.Dict)):
            states = {k: T.tensor(v, dtype=T.float32).to(self.device) for k, v in states_dict.items()}
            next_states = {k: T.tensor(v, dtype=T.float32).to(self.device) for k, v in next_states_dict.items()}
        else:  # Flat observations, buffer stores it under "obs" key (defined in _get_obs_shapes_for_buffer)
            states = T.tensor(states_dict["obs"], dtype=T.float32).to(self.device)
            next_states = T.tensor(next_states_dict["obs"], dtype=T.float32).to(self.device)

        actions = T.tensor(actions_idx, dtype=T.int64).to(self.device).unsqueeze(1)
        rewards = T.tensor(rewards, dtype=T.float32).to(self.device).unsqueeze(1)
        dones_float = T.tensor(dones, dtype=T.float32).to(self.device).unsqueeze(1)

        with T.no_grad():
            next_action_logits = self.actor.forward(next_states)
            next_action_probs = F.softmax(next_action_logits, dim=-1)
            next_action_log_probs = F.log_softmax(next_action_logits, dim=-1)

            q1_target_all = self.target_critic_1.forward(next_states)
            q2_target_all = self.target_critic_2.forward(next_states)
            q_target_min_all = T.min(q1_target_all, q2_target_all)

            soft_state_values_next = (next_action_probs *
                                      (q_target_min_all - self.alpha.detach() * next_action_log_probs)).sum(
                                          dim=1, keepdim=True)
            q_target = self.reward_scale * rewards + self.gamma * (1.0 - dones_float) * soft_state_values_next

        q1_current_all = self.critic_1.forward(states)
        q2_current_all = self.critic_2.forward(states)
        q1_current = T.gather(q1_current_all, dim=1, index=actions)
        q2_current = T.gather(q2_current_all, dim=1, index=actions)

        critic_1_loss = F.mse_loss(q1_current, q_target)
        critic_2_loss = F.mse_loss(q2_current, q_target)
        critic_total_loss = critic_1_loss + critic_2_loss

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_total_loss.backward()
        if self.max_grad_norm is not None:
            T.nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.max_grad_norm)
            T.nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.max_grad_norm)
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        self.last_critic_loss = critic_total_loss.item() / 2.0

    def _update_actor_alpha_and_targets(self, batch_data_tuple):
        states_dict, _, _, _, _, _ = batch_data_tuple

        if self._agent_use_encoder and isinstance(self._agent_obs_shapes_or_space, (dict, gym.spaces.Dict)):
            states = {k: T.tensor(v, dtype=T.float32).to(self.device) for k, v in states_dict.items()}
        else:
            states = T.tensor(states_dict["obs"], dtype=T.float32).to(self.device)

        for p in self.critic_1.parameters():
            p.requires_grad = False
        for p in self.critic_2.parameters():
            p.requires_grad = False

        action_logits = self.actor.forward(states)
        action_probs = F.softmax(action_logits, dim=-1)
        action_log_probs = F.log_softmax(action_logits, dim=-1)

        q1_all_actions = self.critic_1.forward(states)
        q2_all_actions = self.critic_2.forward(states)
        q_min_all_actions = T.min(q1_all_actions, q2_all_actions).detach()

        inside_term = self.alpha.detach() * action_log_probs - q_min_all_actions
        actor_loss = (action_probs * inside_term).sum(dim=1).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        if self.max_grad_norm is not None: T.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor.optimizer.step()
        self.last_actor_loss = actor_loss.item()

        for p in self.critic_1.parameters():
            p.requires_grad = True
        for p in self.critic_2.parameters():
            p.requires_grad = True

        if self.entropy_tuning:
            # Calculate current entropy H(π(s_t)) for each state in the batch
            current_entropy_per_state = -(action_probs * action_log_probs).sum(dim=1)
            # Loss for log_alpha: E_s [ log_alpha * (H_current(s) - H_target) ]
            alpha_loss = (self.log_alpha * (current_entropy_per_state - self.target_entropy).detach()).mean()

            if self.alpha_optimizer is not None:
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                if self.max_grad_norm is not None:
                    T.nn.utils.clip_grad_norm_([self.log_alpha], self.max_grad_norm)  # Clip grad of log_alpha
                self.alpha_optimizer.step()

                # --- Start: Clipping log_alpha to control alpha's range ---
                with T.no_grad():
                    # To cap alpha at 1.0, log_alpha should be capped at log(1.0) = 0.0
                    LOG_ALPHA_MAX = 0.0  # Equivalent to np.log(1.0)

                    # Optional: Define a minimum for alpha (e.g., 0.001) to prevent it from becoming too small
                    # LOG_ALPHA_MIN = np.log(0.001) # Approximately -6.9
                    # self.log_alpha.data.clamp_(LOG_ALPHA_MIN, LOG_ALPHA_MAX)

                    # If you only want to cap the maximum value of alpha at 1.0:
                    self.log_alpha.data.clamp_max_(LOG_ALPHA_MAX)
                # --- End: Clipping log_alpha ---

            self.alpha = self.log_alpha.exp().detach()  # Update alpha based on potentially clipped log_alpha
            self.last_ent_coef_loss = alpha_loss.item()
        else:
            self.last_ent_coef_loss = np.nan  # Not tuning alpha

        self.update_target_networks()
