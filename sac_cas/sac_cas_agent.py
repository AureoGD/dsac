# dsac/sac_cas/sac_cas_agent.py
import os
import numpy as np
import torch as T
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym

from ..core.base_agent import BaseAgent
from ..core.base_networks import SimpleMLPEncoder, IdentityEncoder  # Encoders from core
from .networks import ActorNetworkContinuous, CriticNetworkContinuous
# If FeatureEncoderForContinuousSAC is in a separate encoders.py within sac_cas:
# from .encoders.feature_encoders_cas import FeatureEncoderForContinuousSAC
# For this version, networks.py itself contains FeatureEncoderForContinuousSAC
from .networks import FeatureEncoderForContinuousSAC


class SacCasAgent(BaseAgent):

    def __init__(
            self,
            env,
            obs_shapes_or_space,  # Can be tuple for flat, dict of shapes, or gym.spaces.Dict
            use_encoder=False,
            encoder_mlp_hidden_dims=[64],  # For SimpleMLPEncoder if obs is flat & use_encoder is effectively false
            alpha_init="auto",
            critic_lr=3e-4,
            actor_lr=3e-4,
            reward_scale=1.0,
            gamma=0.99,
            tau=0.005,
            replay_buffer_size=1_000_000,
            batch_size=256,
            learning_starts=100,
            gradient_steps=1,
            policy_delay=2,
            max_grad_norm=None,
            chkpt_dir="tmp/sac_cas",
            log_dir="runs/sac_cas"):  # log_dir for agent's own writer if BaseAgent doesn't get one from Trainer

        if not isinstance(env.action_space, gym.spaces.Box) or len(env.action_space.shape) != 1:
            raise ValueError("SacCasAgent expects a flat Box action space (e.g., shape (act_dim,)).")

        self.n_actions = env.action_space.shape[0]
        action_shape_for_buffer = (self.n_actions, )
        action_dtype_for_buffer = np.float32

        # Store params needed before super().__init__ (for _get_obs_shapes_for_buffer)
        self._agent_obs_shapes_or_space = obs_shapes_or_space
        self._agent_use_encoder = use_encoder  # This flag now means "use a potentially complex dict encoder"
        self._encoder_mlp_hidden_dims = encoder_mlp_hidden_dims  # For SimpleMLPEncoder

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
                         aux_data_specs=None)

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_init_val = alpha_init
        self.reward_scale = reward_scale

        self.hparams.update({
            'obs_shapes_config': str(self._agent_obs_shapes_or_space),
            'use_encoder_config': self._agent_use_encoder,
            'encoder_mlp_hidden_dims_cfg': str(self._encoder_mlp_hidden_dims),
            'actor_lr': self.actor_lr,
            'critic_lr': self.critic_lr,
            'alpha_init': self.alpha_init_val,
            'reward_scale': self.reward_scale,
            'n_actions_continuous': self.n_actions
        })

        self.max_action_tensor = T.as_tensor(self.env.action_space.high, dtype=T.float32, device=self.device)
        if self.max_action_tensor.ndim == 0: self.max_action_tensor = self.max_action_tensor.unsqueeze(0)
        if self.max_action_tensor.shape[0] != self.n_actions and self.max_action_tensor.shape[0] == 1:
            self.max_action_tensor = self.max_action_tensor.repeat(self.n_actions)
        elif self.max_action_tensor.shape[0] != self.n_actions:
            raise ValueError(f"max_action_tensor shape error.")

        self._setup_networks()

    def _get_obs_shapes_for_buffer(self):
        # This needs to return a dict of {key: shape_tuple}
        if self._agent_use_encoder and isinstance(self._agent_obs_shapes_or_space, (dict, gym.spaces.Dict)):
            if isinstance(self._agent_obs_shapes_or_space, gym.spaces.Dict):
                return {k: tuple(v.shape) for k, v in self._agent_obs_shapes_or_space.spaces.items()}
            return {k: tuple(v_s) for k, v_s in self._agent_obs_shapes_or_space.items()}  # Is already a dict of shapes
        elif isinstance(self._agent_obs_shapes_or_space, gym.spaces.Box):
            return {"obs": self._agent_obs_shapes_or_space.shape}
        elif isinstance(self._agent_obs_shapes_or_space, tuple):
            return {"obs": self._agent_obs_shapes_or_space}
        else:
            raise ValueError(
                f"Unsupported obs_shapes_or_space type ({type(self._agent_obs_shapes_or_space)}) for buffer setup.")

    def _setup_networks(self):
        max_action_scalar_for_net = float(self.env.action_space.high[0])
        if np.isscalar(self.env.action_space.high): max_action_scalar_for_net = float(self.env.action_space.high)
        else: max_action_scalar_for_net = float(self.env.action_space.high[0])
        self.hparams['max_action_scalar_for_net'] = max_action_scalar_for_net

        # --- Instantiate the chosen ENCODER ---
        if self._agent_use_encoder and isinstance(self._agent_obs_shapes_or_space, (dict, gym.spaces.Dict)):
            print("SacCasAgent: Using FeatureEncoderForContinuousSAC for dictionary observations.")
            # This encoder is defined in sac_cas.networks and uses cnn_lstm_head.FeatureHead
            shared_encoder = FeatureEncoderForContinuousSAC(obs_shapes_or_space=self._agent_obs_shapes_or_space,
                                                            use_encoder_for_dict_obs=True).to(self.device)
        else:  # Flat observation, self._agent_obs_shapes_or_space is a tuple e.g. (obs_dim,) or Box space
            obs_dim = 0
            if isinstance(self._agent_obs_shapes_or_space, gym.spaces.Box):
                obs_dim = self._agent_obs_shapes_or_space.shape[0]
            elif isinstance(self._agent_obs_shapes_or_space, tuple):
                obs_dim = self._agent_obs_shapes_or_space[0]
            else:
                raise TypeError(
                    f"Expected obs_shapes_or_space to be Box or tuple for flat obs, got {type(self._agent_obs_shapes_or_space)}"
                )

            print(f"SacCasAgent: Using SimpleMLPEncoder/Identity for flat observations (dim: {obs_dim}).")
            if not self._encoder_mlp_hidden_dims:  # If no hidden dims for MLP encoder, use Identity
                shared_encoder = IdentityEncoder(obs_dim=obs_dim).to(self.device)
            else:
                shared_encoder = SimpleMLPEncoder(obs_dim=obs_dim,
                                                  hidden_dims_encoder=self._encoder_mlp_hidden_dims).to(self.device)

        self.actor = ActorNetworkContinuous(actor_lr=self.actor_lr,
                                            n_actions=self.n_actions,
                                            max_action_scalar=max_action_scalar_for_net,
                                            encoder=shared_encoder,
                                            chkpt_dir=self.chkpt_dir,
                                            name="actor_continuous")
        critic_net_params = {
            "critic_lr": self.critic_lr,
            "n_actions": self.n_actions,
            "encoder": shared_encoder,
            "chkpt_dir": self.chkpt_dir
        }
        self.critic_1 = CriticNetworkContinuous(**critic_net_params, name="critic_1_continuous")
        self.critic_2 = CriticNetworkContinuous(**critic_net_params, name="critic_2_continuous")
        self.target_critic_1 = CriticNetworkContinuous(**critic_net_params, name="target_critic_1_continuous")
        self.target_critic_2 = CriticNetworkContinuous(**critic_net_params, name="target_critic_2_continuous")

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        for p in self.target_critic_1.parameters():
            p.requires_grad = False
        for p in self.target_critic_2.parameters():
            p.requires_grad = False

        if isinstance(self.alpha_init_val, str) and self.alpha_init_val.lower() == "auto":
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32).item()
            self.log_alpha = T.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = T.optim.Adam([self.log_alpha], lr=self.critic_lr)
            self.entropy_tuning = True
            self.alpha = self.log_alpha.exp().detach()
        else:
            self.entropy_tuning = False
            self.alpha = T.tensor(float(self.alpha_init_val), device=self.device)

    def choose_action(self, observation, evaluate=False):
        self.actor.eval()
        # The 'observation' here is raw from env (dict or flat numpy array)
        # The encoder within self.actor will handle it.
        # Need to convert to tensor(s) first.
        if isinstance(observation, dict):  # For dict observations
            obs_tensor = {
                k: T.tensor(np.array(v), dtype=T.float32).unsqueeze(0).to(self.device)
                for k, v in observation.items()
            }
        else:  # For flat observations
            obs_tensor = T.tensor(np.array(observation), dtype=T.float32).unsqueeze(0).to(self.device)

        with T.no_grad():
            pre_tanh_mu, sigma = self.actor.forward(obs_tensor)
            action_gaussian = pre_tanh_mu if evaluate else Normal(pre_tanh_mu, sigma).rsample()
            action_tanh = T.tanh(action_gaussian)
        scaled_action = action_tanh * self.max_action_tensor
        self.actor.train()
        return scaled_action.cpu().detach().numpy()[0]

    def _update_critic(self, batch_data_tuple):
        states_data_dict, actions_data, rewards_data, next_states_data_dict, dones_data, _ = batch_data_tuple
        # states_data_dict and next_states_data_dict are dicts from replay buffer e.g. {"obs": np_array} or {"key1":np1, ...}

        # Convert to tensors. Networks expect dicts if they use dict encoders, or single tensor if not.
        # The encoder inside the network handles the actual structure.
        # So, we pass the dictionary from the buffer directly if it's a dict observation setup.
        if self._agent_use_encoder and isinstance(self._agent_obs_shapes_or_space, (dict, gym.spaces.Dict)):
            states = {k: T.tensor(v_arr, dtype=T.float32).to(self.device) for k, v_arr in states_data_dict.items()}
            next_states = {
                k: T.tensor(v_arr, dtype=T.float32).to(self.device)
                for k, v_arr in next_states_data_dict.items()
            }
        else:  # Flat observations, buffer stores it under "obs" key
            states = T.tensor(states_data_dict["obs"], dtype=T.float32).to(self.device)
            next_states = T.tensor(next_states_data_dict["obs"], dtype=T.float32).to(self.device)

        actions = T.tensor(actions_data, dtype=T.float32).to(self.device)
        rewards = T.tensor(rewards_data, dtype=T.float32).to(self.device).unsqueeze(1)
        dones = T.tensor(dones_data, dtype=T.bool).to(self.device).unsqueeze(1)

        with T.no_grad():
            next_policy_actions_scaled, next_log_probs = self.actor.sample_normal(next_states, reparameterize=False)
            target_q1 = self.target_critic_1.forward(next_states, next_policy_actions_scaled)
            target_q2 = self.target_critic_2.forward(next_states, next_policy_actions_scaled)
            target_q_min = T.min(target_q1, target_q2)
            q_target = self.reward_scale * rewards + self.gamma * (1.0 - dones.float()) * (target_q_min -
                                                                                           self.alpha * next_log_probs)

        q1_current = self.critic_1.forward(states, actions)
        q2_current = self.critic_2.forward(states, actions)
        critic_1_loss = F.mse_loss(q1_current, q_target)
        critic_2_loss = F.mse_loss(q2_current, q_target)
        critic_loss_total = critic_1_loss + critic_2_loss

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss_total.backward()
        if self.max_grad_norm is not None:
            T.nn.utils.clip_grad_norm_(self.critic_1.parameters(), self.max_grad_norm)
            T.nn.utils.clip_grad_norm_(self.critic_2.parameters(), self.max_grad_norm)
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        self.last_critic_loss = critic_loss_total.item() / 2.0

    def _update_actor_alpha_and_targets(self, batch_data_tuple):
        states_data_dict, _, _, _, _, _ = batch_data_tuple

        if self._agent_use_encoder and isinstance(self._agent_obs_shapes_or_space, (dict, gym.spaces.Dict)):
            states = {k: T.tensor(v_arr, dtype=T.float32).to(self.device) for k, v_arr in states_data_dict.items()}
        else:
            states = T.tensor(states_data_dict["obs"], dtype=T.float32).to(self.device)

        for p in self.critic_1.parameters():
            p.requires_grad = False
        for p in self.critic_2.parameters():
            p.requires_grad = False

        policy_actions_scaled, log_probs = self.actor.sample_normal(states, reparameterize=True)
        q1_policy = self.critic_1.forward(states, policy_actions_scaled)
        q2_policy = self.critic_2.forward(states, policy_actions_scaled)
        q_policy_min = T.min(q1_policy, q2_policy)
        actor_loss = (self.alpha * log_probs - q_policy_min).mean()

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
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            if self.max_grad_norm is not None: T.nn.utils.clip_grad_norm_([self.log_alpha], self.max_grad_norm)
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().detach()
            self.last_ent_coef_loss = alpha_loss.item()
        else:
            self.last_ent_coef_loss = np.nan

        self.update_target_networks()
