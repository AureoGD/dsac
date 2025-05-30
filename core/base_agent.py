# dsac/core/base_agent.py
import os
import numpy as np
import torch as T
# from torch.utils.tensorboard import SummaryWriter # Trainer will manage and pass writer
from abc import ABC, abstractmethod
from .replay_buffer import ReplayBuffer


class BaseAgent(ABC):

    def __init__(
            self,
            env,
            gamma=0.99,
            tau=0.005,
            replay_buffer_size=1_000_000,
            batch_size=256,
            learning_starts=1000,
            gradient_steps=1,
            policy_delay=2,
            max_grad_norm=None,
            chkpt_dir="tmp/base_agent",  # Should be overridden by subclass
            # For Replay Buffer flexibility:
        action_shape=None,
            action_dtype=np.float32,
            aux_data_specs=None):

        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.gradient_steps = gradient_steps
        self.policy_delay = policy_delay
        self.max_grad_norm = max_grad_norm
        self.chkpt_dir = chkpt_dir
        os.makedirs(self.chkpt_dir, exist_ok=True)

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")

        # _agent_obs_shapes and _agent_use_encoder are expected to be set by subclass before this call
        obs_shapes_for_buffer_dict = self._get_obs_shapes_for_buffer()

        if action_shape is None:
            raise ValueError("action_shape must be provided to BaseAgent for ReplayBuffer setup.")

        self.memory = ReplayBuffer(max_size=replay_buffer_size,
                                   obs_space_dict=obs_shapes_for_buffer_dict,
                                   action_shape=action_shape,
                                   action_dtype=action_dtype,
                                   aux_data_specs=aux_data_specs)

        self.writer = None  # This will be set by the Trainer
        self.learn_step_counter = 0  # Counts critic gradient steps / main learning steps

        self.actor = None
        self.critic_1, self.critic_2 = None, None
        self.target_critic_1, self.target_critic_2 = None, None
        self.log_alpha, self.alpha, self.alpha_optimizer, self.target_entropy, self.entropy_tuning = [None] * 5

        self.last_actor_loss, self.last_critic_loss, self.last_ent_coef_loss = np.nan, np.nan, np.nan

        self.hparams = {
            'base_agent_gamma': gamma,
            'base_agent_tau': tau,
            'replay_buffer_max_size': replay_buffer_size,
            'batch_size': batch_size,
            'learning_starts': learning_starts,
            'gradient_steps': gradient_steps,
            'policy_delay': policy_delay,
            'max_grad_norm_base': max_grad_norm,
            'action_shape_buffer': str(action_shape),
            'action_dtype_buffer': str(action_dtype),
            'aux_data_specs_buffer': str(aux_data_specs)
        }
        # Subclasses must call self._setup_networks() after setting their specific attributes

    @abstractmethod
    def _get_obs_shapes_for_buffer(self):
        """Implemented by subclasses to provide obs_space_dict for ReplayBuffer."""
        pass

    @abstractmethod
    def _setup_networks(self):
        """Implemented by subclasses to initialize actor, critic, alpha."""
        pass

    @abstractmethod
    def choose_action(self, observation, evaluate=False):
        pass

    def learn(self):
        if not self.memory.ready(self.batch_size, self.learning_starts):
            return False

        learned_something_this_call = False
        for _ in range(self.gradient_steps):
            batch_data_tuple = self.memory.sample_buffer(self.batch_size)

            self._update_critic(batch_data_tuple)
            self.learn_step_counter += 1

            if self.learn_step_counter % self.policy_delay == 0:
                self._update_actor_alpha_and_targets(batch_data_tuple)
            learned_something_this_call = True

        return learned_something_this_call

    @abstractmethod
    def _update_critic(self, batch_data_tuple):
        pass

    @abstractmethod
    def _update_actor_alpha_and_targets(self, batch_data_tuple):
        pass

    def remember(self, state_dict_or_flat, action, reward, new_state_dict_or_flat, done, aux_data=None):
        # Convert to dicts if flat, matching _get_obs_shapes_for_buffer structure
        obs_buffer_keys = list(self._get_obs_shapes_for_buffer().keys())

        if not isinstance(state_dict_or_flat, dict):
            if len(obs_buffer_keys) == 1:
                state_dict_to_store = {obs_buffer_keys[0]: state_dict_or_flat}
            else:
                raise ValueError("Flat state provided but buffer expects multiple obs keys.")
        else:
            state_dict_to_store = state_dict_or_flat

        if not isinstance(new_state_dict_or_flat, dict):
            if len(obs_buffer_keys) == 1:
                new_state_dict_to_store = {obs_buffer_keys[0]: new_state_dict_or_flat}
            else:
                raise ValueError("Flat new_state provided but buffer expects multiple obs keys.")
        else:
            new_state_dict_to_store = new_state_dict_or_flat

        self.memory.store_transition(state_dict_to_store, action, reward, new_state_dict_to_store, done, aux_data)

    def update_target_networks(self, tau=None):
        if tau is None: tau = self.tau
        if self.critic_1 is None or self.target_critic_1 is None: return

        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save_models(self, best_model=False):
        if self.actor is None:
            print("Agent models not initialized, cannot save.")
            return
        print(f"... saving {'best ' if best_model else ''}models in {self.chkpt_dir} ...")
        suffix = "_best" if best_model else ""
        if hasattr(self.actor, 'save_checkpoint'): self.actor.save_checkpoint(suffix=suffix)
        if hasattr(self.critic_1, 'save_checkpoint'): self.critic_1.save_checkpoint(suffix=suffix)
        if hasattr(self.critic_2, 'save_checkpoint'): self.critic_2.save_checkpoint(suffix=suffix)

    def load_models(self, best_model=False):
        # Ensure networks are created before trying to load state dicts
        if self.actor is None: self._setup_networks()

        print(f"... loading {'best ' if best_model else ''}models from {self.chkpt_dir} ...")
        suffix = "_best" if best_model else ""
        try:
            if hasattr(self.actor, 'load_checkpoint'): self.actor.load_checkpoint(suffix=suffix)
            if hasattr(self.critic_1, 'load_checkpoint'): self.critic_1.load_checkpoint(suffix=suffix)
            if hasattr(self.critic_2, 'load_checkpoint'): self.critic_2.load_checkpoint(suffix=suffix)

            if self.target_critic_1 and self.critic_1:
                self.target_critic_1.load_state_dict(self.critic_1.state_dict())
            if self.target_critic_2 and self.critic_2:
                self.target_critic_2.load_state_dict(self.critic_2.state_dict())
            print("Models loaded successfully.")
        except FileNotFoundError as e:
            print(f"Error loading models: {e}. Checkpoint files might be missing.")
        except Exception as e:
            print(f"An unexpected error occurred during model loading: {e}")
