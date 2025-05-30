import os
import time
import json
import numpy as np
import gymnasium as gym
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Since Trainer is in core, BaseAgent is in the same directory (or submodule)
from .base_agent import BaseAgent  # For type hinting


class BaseCallback:
    """
    Base class for Callbacks.
    The trainer and agent instances will be injected by the Trainer.
    """

    def __init__(self, verbose=0):
        self.verbose = verbose
        self.trainer = None  # type: Trainer
        self.agent = None  # type: BaseAgent
        self.logger = None  # type: SummaryWriter

    def _on_training_start(self):
        """Called before the first episode of training."""
        pass

    def on_training_start(self):
        self._on_training_start()

    def _on_training_end(self):
        """Called after the training loop is finished."""
        pass

    def on_training_end(self):
        self._on_training_end()

    def _on_episode_start(self):
        """Called at the beginning of each episode."""
        pass

    def on_episode_start(self):
        self._on_episode_start()

    def _on_step(self) -> bool:
        """
        Called after each `env.step()`.
        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def on_step(self) -> bool:
        return self._on_step()

    def _on_rollout_end(self):
        """Called after collecting all steps in an episode, before any learning for that episode's data."""
        pass

    def on_rollout_end(self):
        self._on_rollout_end()

    def _on_episode_end(self):
        """Called after the episode loop, including all learning updates triggered by that episode's steps."""
        pass

    def on_episode_end(self):
        self._on_episode_end()


class Trainer:

    def __init__(
        self,
        agent_class,
        env: gym.Env,
        eval_env: gym.Env = None,
        training_total_timesteps: int = 100000,
        max_steps_per_episode: int = 1000,
        log_interval_timesteps: int = 2048,
        eval_frequency_timesteps: int = 10000,
        n_eval_episodes: int = 5,
        log_root: str = "runs_dsac",
        model_root: str = "models_dsac",
        save_freq_episodes: int = 100,
        callback=None,
        obs_shapes_or_space=None,
        use_encoder: bool = False,
        encoder_mlp_hidden_dims: list = None,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_init="auto",
        critic_lr: float = 3e-4,
        actor_lr: float = 3e-4,
        replay_buffer_size: int = 1_000_000,
        batch_size: int = 256,
        learning_starts: int = 100,
        gradient_steps: int = 1,
        policy_delay: int = 2,
        max_grad_norm: float = None,
        reward_scale: float = 1.0,
        action_spec_for_buffer: tuple = None,
        aux_data_specs_for_buffer: dict = None,
        agent_specific_kwargs: dict = None,
    ):
        self.agent_class = agent_class
        self.env = env
        self.eval_env = eval_env if eval_env is not None else env

        try:
            self.env_name = self.env.spec.id if self.env.spec else "UnknownEnv"
        except AttributeError:
            self.env_name = env.unwrapped.spec.id if hasattr(env, 'unwrapped') and hasattr(
                env.unwrapped, 'spec') and env.unwrapped.spec else "UnknownEnv"

        self.training_total_timesteps = int(training_total_timesteps)
        self.max_steps_per_episode = int(max_steps_per_episode)
        self.log_interval_timesteps = int(log_interval_timesteps)
        self.eval_frequency_timesteps = int(eval_frequency_timesteps)
        self.n_eval_episodes = int(n_eval_episodes)
        self.save_freq_episodes = int(save_freq_episodes)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        agent_name_simple = self.agent_class.__name__.replace("Agent", "").lower()
        run_name = f"{agent_name_simple}_{self.env_name.replace('/', '-')}_{timestamp}"
        self.log_dir_run = os.path.join(log_root, run_name)
        self.model_dir_run = os.path.join(model_root, run_name)

        os.makedirs(self.log_dir_run, exist_ok=True)
        os.makedirs(self.model_dir_run, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir_run)

        self.trainer_hparams = {
            'agent_class_name': self.agent_class.__name__,
            'training_total_timesteps': self.training_total_timesteps,
            'max_steps_per_episode': self.max_steps_per_episode,
            'log_interval_timesteps': self.log_interval_timesteps,
            'eval_frequency_timesteps': self.eval_frequency_timesteps,
            'n_eval_episodes': self.n_eval_episodes,
            'save_freq_episodes': self.save_freq_episodes,
            'env_id': self.env_name,
        }

        self.agent_constructor_hparams = {
            'env': self.env,
            'obs_shapes_or_space': obs_shapes_or_space,
            'use_encoder': use_encoder,
            'encoder_mlp_hidden_dims': encoder_mlp_hidden_dims or [],
            'gamma': gamma,
            'tau': tau,
            'alpha_init': alpha_init,
            'critic_lr': critic_lr,
            'actor_lr': actor_lr,
            'replay_buffer_size': replay_buffer_size,
            'batch_size': batch_size,
            'learning_starts': learning_starts,
            'gradient_steps': gradient_steps,
            'policy_delay': policy_delay,
            'max_grad_norm': max_grad_norm,
            'reward_scale': reward_scale,
            'chkpt_dir': self.model_dir_run,
            'log_dir': self.log_dir_run,
            'action_shape': action_spec_for_buffer[0] if action_spec_for_buffer else None,
            'action_dtype': action_spec_for_buffer[1] if action_spec_for_buffer else np.float32,
            'aux_data_specs': aux_data_specs_for_buffer
        }
        if agent_specific_kwargs: self.agent_constructor_hparams.update(agent_specific_kwargs)

        self.agent: BaseAgent = self._setup_agent_instance()
        if hasattr(self.agent, 'writer') and self.agent.writer is not None:
            if self.agent.writer != self.writer: self.agent.writer.close()
            self.agent.writer = self.writer

        self._save_hyperparameters()

        if callback is None: self._callbacks = []
        elif isinstance(callback, list): self._callbacks = callback
        else: self._callbacks = [callback]
        for cb_instance in self._callbacks:
            cb_instance.trainer = self
            cb_instance.agent = self.agent
            cb_instance.logger = self.writer

        self.total_timesteps = 0
        self.total_episodes_completed = 0
        self.start_time = None
        self.last_log_timestep = 0
        self.last_eval_timestep = 0
        self.last_log_time = None
        self.best_mean_eval_reward = -np.inf
        self._train_metric_accumulators = {}
        self._train_metric_counts = {}

    def _save_hyperparameters(self):
        agent_constructor_serializable = {k: v for k, v in self.agent_constructor_hparams.items() if k != 'env'}
        for k, v in agent_constructor_serializable.items():
            if isinstance(v, type):
                agent_constructor_serializable[k] = str(v)
            elif isinstance(v, gym.spaces.Dict) and k == 'obs_shapes_or_space':
                agent_constructor_serializable[k] = {key_s: str(sp.shape) for key_s, sp in v.spaces.items()}
            elif isinstance(v, (np.ndarray, gym.spaces.Space, tuple)) and k not in ['action_shape']:
                agent_constructor_serializable[k] = str(v)
            elif isinstance(v, tuple) and not isinstance(v, str):
                agent_constructor_serializable[k] = str(v)

        all_hparams = {
            "trainer_hyperparameters": self.trainer_hparams,
            "agent_constructor_hyperparameters_passed": agent_constructor_serializable,
            "agent_internal_hyperparameters": self.agent.hparams if hasattr(self.agent, 'hparams') else {}
        }
        filepath = os.path.join(self.model_dir_run, "hyperparameters.json")
        try:
            with open(filepath, 'w') as f:
                json.dump(all_hparams, f, indent=4, sort_keys=True)
        except TypeError as e:
            print(f"Warning: Could not serialize all HPs to JSON: {e}")
            print(f"Problematic HPs (agent_constructor_serializable dump):")
            for key, val in agent_constructor_serializable.items():
                print(f"  {key}: {val} (type: {type(val)})")
        except Exception as e_gen:
            print(f"An unexpected error occurred during HP saving: {e_gen}")

    def _setup_agent_instance(self):
        kwargs_for_agent = dict(self.agent_constructor_hparams)
        valid_agent_params = list(self.agent_class.__init__.__code__.co_varnames)
        if 'env' in valid_agent_params and 'env' not in kwargs_for_agent:
            kwargs_for_agent['env'] = self.env
        filtered_kwargs_for_agent = {k: v for k, v in kwargs_for_agent.items() if k in valid_agent_params}
        if 'obs_shapes_or_space' not in filtered_kwargs_for_agent and \
           'obs_shapes_or_space' in valid_agent_params and \
           'obs_shapes_or_space' in kwargs_for_agent:
            filtered_kwargs_for_agent['obs_shapes_or_space'] = kwargs_for_agent['obs_shapes_or_space']
        agent_instance = self.agent_class(**filtered_kwargs_for_agent)
        return agent_instance

    def _call_callbacks(self, event_method_name):
        continue_training = True
        for callback_instance in self._callbacks:
            method = getattr(callback_instance, event_method_name, None)
            if method and not method(): continue_training = False
        return continue_training

    def _reset_train_metric_accumulators(self):
        self._train_metric_accumulators = {"actor_loss": 0.0, "critic_loss": 0.0, "ent_coef_loss": 0.0}
        self._train_metric_counts = {"actor_loss": 0, "critic_loss": 0, "ent_coef_loss": 0}

    def _accumulate_train_metrics(self):
        steps_in_learn = self.agent.gradient_steps
        if hasattr(self.agent, 'last_actor_loss') and not np.isnan(self.agent.last_actor_loss):
            self._train_metric_accumulators["actor_loss"] += self.agent.last_actor_loss * steps_in_learn
            self._train_metric_counts["actor_loss"] += steps_in_learn
        if hasattr(self.agent, 'last_critic_loss') and not np.isnan(self.agent.last_critic_loss):
            self._train_metric_accumulators["critic_loss"] += self.agent.last_critic_loss * steps_in_learn
            self._train_metric_counts["critic_loss"] += steps_in_learn
        if hasattr(self.agent,'entropy_tuning') and self.agent.entropy_tuning and \
           hasattr(self.agent, 'last_ent_coef_loss') and not np.isnan(self.agent.last_ent_coef_loss):
            self._train_metric_accumulators["ent_coef_loss"] += self.agent.last_ent_coef_loss * steps_in_learn
            self._train_metric_counts["ent_coef_loss"] += steps_in_learn

    def _run_evaluation(self):
        if self.n_eval_episodes <= 0: return
        eval_rewards, eval_lengths = [], []
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done, ep_rew, ep_len = False, 0, 0
            for _eval_step in range(self.max_steps_per_episode):
                if done: break
                action = self.agent.choose_action(obs, evaluate=True)
                next_obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                ep_rew += reward
                ep_len += 1
                obs = next_obs
            eval_rewards.append(ep_rew)
            eval_lengths.append(ep_len)
        mean_reward, mean_length = np.mean(eval_rewards), np.mean(eval_lengths)
        self.writer.add_scalar("eval/mean_reward", mean_reward, self.total_timesteps)
        self.writer.add_scalar("eval/mean_episode_length", mean_length, self.total_timesteps)

        if mean_reward > self.best_mean_eval_reward:
            print(
                f"Eval: New best eval reward: {mean_reward:.2f} (old: {self.best_mean_eval_reward:.2f}). Saving best model..."
            )
            self.best_mean_eval_reward = mean_reward
            self.agent.save_models(best_model=True)
        # else: print(f"Eval: Mean Reward: {mean_reward:.2f}, Mean Length: {mean_length:.2f}") # Less verbose
        self.last_eval_timestep = self.total_timesteps

    def _log_terminal_summary(self, mean_rollout_reward, mean_rollout_length, interval_fps, total_time_elapsed):
        LABEL_CONTENT_WIDTH = 23
        VALUE_CONTENT_WIDTH = 12
        TOTAL_LINE_WIDTH = LABEL_CONTENT_WIDTH + VALUE_CONTENT_WIDTH + 7

        actor_loss_val, critic_loss_val, ent_coef_loss_val = np.nan, np.nan, np.nan
        if self._train_metric_counts.get("actor_loss", 0) > 0:
            actor_loss_val = self._train_metric_accumulators["actor_loss"] / self._train_metric_counts["actor_loss"]
        if self._train_metric_counts.get("critic_loss", 0) > 0:
            critic_loss_val = self._train_metric_accumulators["critic_loss"] / self._train_metric_counts["critic_loss"]
        if hasattr(self.agent, 'entropy_tuning') and self.agent.entropy_tuning and self._train_metric_counts.get(
                "ent_coef_loss", 0) > 0:
            ent_coef_loss_val = self._train_metric_accumulators["ent_coef_loss"] / self._train_metric_counts[
                "ent_coef_loss"]
        ent_coef_val = self.agent.alpha.item() if hasattr(self.agent,
                                                          'alpha') and self.agent.alpha is not None else np.nan
        actor_lr_val = self.agent.actor.optimizer.param_groups[0]['lr'] if hasattr(
            self.agent, 'actor') and self.agent.actor and hasattr(
                self.agent.actor, 'optimizer') and self.agent.actor.optimizer.param_groups else np.nan
        n_updates_val = self.agent.learn_step_counter if hasattr(self.agent, 'learn_step_counter') else np.nan

        def format_label_cell(text, indent_level=0):
            return f"{'  '*indent_level}{text}".ljust(LABEL_CONTENT_WIDTH)

        def format_value_cell(value, precision_format_char):
            # Check if value is a string first (like the "Value" header or empty string for section rows)
            if isinstance(value, str):
                val_str = value  # Keep it as is for simple string values
            # Check for actual numeric types that could be NaN
            elif isinstance(value,
                            (int, float, np.number)) and np.isnan(value):  # np.number covers numpy int/float types
                val_str = "nan"
            # Proceed with numeric formatting if not a string and not NaN
            elif precision_format_char == 'd':  # Integer
                # Ensure value is convertible to int if it's not already NaN
                val_str = f"{int(value):d}" if not (isinstance(value, float) and np.isnan(value)) else "nan"
            elif 'e' in precision_format_char:  # Scientific
                val_str = f"{value:{precision_format_char}}" if not (isinstance(value, float)
                                                                     and np.isnan(value)) else "nan"
            else:  # Default to float with specified precision (e.g., .2f, .3f, .5f)
                val_str = f"{value:{precision_format_char}}" if not (isinstance(value, float)
                                                                     and np.isnan(value)) else "nan"

            return val_str.rjust(VALUE_CONTENT_WIDTH)

        print("\n" + "-" * TOTAL_LINE_WIDTH)
        print(f"| {format_label_cell('Section/Metric')} | {format_value_cell('Value', '')} |")
        line_sep = f"|{'-'*(LABEL_CONTENT_WIDTH + 2)}|{'-'*(VALUE_CONTENT_WIDTH + 2)}|"
        print(line_sep)

        sections = {
            "rollout/": [("ep_len_mean", mean_rollout_length, '.2f'), ("ep_rew_mean", mean_rollout_reward, '.2e')],
            "time/": [("episodes", self.total_episodes_completed, 'd'), ("fps", interval_fps, 'd'),
                      ("time_elapsed", int(total_time_elapsed), 'd'), ("total_timesteps", self.total_timesteps, 'd')],
            "train/": [("actor_loss", actor_loss_val, '.3f'), ("critic_loss", critic_loss_val, '.3f'),
                       ("ent_coef", ent_coef_val, '.5f'), ("ent_coef_loss", ent_coef_loss_val, '.3f'),
                       ("learning_rate", actor_lr_val, '.1e'), ("n_updates", n_updates_val, 'd')]
        }
        for section_name, metrics in sections.items():
            print(f"| {format_label_cell(section_name)} | {format_value_cell('', '')} |")
            for label_text, value, specific_format_char in metrics:
                print(
                    f"| {format_label_cell(label_text, indent_level=1)} | {format_value_cell(value, specific_format_char)} |"
                )
            if section_name != list(sections.keys())[-1]: print(line_sep)
        print("-" * TOTAL_LINE_WIDTH)

    def train(self):
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.last_log_timestep = 0
        self.last_eval_timestep = 0
        if not self._call_callbacks('on_training_start'):
            self.close()
            return
        self._reset_train_metric_accumulators()
        reward_history_window, length_history_window, current_episode_num = [], [], 0
        window_size = 100

        while self.total_timesteps < self.training_total_timesteps:
            current_episode_num += 1
            self.total_episodes_completed = current_episode_num
            if not self._call_callbacks('on_episode_start'): break
            observation, info = self.env.reset()
            done, episode_reward, episode_steps = False, 0, 0

            while not done and episode_steps < self.max_steps_per_episode and self.total_timesteps < self.training_total_timesteps:
                action = self.agent.choose_action(observation, evaluate=False)
                next_observation, reward, terminated, truncated, info_step = self.env.step(action)
                done = terminated or truncated

                obs_buffer_keys = list(self.agent._get_obs_shapes_for_buffer().keys())
                current_obs_dict = observation if isinstance(observation, dict) else {obs_buffer_keys[0]: observation}
                next_obs_dict = next_observation if isinstance(next_observation, dict) else {
                    obs_buffer_keys[0]: next_observation
                }
                self.agent.remember(current_obs_dict, action, reward, next_obs_dict, done)

                self.total_timesteps += 1
                episode_steps += 1
                episode_reward += reward

                if self.total_timesteps >= self.agent.learning_starts:
                    if self.agent.learn(): self._accumulate_train_metrics()

                if not self._call_callbacks('on_step'):
                    self.total_timesteps = self.training_total_timesteps
                    done = True
                observation = next_observation

            if not self._call_callbacks('on_rollout_end'): break

            reward_history_window.append(episode_reward)
            length_history_window.append(episode_steps)
            if len(reward_history_window) > window_size: reward_history_window.pop(0)
            if len(length_history_window) > window_size: length_history_window.pop(0)
            mean_reward_w = np.mean(reward_history_window) if reward_history_window else np.nan
            mean_length_w = np.mean(length_history_window) if length_history_window else np.nan

            self.writer.add_scalar("rollout/ep_reward", episode_reward, self.total_timesteps)
            self.writer.add_scalar("rollout/ep_length", episode_steps, self.total_timesteps)
            self.writer.add_scalar("rollout/ep_rew_mean", mean_reward_w, self.total_timesteps)
            self.writer.add_scalar("rollout/ep_len_mean", mean_length_w, self.total_timesteps)

            if (self.total_timesteps -
                    self.last_log_timestep) >= self.log_interval_timesteps and self.total_timesteps > 0:
                curr_time = time.time()
                total_time_elapsed = curr_time - self.start_time
                time_for_interval = curr_time - self.last_log_time
                steps_in_interval = self.total_timesteps - self.last_log_timestep
                interval_fps = int(steps_in_interval /
                                   time_for_interval) if time_for_interval > 0 and steps_in_interval > 0 else 0

                self.writer.add_scalar("time/fps", interval_fps, self.total_timesteps)
                self.writer.add_scalar("time/episodes", self.total_episodes_completed, self.total_timesteps)
                self.writer.add_scalar("time/time_elapsed", total_time_elapsed, self.total_timesteps)

                if self._train_metric_counts.get("actor_loss", 0) > 0:
                    self.writer.add_scalar(
                        "train/actor_loss",
                        self._train_metric_accumulators["actor_loss"] / self._train_metric_counts["actor_loss"],
                        self.total_timesteps)
                    self.writer.add_scalar(
                        "train/critic_loss",
                        self._train_metric_accumulators["critic_loss"] / self._train_metric_counts["critic_loss"],
                        self.total_timesteps)
                    if hasattr(self.agent, 'entropy_tuning'
                               ) and self.agent.entropy_tuning and self._train_metric_counts["ent_coef_loss"] > 0:
                        self.writer.add_scalar(
                            "train/ent_coef_loss", self._train_metric_accumulators["ent_coef_loss"] /
                            self._train_metric_counts["ent_coef_loss"], self.total_timesteps)

                ent_coef_val_tb = self.agent.alpha.item() if hasattr(
                    self.agent, 'alpha') and self.agent.alpha is not None else np.nan
                actor_lr_val_tb = self.agent.actor.optimizer.param_groups[0]['lr'] if hasattr(
                    self.agent, 'actor') and self.agent.actor and hasattr(
                        self.agent.actor, 'optimizer') and self.agent.actor.optimizer.param_groups else np.nan
                self.writer.add_scalar("train/ent_coef", ent_coef_val_tb, self.total_timesteps)
                if not np.isnan(actor_lr_val_tb):
                    self.writer.add_scalar("train/learning_rate", actor_lr_val_tb, self.total_timesteps)
                n_updates_tb = self.agent.learn_step_counter if hasattr(self.agent, 'learn_step_counter') else np.nan
                self.writer.add_scalar("train/n_updates", n_updates_tb, self.total_timesteps)
                self.writer.flush()

                self._log_terminal_summary(mean_reward_w, mean_length_w, interval_fps, total_time_elapsed)

                self.last_log_timestep = self.total_timesteps
                self.last_log_time = curr_time
                self._reset_train_metric_accumulators()

            if self.total_timesteps - self.last_eval_timestep >= self.eval_frequency_timesteps and self.total_timesteps > 0:
                self._run_evaluation()
            if current_episode_num > 0 and current_episode_num % self.save_freq_episodes == 0:
                self.agent.save_models(best_model=False)
            if not self._call_callbacks('on_episode_end'): break
            if self.total_timesteps >= self.training_total_timesteps and (done or episode_steps
                                                                          >= self.max_steps_per_episode):
                break

        self._call_callbacks('on_training_end')
        self.close()

    def close(self):
        if self.writer: self.writer.close()
        if self.eval_env is not None and self.eval_env != self.env and hasattr(self.eval_env, 'close'):
            self.eval_env.close()
        if hasattr(self.env, 'close'): self.env.close()
        print("Trainer closed.")
