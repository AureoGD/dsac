import gymnasium as gym
import yaml
import os
import argparse
import importlib
import numpy as np
import sys  # For sys.exit

# Assuming Trainer and BaseAgent are in core and dsac is importable (e.g. via python -m)
from .core.trainer import Trainer
# from dsac.core.base_agent import BaseAgent # Optional: for type hinting if needed locally


def load_config(config_path):
    """Loads training configuration from a YAML file."""
    if not os.path.exists(config_path):
        # Try resolving relative to script directory if not found directly
        script_dir = os.path.dirname(__file__)
        abs_config_path = os.path.join(script_dir, config_path)
        if not os.path.exists(abs_config_path):
            raise FileNotFoundError(f"Configuration file not found at '{config_path}' or '{abs_config_path}'")
        config_path = abs_config_path  # Use absolute path if found relative to script

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded configuration from: {config_path}")
    return config


def get_env_action_spec(env_action_space: gym.Space) -> tuple:
    """Determines action_shape and action_dtype for the replay buffer."""
    if isinstance(env_action_space, gym.spaces.Box):  # Continuous
        action_shape = env_action_space.shape
        action_dtype = np.float32
    elif isinstance(env_action_space, gym.spaces.Discrete):  # Discrete
        action_shape = ()  # Scalar integer for action index
        action_dtype = np.int64
    else:
        raise ValueError(f"Unsupported action space type: {type(env_action_space)}")
    return action_shape, action_dtype


def main(args):
    # 1. Load Configuration
    config = load_config(args.config)

    # 2. Create Environments
    env_id = config.get('env_id', 'Pendulum-v1')  # Default if not specified
    print(f"Creating training environment: {env_id}")
    env = gym.make(env_id)

    eval_env_id = config.get('eval_env_id', env_id)  # Default to training env_id
    print(f"Creating evaluation environment: {eval_env_id}")
    eval_env = gym.make(eval_env_id)

    # 3. Dynamically Load Agent Class
    # Config expects 'agent_module' like "sac_cas.sac_cas_agent" (relative to 'dsac' package)
    # and 'agent_class' like "SacCasAgent"
    agent_module_config_path = config.get("agent_module")
    agent_class_name = config.get("agent_class")

    if not agent_module_config_path or not agent_class_name:
        print("Error: 'agent_module' and 'agent_class' must be specified in the configuration file.")
        sys.exit(1)

    # Construct full module path assuming 'dsac' is the root package for these modules.
    # This works when running as 'python -m dsac.run_example ...'
    full_module_path_for_importlib = f"dsac.{agent_module_config_path}"

    print(f"Attempting to load agent: {full_module_path_for_importlib}.{agent_class_name}")
    try:
        module = importlib.import_module(full_module_path_for_importlib)
        AgentClass = getattr(module, agent_class_name)
    except ModuleNotFoundError as e_import:
        print(f"Error: Could not import module '{full_module_path_for_importlib}' or class '{agent_class_name}'.")
        print(f"Details: {e_import}")
        print("Ensure that the directory *containing* 'dsac' is in your PYTHONPATH,")
        print("and you are running this script as a module (e.g., 'python -m dsac.run_example ...').")
        print("Also check that all subdirectories (dsac, core, sac_cas, sac_das, etc.) have __init__.py files.")
        sys.exit(1)
    except AttributeError as e_attr:
        print(f"Error: Class '{agent_class_name}' not found in module '{full_module_path_for_importlib}'.")
        print(f"Details: {e_attr}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during dynamic agent import: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 4. Prepare Parameters for Trainer and Agent
    # Observation space for agent/encoder
    if config.get('use_encoder', False) and isinstance(env.observation_space, gym.spaces.Dict):
        obs_shapes_or_space_for_agent = env.observation_space  # Pass the whole Dict space
    else:
        obs_shapes_or_space_for_agent = env.observation_space.shape  # Pass shape tuple

    # Action spec for replay buffer
    action_shape, action_dtype = get_env_action_spec(env.action_space)
    action_spec_for_buffer = (action_shape, action_dtype)

    # Max gradient norm
    max_grad_norm_from_config = config.get('max_grad_norm', None)
    if isinstance(max_grad_norm_from_config, (int, float)):
        max_grad_norm_val = float(max_grad_norm_from_config)
    else:
        max_grad_norm_val = None

    # Encoder MLP hidden dims (ensure it's a list)
    encoder_mlp_hidden_dims_val = config.get('encoder_mlp_hidden_dims', [])  # Default to empty list
    if encoder_mlp_hidden_dims_val is None: encoder_mlp_hidden_dims_val = []

    # Agent body hidden dims (ensure they are lists)
    hidden_dims_actor_body_val = config.get('hidden_dims_actor_body', [256, 256])
    if hidden_dims_actor_body_val is None: hidden_dims_actor_body_val = [256, 256]

    hidden_dims_critic_body_val = config.get('hidden_dims_critic_body', [256, 256])
    if hidden_dims_critic_body_val is None: hidden_dims_critic_body_val = [256, 256]

    # 5. Initialize Trainer
    print("Initializing Trainer...")
    try:
        trainer = Trainer(
            agent_class=AgentClass,
            env=env,
            eval_env=eval_env,
            training_total_timesteps=int(config.get('training_total_timesteps', 100000)),
            max_steps_per_episode=int(config.get('max_steps_per_episode', 1000)),
            log_interval_timesteps=int(config.get('log_interval_timesteps', 2048)),
            eval_frequency_timesteps=int(config.get('eval_frequency_timesteps', 10000)),
            n_eval_episodes=int(config.get('n_eval_episodes', 5)),
            log_root=config.get('log_root', "runs_dsac"),
            model_root=config.get('model_root', "models_dsac"),
            save_freq_episodes=int(config.get('save_freq_episodes', 100)),

            # Agent Hyperparameters passed to Trainer, then to Agent constructor
            obs_shapes_or_space=obs_shapes_or_space_for_agent,
            use_encoder=config.get('use_encoder', False),
            encoder_mlp_hidden_dims=encoder_mlp_hidden_dims_val,
            hidden_dims_actor_body=hidden_dims_actor_body_val,  # For agent's network body
            hidden_dims_critic_body=hidden_dims_critic_body_val,  # For agent's network body
            gamma=float(config.get('gamma', 0.99)),
            tau=float(config.get('tau', 0.005)),
            alpha_init=config.get('alpha_init', "auto"),
            critic_lr=float(config.get('critic_lr', 3e-4)),
            actor_lr=float(config.get('actor_lr', 3e-4)),
            alpha_lr=float(config.get('alpha_lr', config.get('actor_lr', 3e-4))),
            replay_buffer_size=int(config.get('replay_buffer_size', 1000000)),
            batch_size=int(config.get('batch_size', 256)),
            learning_starts=int(config.get('learning_starts', 1000)),
            gradient_steps=int(config.get('gradient_steps', 1)),
            policy_delay=int(config.get('policy_delay', 2)),
            max_grad_norm=max_grad_norm_val,
            reward_scale=float(config.get('reward_scale', 1.0)),
            action_spec_for_buffer=action_spec_for_buffer,
            aux_data_specs_for_buffer=config.get('aux_data_specs_for_buffer', None),
            agent_specific_kwargs=config.get('agent_specific_kwargs', None))
    except Exception as e_trainer:
        print(f"Error initializing Trainer: {e_trainer}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 6. Start Training
    print(f"Starting training for {trainer.training_total_timesteps} timesteps...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e_train:
        print(f"An error occurred during training: {e_train}")
        import traceback
        traceback.print_exc()
    finally:
        # 7. Cleanup
        print("Closing trainer and environments...")
        trainer.close()
        print("Cleanup complete. Training run finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run custom SAC training experiments.")
    parser.add_argument(
        "--config",
        type=str,
        # Default config assumes it's in dsac/configs/ when script is run from dsac/ parent
        default="dsac/configs/continuous_sac_pendulum.yaml",
        help="Path to the YAML configuration file for training (relative to project root if using -m).")
    args = parser.parse_args()

    # Best way to run:
    # Navigate to the directory *containing* 'dsac' (e.g., /home/agd/)
    # Then run: python -m dsac.run_example --config dsac/configs/your_config.yaml
    main(args)
