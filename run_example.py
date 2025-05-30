import gymnasium as gym
import yaml
import os
import argparse
import importlib
import numpy as np

# Assuming Trainer and BaseAgent are in core
from core.trainer import Trainer
from core.base_agent import BaseAgent  # For type hinting if needed


def load_config(config_path="configs/continuous_sac_pendulum.yaml"):  # Default to an example config
    """Loads training configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded configuration from: {config_path}")
    return config


def get_env_action_spec(env_action_space):
    """Determines action_shape and action_dtype for the replay buffer based on the environment."""
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
    # Load configuration
    config = load_config(args.config)

    # Create the training environment
    print(f"Creating training environment: {config['env_id']}")
    env = gym.make(config['env_id'])

    # Create the evaluation environment
    eval_env = None
    eval_env_id = config.get('eval_env_id', config['env_id'])  # Default to training env_id
    if eval_env_id:
        print(f"Creating evaluation environment: {eval_env_id}")
        eval_env = gym.make(eval_env_id)
    else:  # Should not happen if default is training env_id
        print("No evaluation environment specified. Evaluation will use the training environment instance.")
        eval_env = env

    # Dynamically load the agent class
    agent_module_name = config.get("agent_module", "sac_cas.sac_cas_agent")  # e.g., "sac_cas.sac_cas_agent"
    agent_class_name = config.get("agent_class", "SacCasAgent")  # e.g., "SacCasAgent"

    print(f"Loading agent: {agent_module_name}.{agent_class_name}")
    try:
        # Assuming this script is run from within dsac/ or dsac/ is in PYTHONPATH
        # For example, if dsac is a package, and agent_module is like "sac_cas.sac_cas_agent"
        # It would correspond to dsac/sac_cas/sac_cas_agent.py
        module = importlib.import_module(agent_module_name)
        AgentClass = getattr(module, agent_class_name)
    except ModuleNotFoundError:
        # Try relative import if script is inside dsac and modules are submodules
        try:
            module = importlib.import_module(f".{agent_module_name}", package="dsac")  # Assuming top package is dsac
            AgentClass = getattr(module, agent_class_name)
        except Exception as e:
            print(
                f"Error importing agent module '{agent_module_name}' or class '{agent_class_name}'. Make sure PYTHONPATH is set up correctly or paths are relative to 'dsac' package."
            )
            raise e

    # Determine obs_shapes_or_space for the agent and trainer
    # If the agent's encoder (like MultiHead) expects the full gym.spaces.Dict:
    if config.get('use_encoder', False) and isinstance(env.observation_space, gym.spaces.Dict):
        obs_shapes_or_space_for_agent = env.observation_space  # Pass the whole Dict space
    else:  # Pass the shape tuple for flat observations
        obs_shapes_or_space_for_agent = env.observation_space.shape

    # Determine action_spec_for_buffer for the Trainer/Agent
    action_shape, action_dtype = get_env_action_spec(env.action_space)
    action_spec_for_buffer = (action_shape, action_dtype)

    # Initialize the Trainer
    print("Initializing Trainer...")

    # Handle max_grad_norm from config (can be float or null/None)
    max_grad_norm_from_config = config.get('max_grad_norm', None)  # Default to None
    if isinstance(max_grad_norm_from_config, (int, float)):
        max_grad_norm_val = float(max_grad_norm_from_config)
    else:  # If it's 'null' from YAML (becomes None) or missing
        max_grad_norm_val = None

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
        # callback=None, # Add callback instantiation here if needed

        # Agent Hyperparameters (passed to Trainer's __init__, then to Agent)
        obs_shapes_or_space=obs_shapes_or_space_for_agent,
        use_encoder=config.get('use_encoder', False),
        encoder_mlp_hidden_dims=config.get('encoder_mlp_hidden_dims', [64, 64]),  # Used if agent uses SimpleMLPEncoder
        gamma=float(config.get('gamma', 0.99)),
        tau=float(config.get('tau', 0.005)),
        alpha_init=config.get('alpha_init', "auto"),  # Keep as str or float
        critic_lr=float(config.get('critic_lr', 3e-4)),
        actor_lr=float(config.get('actor_lr', 3e-4)),
        replay_buffer_size=int(config.get('replay_buffer_size', 1000000)),
        batch_size=int(config.get('batch_size', 256)),
        learning_starts=int(config.get('learning_starts', 100)),
        gradient_steps=int(config.get('gradient_steps', 1)),
        policy_delay=int(config.get('policy_delay', 2)),
        max_grad_norm=max_grad_norm_val,
        reward_scale=float(config.get('reward_scale', 1.0)),
        action_spec_for_buffer=action_spec_for_buffer,
        aux_data_specs_for_buffer=config.get('aux_data_specs_for_buffer', None)
        # agent_specific_kwargs can be added to config and passed here if needed
    )

    print(f"Starting training for {config.get('training_total_timesteps', 100000)} timesteps...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Closing trainer and environments...")
        trainer.close()
        print("Cleanup complete. Training run finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run custom SAC training.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/continuous_sac_pendulum.yaml",  # Default config file
        help="Path to the YAML configuration file for training.")
    args = parser.parse_args()

    # Ensure the script can find modules in the 'dsac' directory
    # If 'dsac' is not installed as a package and you run this script from outside 'dsac',
    # you might need to adjust PYTHONPATH or sys.path.
    # If run as 'python run_experiment.py' from within 'dsac/', imports should work if
    # subdirectories have __init__.py to be treated as packages.
    # For robust execution, ensure 'dsac' or its parent is in PYTHONPATH.

    main(args)
