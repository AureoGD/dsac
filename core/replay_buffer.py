# dsac/core/replay_buffer.py
import numpy as np


class ReplayBuffer:

    def __init__(self, max_size, obs_space_dict, action_shape, action_dtype=np.float32, aux_data_specs=None):
        self.mem_size = int(max_size)
        self.mem_cntr = 0

        self.state_memory = {}
        self.new_state_memory = {}
        if not obs_space_dict: raise ValueError("obs_space_dict cannot be empty for ReplayBuffer.")
        for key, shape in obs_space_dict.items():
            self.state_memory[key] = np.zeros((self.mem_size, *shape), dtype=np.float32)
            self.new_state_memory[key] = np.zeros((self.mem_size, *shape), dtype=np.float32)

        self.action_memory = np.zeros((self.mem_size, *action_shape), dtype=action_dtype)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        self.aux_memory = {}
        self.aux_data_specs = aux_data_specs if aux_data_specs else {}
        if self.aux_data_specs:
            for name, spec in self.aux_data_specs.items():
                if "shape" not in spec or "dtype" not in spec:
                    raise ValueError(f"Spec for aux data '{name}' must include 'shape' and 'dtype'.")
                self.aux_memory[name] = np.zeros((self.mem_size, *spec["shape"]), dtype=spec["dtype"])

    def store_transition(self, state_dict, action, reward, new_state_dict, done, aux_data=None):
        index = self.mem_cntr % self.mem_size

        for key in self.state_memory.keys():
            if key in state_dict:
                self.state_memory[key][index] = state_dict[key]
            else:
                print(f"Warning: Key '{key}' from obs_space_dict not in provided state for storing.")

        for key in self.new_state_memory.keys():
            if key in new_state_dict:
                self.new_state_memory[key][index] = new_state_dict[key]
            else:
                print(f"Warning: Key '{key}' from obs_space_dict not in new_state for storing.")

        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        if self.aux_memory and aux_data:
            for name, data_array_storage in self.aux_memory.items():
                if name in aux_data:
                    data_array_storage[index] = aux_data[name]
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        if batch_size > max_mem:
            raise ValueError(f"Cannot sample {batch_size} elements, only {max_mem} are available.")

        batch_indices = np.random.choice(max_mem, batch_size, replace=False)

        states = {key: self.state_memory[key][batch_indices] for key in self.state_memory.keys()}
        new_states = {key: self.new_state_memory[key][batch_indices] for key in self.new_state_memory.keys()}
        actions = self.action_memory[batch_indices]
        rewards = self.reward_memory[batch_indices]
        dones = self.terminal_memory[batch_indices]
        sampled_aux_data = {name: storage[batch_indices] for name, storage in self.aux_memory.items()}

        return states, actions, rewards, new_states, dones, sampled_aux_data

    def size(self):
        return min(self.mem_cntr, self.mem_size)

    def ready(self, batch_size, learning_starts):
        return self.mem_cntr >= max(batch_size, learning_starts)
