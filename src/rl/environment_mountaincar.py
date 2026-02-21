import numpy as np
from rl.environment import Environment

class MountainCarEnvironment(Environment):
    def __init__(self, env, num_pos_bins=20, num_vel_bins=20):
        super().__init__(env)
        
        # Define a discretização
        self.num_pos_bins = num_pos_bins
        self.num_vel_bins = num_vel_bins
        
        # Cria divisórias para posição e velocidade
        self.pos_bins = np.linspace(
            self.env.observation_space.low[0],
            self.env.observation_space.high[0],
            self.num_pos_bins - 1
        )
        self.vel_bins = np.linspace(
            self.env.observation_space.low[1],
            self.env.observation_space.high[1],
            self.num_vel_bins - 1
        )

    def get_num_states(self):
        return self.num_pos_bins * self.num_vel_bins

    def get_num_actions(self):
        return self.env.action_space.n

    def get_state_id(self, state):
        pos, vel = state

        pos_bin = min(np.digitize(pos, self.pos_bins), self.num_pos_bins - 1)
        vel_bin = min(np.digitize(vel, self.vel_bins), self.num_vel_bins - 1)

        return pos_bin * self.num_vel_bins + vel_bin

    def get_random_action(self):
        return self.env.action_space.sample()