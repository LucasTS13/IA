import numpy as np
from rl.qll_feature_extractor import FeatureExtractor

class CliffWalkingFeatureExtractor(FeatureExtractor):
    
    def __init__(self, env):
        self.env = env
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n

    def get_num_features(self):
        # O número de features é N_ESTADOS * N_AÇÕES
        return self.num_states * self.num_actions

    def get_features(self, state, action):
        """
        Retorna um vetor one-hot para o par (estado, ação).
        """
        feature_index = state * self.num_actions + action
        feature_vector = np.zeros(self.get_num_features())
        feature_vector[feature_index] = 1.0
        
        return feature_vector

    def is_terminal_state(self, state):
        return state == 47 # O objetivo (Goal)

    def get_action_one_hot_encoded(self, action):
        """
        Método "fantasma" para satisfazer a classe abstrata FeatureExtractor.
        """
        return np.array([])