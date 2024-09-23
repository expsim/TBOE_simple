import numpy as np


class RandomOctopus:
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def select_action(self, state):
        return [np.random.randint(self.action_dim) for _ in range(4)]