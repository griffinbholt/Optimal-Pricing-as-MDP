import numpy as np
import tensorflow as tf
from tensorflow import keras

class QLearningMDP:
    def __init__(self, n_actions: int, gamma: float, alpha: float):
        self.n_states = n_states
        self.gamma = gamma # discount
        self.alpha = alpha # learning rate
        self.theta = np.zeros(4) # action value function parameter #TODO verify correct dimensions
        self.beta = 
        self.Q = # parameterized action value function Q(theta, state, action, time)
        self.Qgradient = 

    def get_gradient():
        return self.Qgradient * ()

    def max_action_value(self):

    def update_theta(self):
        self.theta += self.alpha * scale_gradient(1)

    def scale_gradient(self, set_max):
        np.min(set_max / np.norm(self.get_gradient()))*self.get_gradient()

    def get_action_value(self, state, time, action):
        return np.sum(self.theta * self.beta)


