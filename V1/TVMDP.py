import numpy as np
import tensorflow as tf
from tensorflow import keras

class QLearningMDP:
    def __init__(self, n_actions: int, gamma: float, alpha: float):
        self.n_states = n_states
        self.gamma = gamma # discount
        self.alpha = alpha # learning rate
        # initialize beta and theta to initialize the Q function
        self.theta = np.zeros(5) # action value function parameter #TODO verify correct dimensions
        self.B = (np.pi * 2) / 365

    def get_gradient(self, state, action):
        # TODO check
        return beta_point_evaluation(action, state[0], state[1])

    # reference Algorithm 12.2 in book
    def scale_gradient(self, gradient, set_max):
        np.min((set_max / np.norm(gradient)), 1)*gradient

    def update_theta(self, state, reward, next_state, maxQ_term, action):
        partial = maxQ_term * self.gamma
        partial += reward
        partial -= get_action_value(action, state) 
        partial *= get_gradient(state, action)
        self.theta += (self.alpha * scale_gradient(partial, 1))

    # beta = [1, price + action, (price + action)^2, sinBt, cosBt]
    def beta_point_evaluation(self, action, state, time):
        beta = np.ones(5)
        beta[0] = 1.00
        beta[1] = (float)(price + action)
        beta[2] = beta[1]*beta[1]
        beta[3] = np.sin(self.B * time)
        beta[4] = np.cos(self.B * time)
        return beta

    def get_action_value(self, action, state):
        # returns Q(cur_price, time, action)
        return np.sum(self.theta * beta_point_evaluation(action, state[0], state[1]))


