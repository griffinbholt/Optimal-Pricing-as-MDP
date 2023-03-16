import numpy as np
import tensorflow as tf
from tensorflow import keras

class TimeVaryingMDP:
    def __init__(self, n_actions: int, gamma: float, alpha_func):
        # TODO depending on representation of time state, state space either becomes very large or continuous
        self.n_actions = n_actions
        self.gamma = gamma # discount
        #self.alpha = alpha_func # learning rate function TODO can adjust
        self.alpha = .5 
        # TODO initialize beta
        self.


    def set_transition_probs(self, transition_probs):
        self.transition_probs = transition_probs

    ###

    def set_transition_probs(self, transition_probs):
        self.transition_probs = transition_probs

    def set_reward(self, reward):
        self.reward = reward   
