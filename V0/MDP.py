import numpy as np
        
class MaximumLikelihoodMDP:
    def __init__(self, n_states: int, n_actions: int, gamma: float):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.value = np.zeros(self.n_states)
        self.transition_counts = np.zeros((self.n_states, self.n_states, self.n_actions))
        self.transition_probs = np.zeros((self.n_states, self.n_states, self.n_actions))
        self.reward_counts = np.zeros((self.n_states, self.n_actions))
        self.reward = np.zeros((self.n_states, self.n_actions))

    def update_counts(self, state: int, action: int, reward: float, next_state: int):
        self.transition_counts[state, next_state, action] += 1 # N(s, s', a)
        self.reward_counts[state, action] += reward # p(s, a)

    def update_probs(self):
        action_state_counts = np.sum(self.transition_counts, axis=1, keepdims=True)
        self.transition_probs = np.divide(self.transition_counts, action_state_counts, out=np.zeros_like(self.transition_probs), where=(action_state_counts != 0))

    def update_reward(self):
        action_state_counts = np.sum(self.transition_counts, axis=1)
        self.reward = np.divide(self.reward_counts, action_state_counts, out=np.zeros_like(self.reward), where=(action_state_counts != 0))

    def update_probs_reward(self):
        action_state_counts = np.sum(self.transition_counts, axis=1, keepdims=True)
        self.transition_probs = np.divide(self.transition_counts, action_state_counts, out=np.zeros_like(self.transition_probs), where=(action_state_counts != 0))
        action_state_counts = np.squeeze(action_state_counts)
        self.reward = np.divide(self.reward_counts, action_state_counts, out=np.zeros_like(self.reward), where=(action_state_counts != 0))

    def value_iteration(self, tol): # Synchronous Update
        max_abs_change = tol + 1
        while max_abs_change >= tol:
            new_value = self._bellman_backup()
            max_abs_change = np.linalg.norm(self.value - new_value, ord=np.inf)
            self.value = new_value
        self.policy = (np.argmax(self.reward + self.gamma * self._expected_future_rewards(), axis=1)).astype(int) # pi(s): Shape n_states

    def _bellman_backup(self):
        return np.max(self.reward + self.gamma * self._expected_future_rewards(), axis=1)

    def _expected_future_rewards(self):
        return np.vstack([self.transition_probs[s].T @ self.value for s in range(self.n_states)])

    def set_transition_probs(self, transition_probs):
        self.transition_probs = transition_probs

    def set_reward(self, reward):
        self.reward = reward   

    def get_action_value(self, state):
        return self.reward[state] + self.gamma * (self.transition_probs[state].T @ self.value)