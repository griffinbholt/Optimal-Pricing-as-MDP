# experiment using V1 implementation

import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt

from Environment import DynamicPricingEnvironment
from Environment2 import TVEnvironment
from MDP import MaximumLikelihoodMDP
from TVMDP import TimeVaryingMDP

from tqdm import tqdm
import time

def compute_transition_probs():
    transition_probs = np.zeros((len(STATES), len(STATES), len(ACTIONS)))
    for state in range(len(STATES)):
        for action_idx in range(len(ACTIONS)):
            next_price = STATES[state] + ACTIONS[action_idx]
            next_state = round((next_price - L_PRICE) / 0.05)
            next_state = 0 if next_state < 0 else next_state
            next_state = len(STATES) - 1 if next_state >= len(STATES) else next_state
            transition_probs[state, next_state, action_idx] = 1
    return transition_probs

N_PEOPLE = 2000
U_PRICE = 15.00
L_PRICE = 1.00
ACTIONS = np.array([-1.00, -0.50, -0.25, -0.10, -0.05, 0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00])
STATES = np.arange(L_PRICE, U_PRICE + 0.05, 0.05)
GAMMA = 0.01
N_DAYS = 365
TRANSITION_PROBS = compute_transition_probs()

def main():
    n_trials_per_experiment = 1
    distrib_choices = ["Unimodal_Gaussian", "Multimodal_Gaussian", "Uniform"]
    restart_choices = [False, True]
    action_strategies = ["epsilon_greedy", "softmax", "UCB1"]
    c = [0.1, 1, 10, 100, 1000, 10000, 100000]
    alpha = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    epsilon = [0.1, 0.5, 0.9]
    change_over_time_choices = [False, True]
    growth = [1.01, 1.1, 1.5, 2]
    decay = [0.99, 0.9, 0.75, 0.5]

    start_time = time.time()

    for distrib in distrib_choices:
        for restart in restart_choices:
            for action_strategy in action_strategies:
                if action_strategy == "epsilon_greedy":
                    parameters = epsilon
                elif action_strategy == "softmax":
                    parameters = alpha
                else: # Uniform
                    parameters = c 
                for parameter in parameters:
                    for change_over_time in change_over_time_choices:
                        if change_over_time:
                            if (action_strategy == "epsilon_greedy") or (action_strategy == "UCB1"):
                                mult_factors = decay
                            else: # Softmax
                                mult_factors = growth
                        else:
                            mult_factors = [1]
                        for mult_factor in mult_factors:
                            print("Experiment: {} {} {} {:f} {} {:f}".format(distrib, restart, action_strategy, parameter, change_over_time, mult_factor))
                            results = experiment(n_trials_per_experiment, distrib, restart, action_strategy, parameter, mult_factor)
                            
                            # TODO - Figure out what to do with results

    total_time = time.time() - start_time
    print(total_time)

def experimenttime(n_trials, env_distribution, restart_every_two_weeks, action_strategy, action_param, mult_factor):
    env = DynamicPricingEnvironment(N_PEOPLE)

    for _ in range(n_trials):
        env.set_random_distribution(env_distribution, L_PRICE, U_PRICE)
        opt_price = env.get_optimal_price()
        opt_profit = env.get_profit(opt_price)

        mdp = MaximumLikelihoodMDP(n_states=len(STATES), n_actions=len(ACTIONS), gamma=GAMMA)
        mdp.set_transition_probs(TRANSITION_PROBS)

        price = (U_PRICE - L_PRICE) / 2
        # converted states to tuples to account for day (initialized to 0)
        state = (get_state(price), 0)
        curr_opt_price = price

        price_trajectory = [price]
        profit_trajectory = [env.get_profit(price)]
        opt_price_trajectory = [curr_opt_price]
        opt_profit_trajectory = [env.get_profit(curr_opt_price)]

        for t in range(N_DAYS):
            if restart_every_two_weeks and (t % 14 == 0):
                price = (U_PRICE - L_PRICE) / 2
                # increment day for state for each iteration looping around to 0 each year
                state = (get_state(price), t % 365)

            action_idx, action_param = get_action2(action_strategy, action_param, state, mdp, mult_factor)
            price_change = ACTIONS[action_idx]
            next_price = price + price_change
            next_state = get_state(next_price)
            reward = env.get_profit(next_state)

            env.update_tau(t)
            mdp.update_counts(state, action_idx, reward, next_state)
            mdp.update_reward()
            mdp.value_iteration(tol=1e-2)

            state = next_state
            price = next_price
            curr_opt_price = STATES[np.argmax(mdp.value)] + ACTIONS[mdp.policy[np.argmax(mdp.value)]]

            price_trajectory.append(price)
            profit_trajectory.append(reward)
            opt_price_trajectory.append(curr_opt_price)
            opt_profit_trajectory.append(env.get_profit(curr_opt_price))
        
    # TODO - Record & return analysis

def get_state(price):
    state = round((price - L_PRICE) / 0.05)
    state = 0 if state < 0 else state
    state = len(STATES) - 1 if state >= len(STATES) else state
    return state

def get_action(action_strategy, action_param, state, mdp, mult_factor):
    if action_strategy == "epsilon_greedy":
        action_idx = get_epsilon_greedy_action(state, mdp, action_param)
    elif action_strategy == "softmax":
        action_idx = get_softmax_strategy_action(state, mdp, action_param)
    else: # UCB1
        action_idx = get_ucb1_action(state, mdp, action_param)
    return action_idx, (mult_factor * action_param)
    
# note: only takes into account price aspect of state
def get_action2(action_strategy, action_param, state, mdp, mult_factor):
    if action_strategy == "epsilon_greedy":
        action_idx = get_epsilon_greedy_action(state[0], mdp, action_param)
    elif action_strategy == "softmax":
        action_idx = get_softmax_strategy_action(state[0], mdp, action_param)
    else: # UCB1
        action_idx = get_ucb1_action(state[0], mdp, action_param)
    return action_idx, (mult_factor * action_param)
    
def get_epsilon_greedy_action(state, mdp, epsilon): # Choose random action with probability epsilon
    if np.random.choice([False, True], p=[1 - epsilon, epsilon]):
        return int(np.random.choice(np.arange(mdp.n_actions)))
    expected_rewards = mdp.get_action_value(state)
    return np.random.choice(np.argwhere(expected_rewards == expected_rewards.max()).T[0])
    

def get_softmax_strategy_action(state, mdp, alpha):
    expected_rewards = mdp.get_action_value(state)
    return int(np.random.choice(np.arange(mdp.n_actions), p=softmax(alpha * expected_rewards)))

def get_ucb1_action(state, mdp, c):
    expected_rewards = mdp.get_action_value(state)
    N_a = np.sum(mdp.transition_counts[state], axis=0)
    N = N_a.sum()
    if N != 0:
        exploration_bonus = np.sqrt(np.divide(np.log(N), N_a, out=(np.inf * np.ones(mdp.n_actions)), where=(N_a != 0)))
        ucb1 = expected_rewards + c * exploration_bonus
        return np.random.choice(np.argwhere(ucb1 == ucb1.max()).T[0])
    return int(np.random.choice(np.arange(mdp.n_actions)))
        

if __name__ == "__main__":
    main()
