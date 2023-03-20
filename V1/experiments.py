import numpy as np

# TODO adjust state to be a tuple of price and time

from Environment import TVEnvironment
from TVMDP import QLearningMDP

from tqdm import tqdm
import time

N_PEOPLE = 2000
L_PRICE = 1.00
H_PRICE = 15.00
ACTIONS = np.array([-1.00, -0.50, -0.25, -0.10, -0.05, 0.00, 0.05, 0.10, 0.25, 0.50, 1.00])
# no explicit state space because made unbounded
PRICES = np.arange(L_PRICE, H_PRICE + 0.05, 0.05) 
GAMMA = 0.01
# TODO adjust alpha parameter?
ALPHA = 0.1
N_DAYS = 365
# can make q learning run for a number of episodes or until a level of convergence met
# specifies time horizon in the case that we use this as the stopping condition
N_EPISODES = 100

def main():
    dist_choices = ["Unimodal_Gaussian", "Multimodal_Gaussian", "Uniform"]
    epsilon = 0.1
    # TODO add parameter for epsilon greedy decay
    alpha = 0.01

    start_time = time.time()

    #for dist in dist_choices:
    #    results = experiment(epsilon, alpha, dist)
    results = experiment(epsilon, alpha, "Unimodal_Gaussian")

    total_time = time.time() - start_time
    print(total_time)

def experiment(epsilon, alpha, distribution): 
    env = TVEnvironment(N_PEOPLE)

    # initialize the environment
    env.set_random_distribution(distribution, L_PRICE, H_PRICE)
    opt_price = env.get_optimal_price()
    opt_profit = env.get_profit(opt_price)

    # initialize MDP and function parameters
    mdp = QLearningMDP(n_actions = len(ACTIONS), gamma=GAMMA, alpha=ALPHA)

    # initialize state
    price = (H_PRICE - L_PRICE) / 2
    state = [price, 0]
    # current optimal price
    cur_opt_price = state[0]

    # store histories
    price_traj = [state[0]]
    profit_traj = [env.get_profit(state[0])]
    #opt_price_traj = [cur_opt_price]
    #opt_profit_traj = [env.get_profit(cur_opt_price)]

    for t in range(N_EPISODES):
        # if year has passed reset
        day = t % 365

        # update state (note price already set; update time)
        state[1] = day

        # select an action using the current policy
        action_idx = get_action(state, mdp, epsilon)
        # update parameters
        price_change = ACTIONS[action_idx]
        next_price = state[0] + price_change
        # observe reward
        env.update_tau(state[1]) # update tau (price thresholds for individuals)
        reward = env.get_profit(next_price)
        # update state
        next_state = [next_price, (day + 1) % 365]
        # get next action for update
        aprime_idx = get_action(next_state, mdp, epsilon)
        aprime = ACTIONS[aprime_idx]
        maxQ = env.get_profit(next_price + aprime)

        # update q approximation given observation and chosen action
        mdp.update_theta(state, reward, next_state, maxQ, price_change)

        # updates for next iteration
        state[0] = next_price

        # update trajectories
        price_traj.append(next_price)
        profit_traj.append(reward)


def get_action(state, mdp, epsilon):
    return get_epsilon_greedy_action(state, mdp, epsilon)

def get_epsilon_greedy_action(state, mdp, epsilon):
    # TODO plan on using epsilon decay for V1 problem desc
    if (np.random.choice([False, True], p=[1 - epsilon, epsilon])):
        # choose a random action
        return int(np.random.choice(np.arange(mdp.n_actions)))
    # get action that maximizes current Q
    expected_rewards = []
    for i in range(len(ACTIONS)):
        expected_rewards.append(mdp.get_action_value(PRICES, state))
    return np.argmax(expected_rewards)

if __name__ == "__main__":
    main()
