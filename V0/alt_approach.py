import numpy as np
import matplotlib.pyplot as plt

from Environment import DynamicPricingEnvironment
from MDP import MaximumLikelihoodMDP

from tqdm import tqdm

N_PEOPLE = 2000
U_PRICE = 15.00
L_PRICE = 1.00
ACTIONS = np.array([-1.00, -0.50, -0.25, -0.10, 0, 0.10, 0.25, 0.50, 0.75, 1.00])
GAMMA = 0.1
N_DAYS = 365

def main():
    np.random.seed(42)
    env = DynamicPricingEnvironment(N_PEOPLE)
    env.set_gaussian_distribution(mean=6.32, std=2)
    opt_price = env.get_optimal_price()
    opt_n_bought = env.get_n_people_bought(opt_price)

    print("Optimal Price: ", opt_price)
    print("Optimal N. Bought: ", opt_n_bought)
    print("Optimal Profit: ", env.get_profit(opt_price))

    mle_mdp = MaximumLikelihoodMDP(n_states=(N_PEOPLE + 1), n_actions=len(ACTIONS), gamma=GAMMA)

    curr_price = (U_PRICE - L_PRICE) / 2
    state = env.get_n_people_bought(curr_price)
    state_trajectory = [state]
    price_trajectory = [curr_price]

    for t in tqdm(range(N_DAYS)):
        action_idx = get_epsilon_greedy_action(state, mle_mdp, epsilon=0.9)
        price_change = ACTIONS[action_idx]
        curr_price += price_change
        next_state = env.get_n_people_bought(curr_price)
        reward = env.get_profit(curr_price)

        mle_mdp.update_counts(state, action_idx, reward, next_state)
        mle_mdp.update_probs_reward()
        mle_mdp.value_iteration(tol=1e-2)

        state = next_state
        state_trajectory.append(state)
        price_trajectory.append(curr_price)

    print("Final State: ", state)
    print("Final Price: ", curr_price)

    plt.plot(np.arange(N_PEOPLE + 1), mle_mdp.value)
    plt.savefig("utility.pdf")
    plt.show()

    plt.plot(np.arange(N_DAYS + 1), state_trajectory)
    plt.hlines([opt_n_bought], xmin=0, xmax=365, color='red', linestyle='dashed', label="Optimal")
    plt.savefig("states.pdf")
    plt.show()

    plt.plot(np.arange(N_DAYS + 1), price_trajectory)
    plt.hlines([opt_price], xmin=0, xmax=365, color='red', linestyle='dashed', label="Optimal")
    plt.savefig("prices.pdf")
    plt.show()

def get_epsilon_greedy_action(state, mdp, epsilon=0.9):
    if np.random.choice([False, True], p=[1 - epsilon, epsilon]):
        expected_rewards = mdp.transition_probs[state].T @ mdp.value
        return np.random.choice(np.argwhere(expected_rewards == expected_rewards.max()).T[0])
    return int(np.random.choice(np.arange(mdp.n_actions)))


if __name__ == "__main__":
    main()