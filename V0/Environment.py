import numpy as np 

'''
To use the DynamicPricingEnvironment:
    1) First, set the distribution of tau by calling one of the `set_..._distribution` functions
    2) At each time step, you can then either get an observation, profit, or number of people who bought the item
       by inputting the price.
'''
class DynamicPricingEnvironment:
    def __init__(self, n_people):
        self.n_people = n_people

    def set_random_distribution(self, distrib_type, low, high):
        if distrib_type == "Unimodal Gaussian":
            mean = np.random.uniform(low=round(((high - low)/4) + low), high=round((3*(high - low)/4) + low))
            std = np.random.uniform(low=0.5, high=3)
            self.set_gaussian_distribution(mean, std)
        elif distrib_type == "Multimodal Gaussian":
            n_modes = np.random.randint(low=2, high=5)
            means = np.random.uniform(low=round(((high - low)/4) + low), high=round((3*(high - low)/4) + low), size=n_modes)
            stds = np.random.uniform(low=0.5, high=3, size=n_modes)
            pcnts = np.random.randint(low=1, high=11, size=n_modes)
            pcnts = pcnts / pcnts.sum()
            self.set_multimodal_gaussian_distribution(n_modes, means, stds, pcnts)
        else: # Uniform Distribution
            self.set_uniform_distribution(low, high)

    def set_gaussian_distribution(self, mean, std):
        self.tau = np.sort(np.random.normal(mean, std, size=self.n_people))
        self._enforce_nonnegative()

    def set_multimodal_gaussian_distribution(self, n_modes, means, stds, pcnts):
        self.tau = np.concatenate([np.random.normal(means[i], stds[i], size=int(pcnts[i] * self.n_people)) for i in range(n_modes)])
        self.tau = np.sort(self.tau)
        self._enforce_nonnegative()
    
    def _enforce_nonnegative(self):
        self.tau = np.where(self.tau >= 0, self.tau, 0)

    def set_uniform_distribution(self, low, high):
        self.tau = np.random.uniform(low, high, size=self.n_people)

    def get_profit(self, price):
        return price * self.get_n_people_bought(price)

    def get_n_people_bought(self, price):
        return (self.get_observation(price)).sum()

    def get_observation(self, price):
        return self.tau >= price

    def get_optimal_price(self):
        profits = np.array([self.get_profit(self.tau[i]) for i in range(self.n_people)])
        return self.tau[np.argmax(profits)]
