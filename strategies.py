import karantools as kt
import bandits
import random
import numpy as np
import functools
from matplotlib import pyplot as plt

NUM_PULLS = 1000
NUM_ARMS = 100

NUM_TRIALS = 200

def random_pulls_strategy(bandit, num_pulls):
	reward = 0.0

	for _ in range(num_pulls):
		arm = random.randrange(len(bandit.arms))
		reward += bandit.pull(arm)

	return reward

# Randomly explores for the first epislon portion of the pulls, then repeatedly
# pulls the arm with the highest average payoff at that point.
def explore_then_exploit_strategy(bandit, num_pulls, epsilon=.3):
	reward = 0.0

	arm_total_payoffs = np.zeros(len(bandit.arms))
	arm_total_pulls = np.ones(len(bandit.arms))

	explore_pulls = int(num_pulls * epsilon)
	exploit_pulls = num_pulls - explore_pulls

	for _ in range(explore_pulls):
		arm = random.randrange(len(bandit.arms))
		curr_reward = bandit.pull(arm)
		reward += curr_reward

		arm_total_payoffs[arm] += curr_reward
		arm_total_pulls[arm] += 1

	best_arm = np.argmax(arm_total_payoffs / arm_total_pulls)

	for _ in range(exploit_pulls):
		reward += bandit.pull(best_arm)

	return reward

strategies = {
	'Random pulls': random_pulls_strategy,
	'Explore then exploit basic epsilon .1': functools.partial(explore_then_exploit_strategy, epsilon=.1),
	'Explore then exploit basic epsilon .3': explore_then_exploit_strategy,
	'Explore then exploit basic epsilon .5': functools.partial(explore_then_exploit_strategy, epsilon=.5),
}

for strategy_desc, strategy in strategies.items():

	avg_streamer = kt.AverageStreamer()
	total_rewards = np.zeros(NUM_PULLS)
	for _ in range(NUM_TRIALS):
		bandit = bandits.get_random_multi_armed_bernoulli_bandit(NUM_ARMS)
		reward = strategy(bandit, NUM_PULLS)
		avg_streamer.add(reward)
		rewards = bandit.get_rewards()
		total_rewards += rewards

	plt.plot(total_rewards / NUM_TRIALS)
	plt.title(strategy_desc)
	plt.show()

	kt.print_header_block('Running strategy: ' + strategy_desc + '    Result: ' + str(avg_streamer.query()))
