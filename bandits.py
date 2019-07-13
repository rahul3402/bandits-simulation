import random
import math
import numpy as np

class SingleArmedBernoulliBandit(object):
	def __init__(self, prob):
		self.prob = prob

	def pull(self):
		if random.random() < self.prob:
			return 1
		else:
			return 0

class MultiArmedBernoulliBandit(object):
	def __init__(self, probs):
		self.arms = [SingleArmedBernoulliBandit(prob) for prob in probs]
		self.total_pulls = 0
		self.rewards = []

	def pull(self, i):
		self.total_pulls += 1

		reward = self.arms[i].pull()
		self.rewards.append(reward)
		return reward

	def get_rewards(self):
		return np.array(self.rewards)


def get_random_multi_armed_bernoulli_bandit(num_arms):
	probs = [random.random() for _ in range(num_arms)]
	return MultiArmedBernoulliBandit(probs)
