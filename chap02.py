import numpy as np


class Bandit:

    def __init__(self, k=10):
        self.k = k
        self.options = np.arange(k)
        self.q_ast = np.random.normal(0, 1, k)
        self.q_opt = np.argmax(self.q_ast)

    def get_reward(self, action: int) -> float:
        return self.q_ast[action] + np.random.normal(0, 0.1)

    def get_action(self, threshold: float, *args):
        e = np.random.uniform()
        if e > threshold:
            self.update_estimates(*args)
            return np.argmax(self.estimates)
        else:
            i = np.random.randint(0, self.k)
            return self.options[i]


class BanditActionValue(Bandit):

    def update_estimates(self, action, actions, rewards):
        assert len(actions) == len(rewards)
        indices, = np.where(actions == action)
        n = len(indices)
        if n != 0:
            v = rewards[indices].sum()
            self.estimates[action] = v / n

    def run(self, num_runs=100, num_steps=1000, threshold=0.1):
        avg_actions = np.zeros((num_steps,))  # avg through runs
        avg_rewards = np.zeros((num_steps,))  # avg through runs
        for _ in range(num_runs):
            actions = np.zeros((num_steps,))
            rewards = np.zeros((num_steps,))
            self.estimates = np.zeros((self.k,))
            action = np.random.randint(0, self.k)
            reward = self.get_reward(action)
            for i in range(num_steps):
                action = self.get_action(threshold, action, actions, rewards)
                reward = self.get_reward(action)
                actions[i] = action
                rewards[i] = reward
                avg_actions[i] += 1 / num_runs if action == self.q_opt else 0
                avg_rewards[i] += reward / num_runs
        return avg_actions, avg_rewards


class BanditIncremental(Bandit):

    def update_estimates(self, action, reward):
        self.step_sizes[action] += 1
        t = self.step_sizes[action]
        a = 0 if t == 0 else 1 / t
        self.estimates[action] = self.estimates[action] + \
            a * (reward - self.estimates[action])

    def run(self, num_runs=100, num_steps=1000, threshold=0.1):
        avg_actions = np.zeros((num_steps,))  # avg through runs
        avg_rewards = np.zeros((num_steps,))  # avg through runs
        for _ in range(num_runs):
            self.step_sizes = np.zeros((self.k,))
            self.estimates = np.zeros((self.k,))
            action = np.random.randint(0, self.k)
            reward = self.get_reward(action)
            for i in range(num_steps):
                action = self.get_action(threshold, action, reward)
                reward = self.get_reward(action)
                avg_actions[i] += 1 / num_runs if action == self.q_opt else 0
                avg_rewards[i] += reward / num_runs
        return avg_actions, avg_rewards
