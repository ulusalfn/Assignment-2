import random
import numpy as np

class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        
        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        action = None
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.Q[state])
        return action
        
    def update(self, state, action, reward, next_state, done):
        target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

    def train(self, n_episodes, env):
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            episode_return = 0
            while not done:
                action = self.select_action(state)
                reward = env.step(action)
                next_state = env.state()
                done = env.done()
                self.update(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
            episode_returns.append(episode_return)
        return episode_returns

class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state):
        action = None
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.Q[state])
        return action
        
    def update(self, state, action, reward, next_state, next_action, done):
        target = reward + self.gamma * self.Q[next_state, next_action]
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])


    def train(self, n_episodes, env):
        episode_returns = []
        for episode in range(n_episodes):
            state = env.reset()
            action = self.select_action(state)
            done = False
            episode_return = 0
            while not done:
                reward = env.step(action)
                next_state = env.state()
                done = env.done()
                next_action = self.select_action(next_state)
                self.update(state, action, reward, next_state, next_action, done)
                state = next_state
                action = next_action
                episode_return += reward
            episode_returns.append(episode_return)
        return episode_returns


class ExpectedSARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        action = None
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.Q[state])
        return action
        
    def update(self, state, action, reward, next_state, done): # Augment arguments if necessary
        policy_probs = np.ones(self.n_actions) * self.epsilon / self.n_actions # exploration distribution
        policy_probs[np.argmax(self.Q[next_state])] += 1 - self.epsilon # greedy action
        expected_Q= np.sum(policy_probs * self.Q[next_state])
        # update Q
        target = reward + self.gamma * expected_Q
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])
        # TO DO: Implement Expected SARSA update

    def train(self, n_episodes, env):
        episode_returns = []
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            episode_return = 0
            while not done:
                action = self.select_action(state)
                reward = env.step(action)
                next_state = env.state()
                done = env.done()
                self.update(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
            episode_returns.append(episode_return)
        return episode_returns    


class nStepSARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=0.5, n=3):
        
        self.n_actions = n_actions
        self.n_states = n_states
        self.n = n

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        action = None
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.Q[state])
        return action
        
    def update(self, states, actions, rewards, done): 
        G = 0
        for t in range(len(rewards)):
            G += self.gamma**t * rewards[t]

        if not done and len(actions) > self.n:
            bootstrapped = (self.gamma**self.n) * self.Q[states[-1], actions[-1]]
            G += bootstrapped
            print(f"[update] Bootstrapped Q[{states[-1]}, {actions[-1]}]: {bootstrapped}")

        target = G
        old_q = self.Q[states[0], actions[0]]
        new_q = old_q + self.alpha * (target - old_q)

        print(f"[update] G: {G}, Q[{states[0]}, {actions[0]}] old: {old_q}, new: {new_q}")

        self.Q[states[0], actions[0]] = new_q

    def train(self, n_episodes, env):
        episode_returns = []

        for episode in range(n_episodes):
            state = env.reset()
            done = False
            episode_return = 0
            states = []
            actions = []
            rewards = []

            # Take n+1 actions before the first update
            for _ in range(self.n + 1):
                if not done:
                    action = self.select_action(state)
                    reward = env.step(action)
                    next_state = env.state()
                    done = env.done()

                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)

                    state = next_state
                    episode_return += reward

            # Perform updates as the episode progresses
            while not done:
                action = self.select_action(state)
                reward = env.step(action)
                next_state = env.state()
                done = env.done()

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                self.update(states[:self.n], actions[:self.n], rewards[:self.n], done)

                states.pop(0)
                actions.pop(0)
                rewards.pop(0)

                state = next_state
                episode_return += reward

            # Perform updates for the remaining state-action pairs after the episode ends
            while len(states) > 0:
                n_steps = len(rewards)
                self.update(states[:n_steps], actions[:n_steps], rewards[:n_steps], done)
                states.pop(0)
                actions.pop(0)
                rewards.pop(0)

            episode_returns.append(episode_return)
            
        return episode_returns