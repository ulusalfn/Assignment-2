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
        #print("Alpha is ", alpha)

        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        action = None
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.Q[state])
        return action
        
    def update(self, state, action, reward, next_state, done): # Augment arguments if necessary
        if done:
            expected_Q = 0 # no future rewards if terminal state
        else:
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

    def __init__(self, n_actions, n_states, n, alpha, epsilon=0.1, gamma=1.0):
        self.n_actions = n_actions
        self.n_states = n_states
        self.n = n
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)  # Explore
        return np.argmax(self.Q[state])  # Exploit

    def update(self, states, actions, rewards, done, next_state, next_action):
        G = sum(self.gamma**t * rewards[t] for t in range(len(rewards)))
        if not done:
            G += (self.gamma**self.n) * self.Q[next_state, next_action]
        self.Q[states[0], actions[0]] += self.alpha * (G - self.Q[states[0], actions[0]])
        #print(self.Q[states[0], actions[0]])

    def train(self, n_episodes, env):
        episode_returns = []

        for episode in range(n_episodes):
            #if episode % 100 == 0:
                #print(f"Episode {episode}")
            state = env.reset()
            action = self.select_action(state)
            
            episode_return = 0

            states = [state]
            actions = [action]
            rewards = []

            done = False
            #i = 0
            while not done:
                #if i % 100 == 0:
                    #print(f"Step {i}")
                #i += 1
                reward = env.step(action)
                done = env.done()
                next_state = env.state()
                next_action = self.select_action(next_state)

                rewards.append(reward)
                states.append(next_state)
                actions.append(next_action)
                episode_return += reward

                if len(rewards) >= self.n:
                    self.update(states[:self.n], actions[:self.n], rewards[:self.n], done, next_state, next_action)
                    states.pop(0)
                    actions.pop(0)
                    rewards.pop(0)

                state = next_state
                action = next_action

            while rewards:
                self.update(states, actions, rewards, done, next_state, next_action)
                states.pop(0)
                actions.pop(0)
                rewards.pop(0)

            episode_returns.append(episode_return)

        #print(f"Episode {episode + 1}/{n_episodes}, Return: {episode_return}")
        return episode_returns
