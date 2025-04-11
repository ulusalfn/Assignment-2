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

    def __init__(self, n_actions, n_states, epsilon=0.1, alpha=0.1, gamma=0.5, n=0):
        
        self.n_actions = n_actions
        self.n_states = n_states
        self.n = n

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.Q = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        # TO DO: Implement policy
        action = None
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions) # Explore
        else:
            action = np.argmax(self.Q[state]) # Exploit
        
        return action
        
    def update(self, states, actions, rewards, done): # Augment arguments if necessary
        # TO DO: Implement n-step SARSA update
        G = 0
        #print(rewards)
        for t in range(len(rewards)):
            #print("t: ", t)
            G += self.gamma**t * rewards[t]
            #print("G before boot =" ,G)

        if not done and len(actions) == self.n + 1:
            #and len(states) == self.n and len(actions) == self.n + 1:
            
            G += (self.gamma**self.n) * self.Q[states[-1], actions[-1]]
            #print("G: ", G)

        self.Q[states[0], actions[0]] = np.clip(self.Q[states[0], actions[0]] + self.alpha * (G - self.Q[states[0], actions[0]]), -100, 100)
        #print("Q: ", self.Q[states[0], actions[0]])    
    
    def train(self, n_episodes, env):
        # TO DO: Implement the agent loop that trains for n_episodes. 
        # Return a vector with the the cumulative reward (=return) per episode
        episode_returns = []
        self.n = 3

        for episode in range(n_episodes):
            print(episode)
            state = env.reset()
            done = False
            episode_return = 0
            self.states = []
            self.actions = []
            self.rewards = []

            while not done:
                action = self.select_action(state)
                self.actions.append(action)

                reward = env.step(action)
                next_state = env.state()
                done = env.done()
                self.states.append(state)
                self.rewards.append(reward)

                if len(self.states) > self.n:
                    #print("Updating ", x)
                    self.update(self.states[:self.n], self.actions[:self.n], self.rewards[:self.n], done)

                    self.states.pop(0)
                    self.actions.pop(0)
                    self.rewards.pop(0)

                state = next_state
                episode_return += reward

            while len(self.states) > 0:

                n_steps = len(self.rewards)
                self.update(self.states[:n_steps], self.actions[:n_steps], self.rewards[:n_steps], done)

                self.states.pop(0)
                self.actions.pop(0)
                self.rewards.pop(0)
            

            #print("Episode: ", episode, "Return: ", episode_return)
            episode_returns.append(episode_return)
            #print(episode_return)
            
        return episode_returns  
    
    
    