import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from ShortCutEnvironment import ShortcutEnvironment, WindyShortcutEnvironment
import ShortCutAgents


def run_repititions(agent, n_reps, n_episodes, alpha_values, epsilon, env):
    """
    Run the agent for n_reps repetitions in the environment env.
    """
    for alpha in alpha_values:
        rewards = np.zeros((n_reps, n_episodes))
        agente = agent
        for rep in range(n_reps):
            print(f"Running repetition {rep+1} of {n_reps} for alpha {alpha}")
            print(f"Agent: {agent}")
            if agent == "QLearningAgent": 
                current_agent = ShortCutAgents.QLearningAgent(env.action_size(), env.state_size(), epsilon, alpha)
                name = "Q-learning"
            elif agent == "SARSAAgent":
                current_agent = ShortCutAgents.SARSAAgent(env.action_size(), env.state_size(), epsilon, alpha)
                name = "SARSA"
            elif agent == "ExpectedSARSAAgent":
                current_agent = ShortCutAgents.ExpectedSARSAAgent(env.action_size(), env.state_size(), epsilon, alpha)
                name = "Expected SARSA"
            elif agent == "nStepSARSAAgent":
                current_agent = ShortCutAgents.nStepSARSAAgent(env.action_size(), env.state_size(), epsilon, alpha)
                name = "n-step SARSA"
            else:
                raise ValueError("Unknown agent type")
            episode_returns = current_agent.train(n_episodes, env)
            rewards[rep] = episode_returns

            final_agent = current_agent
            final_env = env
        
        mean_rewards = np.mean(rewards, axis=0)
        plt.plot(smooth(mean_rewards, 31), label=f"alpha={alpha}")
        final_env.render_greedy(final_agent.Q)

    if final_env == WindyShortcutEnvironment():
        plt.title(f"Learning curve for {name} in Windy environment")
        plt.xlabel("Episodes")
        plt.ylabel("Mean reward")
        plt.legend()
        plt.savefig(f"LearningCurves_{name}_Windy.png")
        plt.show()
    else:
        plt.title(f"Learning curves for {name}")
        plt.xlabel("Episodes")
        plt.ylabel("Mean reward")
        plt.legend()
        plt.savefig(f"LearningCurves_{name}.png")
        plt.show()    


def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed 
    window: size of the smoothing window '''
    return savgol_filter(y,window,poly)


if __name__ == "__main__":
    #greedy policies
    #run_repititions(agent = "QLearningAgent", n_reps = 1, n_episodes = 10000, alpha_values =[0.1], epsilon = 0.1, env = ShortcutEnvironment())
    #run_repititions(agent = "SARSAAgent", n_reps = 1, n_episodes = 10000, alpha_values =[0.1], epsilon = 0.1, env = ShortcutEnvironment())
    #run_repititions(agent = "ExpectedSARSAAgent", n_reps = 1, n_episodes = 10000, alpha_values =[0.1], epsilon = 0.1, env = ShortcutEnvironment())
    #run_repititions(agent = "nStepSARSAAgent", n_reps = 1, n_episodes = 10000, alpha_values =[0.1], epsilon = 0.1, env = ShortcutEnvironment())
    

    #learning curves
    #run_repititions(agent = "QLearningAgent", n_reps = 100, n_episodes = 1000, alpha_values =[0.01, 0.1, 0.5, 0.9], epsilon = 0.1, env = ShortcutEnvironment())
    #run_repititions(agent = "SARSAAgent", n_reps = 100, n_episodes = 1000, alpha_values =[0.01, 0.1, 0.5, 0.9], epsilon = 0.1, env = ShortcutEnvironment())
    run_repititions(agent = "ExpectedSARSAAgent", n_reps = 1, n_episodes = 10000, alpha_values =[0.1], epsilon = 0.1, env = ShortcutEnvironment())
    #run_repititions(agent = "nStepSARSAAgent", n_reps = 100, n_episodes = 1000, alpha_values =[0.01, 0.1, 0.5, 0.9], epsilon = 0.1, env = ShortcutEnvironment())

    #Windy environment
    #run_repititions(agent = "QLearningAgent", n_reps = 1, n_episodes = 10000, alpha_values =[0.1], epsilon = 0.1, env = WindyShortcutEnvironment())
    #run_repititions(agent = "SARSAAgent", n_reps = 1, n_episodes = 10000, alpha_values =[0.1], epsilon = 0.1, env = WindyShortcutEnvironment())
