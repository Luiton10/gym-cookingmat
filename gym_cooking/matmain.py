import gym
from gym_cooking.utils.mat_algorithm.ma_transformer import MultiAgentTransformer

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
import os
from gym_cooking.utils.mat_algorithm.ma_transformer import MultiAgentTransformer


import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class MATAgent:
    """Wrapper for Multi-Agent Transformer to interact with gym_cooking levels."""
    def __init__(self, num_agents, obs_dim, act_dim, lr=1e-4):
        self.num_agents = num_agents
        self.mat = MultiAgentTransformer(num_agents=num_agents, obs_dim=obs_dim, act_dim=act_dim)
        self.optimizer = optim.Adam(self.mat.parameters(), lr=lr)

    def select_actions(self, observations):
        """Encode observations and decode actions."""
        encoded_obs = self.mat.encode(observations)
        actions = self.mat.decode(encoded_obs)
        return actions

    def train(self, observations, actions, rewards):
        """Train MAT using observations, actions, and rewards."""
        loss = self.mat.compute_loss(observations, actions, rewards)  # Adjust with MAT's loss logic
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

def main():
    # Environment setup
    env_id = "gym_cooking:overcookedEnv-v0"
    env = gym.make(env_id)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    num_agents = 2  # Adjust based on the task

    # Initialize MAT agent
    agent = MATAgent(num_agents=num_agents, obs_dim=obs_dim, act_dim=act_dim)

    # Training parameters
    num_episodes = 100
    max_timesteps = 100
    reward_history = []

    for episode in range(num_episodes):
        observations = env.reset()
        total_reward = 0

        for t in range(max_timesteps):
            # Select actions
            actions = agent.select_actions(observations)

            # Step the environment
            next_observations, rewards, done, info = env.step(actions)
            total_reward += sum(rewards)

            # Train MAT
            loss = agent.train(observations, actions, rewards)

            observations = next_observations

            if done:
                break

        reward_history.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}, Loss: {loss:.4f}")

    # Visualization
    plt.plot(range(num_episodes), reward_history)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()

    # Save MAT model
    torch.save(agent.mat.state_dict(), "mat_model.pth")
    print("Model saved as mat_model.pth")

if __name__ == "__main__":
    main()
