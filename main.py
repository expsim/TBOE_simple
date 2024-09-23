import numpy as np
import torch
from octopus_env import OctopusEnv
from octopus import Octopus
from random_agent import RandomOctopus
import matplotlib.pyplot as plt


def evaluate_agent(env, agent, num_episodes):
    total_rewards = []
    success_rates = []
    steps_to_goal = []

    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            actions = agent.select_action(state)
            next_state, reward, done, _ = env.step(actions)
            state = next_state
            total_reward += reward
            steps += 1

        total_rewards.append(total_reward)
        success_rates.append(1 if env.is_food_reached() else 0)
        if env.is_food_reached():
            steps_to_goal.append(steps)

    avg_reward = np.mean(total_rewards)
    success_rate = np.mean(success_rates)
    avg_steps = np.mean(steps_to_goal) if steps_to_goal else float('inf')

    return avg_reward, success_rate, avg_steps


def train_and_compare(num_episodes, batch_size, update_target_every, eval_interval):
    env = OctopusEnv()
    state_dim = env._get_obs().shape[0]
    action_dim = 5  # R, N, S, W, E

    octopus = Octopus(state_dim=state_dim, action_dim=action_dim,
                        epsilon_start=1.0, epsilon_end=0.01, epsilon_decay_steps=num_episodes * 10)
    random_octopus = RandomOctopus(action_dim)

    episode_rewards = []
    success_rates = []
    steps_to_goal = []

    octopus_eval_rewards = []
    octopus_eval_success_rates = []
    octopus_eval_steps = []
    random_eval_rewards = []
    random_eval_success_rates = []
    random_eval_steps = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            actions = octopus.select_action(state)
            next_state, reward, done, _ = env.step(actions)
            octopus.remember(state, actions, reward, next_state, done)
            octopus.train(batch_size)
            state = next_state
            total_reward += reward
            steps += 1

        if episode % update_target_every == 0:
            octopus.update_target_network()

        episode_rewards.append(total_reward)
        success_rates.append(1 if env.is_food_reached() else 0)
        if env.is_food_reached():
            steps_to_goal.append(steps)

        if episode % eval_interval == 0:
            octopus_reward, octopus_success, octopus_steps = evaluate_agent(env, octopus, 100)
            random_reward, random_success, random_steps = evaluate_agent(env, random_octopus, 100)

            octopus_eval_rewards.append(octopus_reward)
            octopus_eval_success_rates.append(octopus_success)
            octopus_eval_steps.append(octopus_steps)
            random_eval_rewards.append(random_reward)
            random_eval_success_rates.append(random_success)
            random_eval_steps.append(random_steps)

            print(f"Episode {episode}")
            print(
                f"Octopus - Avg Reward: {octopus_reward:.2f}, Success Rate: {octopus_success:.2f}, Avg Steps: {octopus_steps:.2f}")
            print(
                f"Random Octopus - Avg Reward: {random_reward:.2f}, Success Rate: {random_success:.2f}, Avg Steps: {random_steps:.2f}")
            print(f"DQN Epsilon: {octopus.epsilon:.2f}")
            print("---")

    # Save the trained model
    torch.save(octopus.q_network.state_dict(), 'multi_agent_dqn.pth')

    return (episode_rewards, success_rates, steps_to_goal,
            octopus_eval_rewards, octopus_eval_success_rates, octopus_eval_steps,
            random_eval_rewards, random_eval_success_rates, random_eval_steps)


def plot_results(episode_rewards, success_rates, steps_to_goal,
                 octopus_eval_rewards, octopus_eval_success_rates, octopus_eval_steps,
                 random_eval_rewards, random_eval_success_rates, random_eval_steps):
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))

    # Plot episode rewards
    axes[0, 0].plot(episode_rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')

    # Plot success rates
    window_size = 100
    moving_avg = np.convolve(success_rates, np.ones(window_size) / window_size, mode='valid')
    axes[0, 1].plot(moving_avg)
    axes[0, 1].set_title('Success Rate (Moving Average)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Success Rate')
    axes[0, 1].set_ylim([0, 1])

    # Plot steps to goal
    axes[1, 0].plot(steps_to_goal)
    axes[1, 0].set_title('Steps to Reach food')
    axes[1, 0].set_xlabel('Successful Episode')
    axes[1, 0].set_ylabel('Steps')

    # Plot evaluation rewards
    axes[1, 1].plot(octopus_eval_rewards, label='DQN Agent')
    axes[1, 1].plot(random_eval_rewards, label='Random Agent')
    axes[1, 1].set_title('Evaluation Rewards')
    axes[1, 1].set_xlabel('Evaluation')
    axes[1, 1].set_ylabel('Average Reward')
    axes[1, 1].legend()

    # Plot evaluation success rates
    axes[2, 0].plot(octopus_eval_success_rates, label='DQN Agent')
    axes[2, 0].plot(random_eval_success_rates, label='Random Agent')
    axes[2, 0].set_title('Evaluation Success Rates')
    axes[2, 0].set_xlabel('Evaluation')
    axes[2, 0].set_ylabel('Success Rate')
    axes[2, 0].set_ylim([0, 1])
    axes[2, 0].legend()

    # Plot evaluation steps
    axes[2, 1].plot(octopus_eval_steps, label='DQN Agent')
    axes[2, 1].plot(random_eval_steps, label='Random Agent')
    axes[2, 1].set_title('Evaluation Steps to Reach food')
    axes[2, 1].set_xlabel('Evaluation')
    axes[2, 1].set_ylabel('Average Steps')
    axes[2, 1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    num_episodes = 5000
    batch_size = 64
    update_target_every = 10
    eval_interval = 100

    results = train_and_compare(num_episodes, batch_size, update_target_every, eval_interval)
    plot_results(*results)

    print("Training completed. Model saved as 'multi_agent_dqn.pth'")
    print(f"Final Octopus Average Reward (last evaluation): {results[3][-1]:.2f}")
    print(f"Final Octopus Success Rate (last evaluation): {results[4][-1]:.2f}")
    print(f"Final octopus Average Steps to Goal (last evaluation): {results[5][-1]:.2f}")
    print(f"Final Random Average Reward (last evaluation): {results[6][-1]:.2f}")
    print(f"Final Random Success Rate (last evaluation): {results[7][-1]:.2f}")
    print(f"Final Random Average Steps to Goal (last evaluation): {results[8][-1]:.2f}")