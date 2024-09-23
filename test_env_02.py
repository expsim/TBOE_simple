import numpy as np
import torch
from octopus_env import OctopusEnv
from octopus import MultiAgentDQN, Octopus
import json


class RandomAgent:
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def select_action(self, state):
        return [np.random.randint(self.action_dim) for _ in range(4)]

def load_trained_model(model_path, state_dim, action_dim):
    model = MultiAgentDQN(state_dim, action_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def evaluate_agent_with_replay(agent, env, num_episodes, save_replay=False):
    total_rewards = []
    steps_to_goal = []
    success_count = 0
    replay_data = None

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False
        episode_data = []

        while not done:
            if isinstance(agent, RandomAgent):
                actions = agent.select_action(state)
            else:
                actions = agent.select_action(state)
            next_state, reward, done, _ = env.step(actions)

            if save_replay and replay_data is None:
                episode_data.append({
                    'state': state.tolist(),
                    'actions': actions,
                    'reward': reward,
                    'next_state': next_state.tolist(),
                    'done': done,
                    'food_pos': env.octopus_pos.tolist(),
                    'octopus_pos': env.food_pos.tolist(),
                    'energy': env.energy
                })

            state = next_state
            total_reward += reward
            steps += 1

        total_rewards.append(total_reward)
        if env.is_food_reached():
            steps_to_goal.append(steps)
            success_count += 1

        if save_replay and replay_data is None and env.is_food_reached():
            replay_data = {
                'grid_size': env.size,
                'max_energy': env.max_energy,
                'episode_data': episode_data
            }

    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(steps_to_goal) if steps_to_goal else float('inf')
    success_rate = success_count / num_episodes

    return avg_reward, avg_steps, success_rate, replay_data


def run_evaluation(model_path, grid_sizes, energy_amounts, num_episodes=100):
    results = {}
    replay_data = None

    for grid_size in grid_sizes:
        for energy in energy_amounts:
            env = OctopusEnv(size=grid_size, max_energy=energy)
            state_dim = env._get_obs().shape[0]
            action_dim = 5  # R, N, S, W, E

            # DQN Agent
            dqn_model = load_trained_model(model_path, state_dim, action_dim)
            dqn_agent = Octopus(state_dim, action_dim)
            dqn_agent.q_network = dqn_model
            dqn_agent.epsilon = 0  # No exploration during evaluation

            # Random Agent
            random_agent = RandomAgent(action_dim)

            # Evaluate DQN Agent
            dqn_reward, dqn_steps, dqn_success, dqn_replay = evaluate_agent_with_replay(dqn_agent, env, num_episodes,
                                                                                        save_replay=True)
            if dqn_replay and replay_data is None:
                replay_data = dqn_replay

            # Evaluate Random Agent
            random_reward, random_steps, random_success, _ = evaluate_agent_with_replay(random_agent, env, num_episodes)

            results[(grid_size, energy)] = {
                'DQN': {'reward': dqn_reward, 'steps': dqn_steps, 'success_rate': dqn_success},
                'Random': {'reward': random_reward, 'steps': random_steps, 'success_rate': random_success}
            }

            print(f"\nGrid Size: {grid_size}x{grid_size}, energy: {energy}")
            print("Octopus:")
            print(f"  Avg Reward: {dqn_reward:.2f}")
            print(f"  Avg Steps: {dqn_steps:.2f}")
            print(f"  Success Rate: {dqn_success:.2f}")
            print("Random Octopus:")
            print(f"  Avg Reward: {random_reward:.2f}")
            print(f"  Avg Steps: {random_steps:.2f}")
            print(f"  Success Rate: {random_success:.2f}")

    return results, replay_data


if __name__ == "__main__":
    model_path = 'multi_agent_dqn.pth'
    grid_sizes = [10]
    energy_amounts = [20]
    num_episodes = 100  # Number of episodes for each evaluation

    results, replay_data = run_evaluation(model_path, grid_sizes, energy_amounts, num_episodes)

    if replay_data:
        with open('replay_data.json', 'w') as f:
            json.dump(replay_data, f)
        print("Replay data saved to 'replay_data.json'")

    print("\nEvaluation Complete.")