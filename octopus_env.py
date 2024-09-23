import numpy as np


class OctopusEnv:
    def __init__(self, size=5, max_energy=20):
        self.size = size
        self.max_energy = max_energy
        self.action_map = ['R', 'N', 'S', 'W', 'E']
        self.step_count = 0
        self.reset()

    def reset(self):
        self.octopus_pos = np.random.randint(0, self.size, size=2)
        self.food_pos = self.octopus_pos
        while np.array_equal(self.food_pos, self.octopus_pos):
            self.food_pos = np.random.randint(0, self.size, size=2)
        self.energy = self.max_energy
        self.total_reward = 0
        self.previous_distance = self._calculate_distance()
        self.last_tactile_results = [0, 0, 0, 0]  # N, S, W, E
        self.step_count = 0
        return self._get_obs()

    def _calculate_distance(self):
        return np.linalg.norm(np.array(self.octopus_pos) - np.array(self.food_pos))


    def _move_food(self):
        direction = np.random.choice(['N', 'S', 'W', 'E'])
        if direction == 'N' and self.food_pos[0] > 0:
            self.food_pos[0] -= 1
        elif direction == 'S' and self.food_pos[0] < self.size - 1:
            self.food_pos[0] += 1
        elif direction == 'W' and self.food_pos[1] > 0:
            self.food_pos[1] -= 1
        elif direction == 'E' and self.food_pos[1] < self.size - 1:
            self.food_pos[1] += 1

    def step(self, actions):
        move_votes = {'N': 0, 'S': 0, 'W': 0, 'E': 0}
        tactile_perception_used = False

        for action_idx in actions:
            action = self.action_map[action_idx]
            if action == 'R':
                tactile_perception_used = True
            else:
                move_votes[action] += 1

        # Determine and apply movement
        vertical_movement = move_votes['S'] - move_votes['N']
        horizontal_movement = move_votes['E'] - move_votes['W']

        if abs(vertical_movement) + abs(horizontal_movement) >= 2:
            # Diagonal or double movement
            self.octopus_pos[0] = max(0, min(self.size - 1, self.octopus_pos[0] + np.sign(vertical_movement)))
            self.octopus_pos[1] = max(0, min(self.size - 1, self.octopus_pos[1] + np.sign(horizontal_movement)))
        elif vertical_movement != 0:
            # Vertical movement
            self.octopus_pos[0] = max(0, min(self.size - 1, self.octopus_pos[0] + np.sign(vertical_movement)))
        elif horizontal_movement != 0:
            # Horizontal movement
            self.octopus_pos[1] = max(0, min(self.size - 1, self.octopus_pos[1] + np.sign(horizontal_movement)))

        # Update tactile results after movement
        if tactile_perception_used:
            self.last_tactile_results = self._use_tactile_perception()
        else:
            self.last_tactile_results = [0, 0, 0, 0]

        self.energy -= 1
        self.step_count += 1

        # Move food every two steps
        if self.step_count % 2 == 0:
            self._move_food()

        done = self._is_done()
        reward = self._get_reward(done, tactile_perception_used, vertical_movement, horizontal_movement)
        self.total_reward += reward

        return self._get_obs(), reward, done, {}

    def _use_tactile_perception(self):
        x, y = self.octopus_pos
        fx, fy = self.food_pos
        return [
            int(fy < y),  # North
            int(fy > y),  # South
            int(fx < x),  # West
            int(fx > x)  # East
        ]

    def _is_done(self):
        return np.array_equal(self.octopus_pos, self.food_pos) or self.energy <= 0

    def _get_reward(self, done, tactile_perception_used, vertical_movement, horizontal_movement):
        if np.array_equal(self.octopus_pos, self.food_pos):
            return 100  # Large positive reward for reaching the food
        elif self.energy <= 0:
            return -100  # Large negative reward for running out of energy

        current_distance = self._calculate_distance()
        distance_reward = (self.previous_distance - current_distance) * 10
        self.previous_distance = current_distance

        step_penalty = -1
        tactile_reward = 2 if tactile_perception_used and any(self.last_tactile_results) else 0

        # Reward for coordinated movement
        movement_reward = 0
        if vertical_movement != 0 and horizontal_movement != 0:
            movement_reward = 5  # Reward for diagonal movement
        elif vertical_movement != 0 or horizontal_movement != 0:
            movement_reward = 2  # Reward for any movement

        return distance_reward + step_penalty + tactile_reward + movement_reward

    def _get_obs(self):
        octopus_pos = self.octopus_pos / (self.size - 1)  # Normalize position
        energy = self.energy / self.max_energy  # Normalize energy
        return np.concatenate([octopus_pos, [energy], self.last_tactile_results])

    def is_food_reached(self):
        return np.array_equal(self.octopus_pos, self.food_pos)