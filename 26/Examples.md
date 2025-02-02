Implement a practical Reinforcement Learning example: Training an AI to play a simple game of "Catch the Apple." This example demonstrates real-world RL principles in a visual and engaging way.

```python
import numpy as np
import pygame
import random
import time

# Initialize Pygame
pygame.init()

# Constants
WINDOW_SIZE = 400
PLAYER_SIZE = 50
APPLE_SIZE = 20
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

class CatchGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.player_x = WINDOW_SIZE // 2
        self.apple_x = random.randint(0, WINDOW_SIZE - APPLE_SIZE)
        self.apple_y = 0
        self.score = 0
        return self._get_state()

    def _get_state(self):
        # Discretize state space for simplicity
        player_pos = self.player_x // 50
        apple_pos = self.apple_x // 50
        apple_height = self.apple_y // 50
        return (player_pos, apple_pos, apple_height)

    def step(self, action):
        # Action: 0 = left, 1 = stay, 2 = right
        if action == 0:
            self.player_x = max(0, self.player_x - 10)
        elif action == 2:
            self.player_x = min(WINDOW_SIZE - PLAYER_SIZE, self.player_x + 10)

        # Move apple down
        self.apple_y += 5

        # Check if apple is caught
        reward = 0
        done = False

        if self.apple_y + APPLE_SIZE >= WINDOW_SIZE - PLAYER_SIZE:
            if (self.player_x < self.apple_x + APPLE_SIZE and
                self.player_x + PLAYER_SIZE > self.apple_x):
                reward = 1
                self.score += 1
            else:
                reward = -1

            self.apple_y = 0
            self.apple_x = random.randint(0, WINDOW_SIZE - APPLE_SIZE)

        return self._get_state(), reward, done

    def render(self):
        self.screen.fill(WHITE)

        # Draw player
        pygame.draw.rect(self.screen, BLUE,
                        (self.player_x, WINDOW_SIZE - PLAYER_SIZE, PLAYER_SIZE, PLAYER_SIZE))

        # Draw apple
        pygame.draw.rect(self.screen, RED,
                        (self.apple_x, self.apple_y, APPLE_SIZE, APPLE_SIZE))

        # Display score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {self.score}', True, BLACK)
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(FPS)

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.q_table = {}
        self.lr = 0.1
        self.gamma = 0.95
        self.epsilon = 0.1
        self.action_size = action_size

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)

        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.action_size)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.action_size)

        # Q-learning update formula
        self.q_table[state][action] += self.lr * (
            reward + self.gamma * np.max(self.q_table[next_state]) -
            self.q_table[state][action]
        )

# Training loop
env = CatchGame()
agent = QLearningAgent(state_size=1000, action_size=3)  # 3 actions: left, stay, right

episodes = 1000
try:
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            # Get action
            action = agent.get_action(state)

            # Take action
            next_state, reward, done = env.step(action)

            # Update agent
            agent.update(state, action, reward, next_state)

            episode_reward += reward
            state = next_state

            # Render game
            env.render()

            # Check for quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            if episode_reward < -10:  # End episode if too many misses
                break

        if episode % 10 == 0:
            print(f"Episode: {episode}, Score: {env.score}")
            agent.epsilon = max(0.01, agent.epsilon * 0.995)  # Decay exploration

except KeyboardInterrupt:
    print("Training interrupted!")

pygame.quit()
```

This example demonstrates several real-world RL concepts:

1. **Visual Environment**

   - A graphical game where an agent (blue paddle) tries to catch falling apples (red squares)
   - Similar to robotics tasks where visual input guides actions

2. **Continuous Action Space**

   - The agent can move left, right, or stay still
   - Maps to real-world control problems like robot movement

3. **State Management**

   - Tracks position of player and apple
   - Handles discrete state space for learning

4. **Reward System**

   - Positive reward for catching apples
   - Negative reward for missing
   - Similar to real-world task success/failure metrics

5. **Learning Process**
   - Q-learning implementation with exploration/exploitation balance
   - Epsilon decay for gradually reducing random actions
   - Performance improvement over time

To run this example:

1. Make sure you have pygame installed (`pip install pygame`)
2. Run the code
3. Watch the AI learn to catch apples
4. Close the window to stop training

The agent will gradually improve its performance by:

- Learning to position itself under falling apples
- Predicting apple trajectories
- Maximizing its score over time

This demonstrates core RL principles used in more complex applications like:

- Industrial robotics
- Autonomous vehicles
- Game AI
- Resource management systems

The learning process is visible and intuitive, making it a good example for understanding how RL works in practice.
