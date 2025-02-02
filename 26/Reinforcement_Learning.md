LETS understand Reinforcement Learning (RL) and implement a basic Q-learning algorithm. Let's break this down into digestible parts.

### 1. Introduction to Reinforcement Learning

Reinforcement Learning is a type of machine learning where an agent learns to make decisions by interacting with an environment. The key components are:

- **Agent**: The learner or decision-maker
- **Environment**: The world the agent interacts with
- **State**: Current situation of the agent
- **Action**: What the agent can do
- **Reward**: Feedback from the environment
- **Policy**: Strategy that the agent uses to determine actions

Here's a visual representation:

```
Agent → takes Action → Environment
Environment → gives State and Reward → Agent
```

### 2. Q-Learning Implementation

Let's implement Q-learning with a simple environment: A 4x4 grid world where an agent needs to find the optimal path to a goal.

```python
import numpy as np

class GridWorld:
    def __init__(self):
        self.grid_size = 4
        self.state = 0  # Start position
        self.goal = 15  # Goal position

    def get_possible_actions(self, state):
        actions = []
        # Can move right
        if (state + 1) % self.grid_size != 0:
            actions.append(0)  # Right
        # Can move left
        if state % self.grid_size != 0:
            actions.append(1)  # Left
        # Can move down
        if state + self.grid_size < self.grid_size * self.grid_size:
            actions.append(2)  # Down
        # Can move up
        if state - self.grid_size >= 0:
            actions.append(3)  # Up
        return actions

    def take_action(self, state, action):
        if action == 0:  # Right
            next_state = state + 1
        elif action == 1:  # Left
            next_state = state - 1
        elif action == 2:  # Down
            next_state = state + self.grid_size
        else:  # Up
            next_state = state - self.grid_size

        reward = 100 if next_state == self.goal else -1
        done = next_state == self.goal

        return next_state, reward, done

class QLearning:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95):
        self.q_table = np.zeros((state_size, action_size))
        self.lr = learning_rate
        self.gamma = discount_factor

    def get_action(self, state, possible_actions, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.choice(possible_actions)
        return possible_actions[np.argmax(self.q_table[state][possible_actions])]

    def update(self, state, action, reward, next_state, possible_next_actions):
        best_next_action = np.max(self.q_table[next_state][possible_next_actions])
        self.q_table[state][action] += self.lr * (
            reward + self.gamma * best_next_action - self.q_table[state][action]
        )

# Training
env = GridWorld()
agent = QLearning(16, 4)  # 16 states, 4 actions
episodes = 1000

for episode in range(episodes):
    state = 0  # Start state
    done = False

    while not done:
        possible_actions = env.get_possible_actions(state)
        action = agent.get_action(state, possible_actions)
        next_state, reward, done = env.take_action(state, action)
        possible_next_actions = env.get_possible_actions(next_state)

        agent.update(state, action, reward, next_state, possible_next_actions)
        state = next_state

# Print the learned Q-table
print("\nLearned Q-table:")
print(agent.q_table)

# Find the optimal path
def get_optimal_path():
    state = 0
    path = [state]
    while state != env.goal:
        possible_actions = env.get_possible_actions(state)
        action = possible_actions[np.argmax(agent.q_table[state][possible_actions])]
        state, _, _ = env.take_action(state, action)
        path.append(state)
    return path

optimal_path = get_optimal_path()
print("\nOptimal path:", optimal_path)
```

### 3. Explanation of the Code

1. **GridWorld Class**:

   - Implements a 4x4 grid environment
   - Handles possible actions and state transitions
   - Provides rewards (-1 for each step, 100 for reaching the goal)

2. **QLearning Class**:

   - Maintains the Q-table (state-action values)
   - Implements ε-greedy action selection
   - Updates Q-values using the Q-learning formula:
     Q(s,a) ← Q(s,a) + α[R + γ max Q(s',a') - Q(s,a)]

3. **Training Process**:
   - Runs multiple episodes
   - Agent explores the environment and updates Q-values
   - Gradually learns the optimal policy

### 4. Key Concepts

- **Exploration vs Exploitation**: Balanced using ε-greedy strategy
- **Q-table**: Stores action values for each state
- **Learning Rate (α)**: Controls how much new information overrides old
- **Discount Factor (γ)**: Balances immediate and future rewards

### 5. Example Output

The Q-table will show the learned values for each state-action pair, and the optimal path shows the sequence of states from start to goal.

Try experimenting with different parameters:

- Learning rate
- Discount factor
- Number of episodes
- Epsilon value for exploration

This is a basic implementation, but it demonstrates the core concepts of reinforcement learning and Q-learning specifically.
