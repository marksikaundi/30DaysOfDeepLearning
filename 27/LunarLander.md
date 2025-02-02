I'll help you implement the LunarLander solution step by step, focusing first on the core structure and then the training loop with the 200+ reward target.

Here's the complete implementation:

```python
import gym
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 1. DQN Model
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.network(x)

# 2. DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(state)
            return q_values.argmax().item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([t[0] for t in minibatch]).to(self.device)
        actions = torch.LongTensor([t[1] for t in minibatch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in minibatch]).to(self.device)
        next_states = torch.FloatTensor([t[3] for t in minibatch]).to(self.device)
        dones = torch.FloatTensor([t[4] for t in minibatch]).to(self.device)

        # Current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

        # Next Q values
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss and update
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 3. Training Function
def train_lunar_lander():
    env = gym.make('LunarLander-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    batch_size = 64
    episodes = 1000
    scores = []
    recent_scores = deque(maxlen=100)

    for episode in range(episodes):
        state = env.reset()
        score = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        scores.append(score)
        recent_scores.append(score)
        avg_score = np.mean(recent_scores)

        print(f'Episode {episode+1}/{episodes}, Score: {score:.2f}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}')

        # Check if environment is solved
        if len(recent_scores) == 100 and avg_score >= 200:
            print(f'\nEnvironment solved in {episode+1} episodes!')
            print(f'Average Score: {avg_score:.2f}')
            break

    return scores

# 4. Plot Results
def plot_scores(scores):
    plt.figure(figsize=(10, 6))
    plt.plot(scores)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.show()

# 5. Run Training
if __name__ == "__main__":
    scores = train_lunar_lander()
    plot_scores(scores)
```

Key features of this implementation:

1. **Success Criterion**: The environment is considered solved when the average reward over 100 consecutive episodes reaches 200 or more.

2. **Monitoring Progress**:

   - Tracks scores for each episode
   - Maintains a rolling average of the last 100 episodes
   - Prints progress regularly

3. **Early Stopping**: Training stops when the success criterion is met

4. **Visualization**: Includes a plotting function to visualize training progress

To run this:

1. Make sure you have all required packages installed:

```bash
pip install gym torch numpy matplotlib
```

2. Run the script:

```python
scores = train_lunar_lander()
plot_scores(scores)
```

You should see output like:

```
Episode 1/1000, Score: -154.23, Average Score: -154.23, Epsilon: 0.995
Episode 2/1000, Score: -128.45, Average Score: -141.34, Epsilon: 0.990
...
```

The training might take a while (potentially several hundred episodes) before solving the environment. The epsilon value will gradually decrease, shifting from exploration to exploitation.
