import numpy as np
import random

# --------------------------
# Environment (simple 1D world)
# States: 0 → 4 (goal at 4)
# --------------------------
states = 5
actions = [0, 1]  # 0 = left, 1 = right

goal = 4

# Q-table
Q = np.zeros((states, len(actions)))

# Hyperparameters
alpha = 0.1   # learning rate
gamma = 0.9   # discount factor
epsilon = 0.2 # exploration

episodes = 500

# --------------------------
# Training
# --------------------------
for episode in range(episodes):
    state = 0  # start position

    while state != goal:

        # exploration vs exploitation
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)
        else:
            action = np.argmax(Q[state])

        # take action
        if action == 1:
            next_state = min(state + 1, goal)
        else:
            next_state = max(state - 1, 0)

        # reward
        reward = 1 if next_state == goal else 0

        # Q update (Bellman equation)
        Q[state, action] = Q[state, action] + alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state

# --------------------------
# Results
# --------------------------
print("Trained Q-table:")
print(Q)

print("\nOptimal Policy:")
for s in range(states):
    print(f"State {s} -> Action {np.argmax(Q[s])}")