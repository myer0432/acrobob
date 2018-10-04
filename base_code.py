#!/usr/bin/env python3

# Raw state space: [cos(theta1) sin(theta1) cos(theta2) sin(theta2) dottheta1 dottheta2]
#     •Bounds: -1 <= cos or sin(theta1 or theta2) <= 1; -4pi <= dottheta1 <= 4pi; -9pi <= dottheta2 <= 9pi
# Tiled state space: ?

# Raw action space: [-1, 0, +1] for torque of middle link

import gym
env = gym.make("Acrobot-v1")
env.reset()

for _ in range(1):
    obs, reward, done, info = env.step(env.action_space.sample())
    print(obs)


# Tiles state space into distinct partitions. The intersections between those partitions are unique, and each is one element.
# Need to decide increments from -1 to +1, -4pi to +4pi, -9pi to +9pi.
# Tiling will be asymmetrical, but still square
# Offsets will be determined by number of partitions desired.
def tiling():
    a = 0

# Feed this function raw state list
def calculate_reward(state_list):
    
    reward = -10000

    if (len(state_list) != 6):
        print("Error: need exactly 6 states")
    else:
        # Initialize reward to cos position of center link
        # This will be less than 1 when below 0 altitude, and greater than 1 when above 0 altitude
        reward = state_list[0]*-1

        # Decide if center link is to the left of center divide or to the right
        # If left, add reward -1*sin(theta2)
        # If right, add reward sin(theta2)
        
        # Left
        if (state_list[1] < 0):
            reward += -1*state_list[3]
        # Right
        else:
            reward += state_list[3]
            
            # Subtract both link angular velocities from reward
            reward -= state_list[4]
            reward -= state_list[5]

    return reward

# Learning algorithm to implemented: Q-Learning, Sarsa, Expected Sarsa
def learning_algo():
    a = 0

# Training
def train():
    a = 0
