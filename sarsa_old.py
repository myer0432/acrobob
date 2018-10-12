# -*- coding: utf-8 -*-
#!/usr/bin/env python3

# Raw state space: [cos(theta1) sin(theta1) cos(theta2) sin(theta2) dottheta1 dottheta2]
#     •Bounds: -1 <= cos or sin(theta1 or theta2) <= 1; -4pi <= dottheta1 <= 4pi; -9pi <= dottheta2 <= 9pi
# Processed state space: [theta1 theta2 dottheta1 dottheta2]
# Tiled state space: ?

# Raw action space: [-1, 0, +1] for torque of middle link

import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt

####################
# State processing #
####################

# Tiles state space into distinct partitions. The intersections between those partitions are unique, and each is one element.
# 4 tilings/partitions, calculated by pg 180
# Unit distance is (tile width / # of tiles).
#     # of tiles per dimension: (12,12,48,108)
#     Range of each dimension: (-pi->pi, -pi->pi, -4pi->+4pi, -9pi->+9pi)
#     Length of each dimension: (2pi,2pi,8pi,18pi)
#     Tile width per dimension: (2.2pi/12, 2.2pi/12, 8.8pi/48, 19.8pi/108)
#     Unit distance per dimension: ( (2.2pi/12)/4, (2.2pi/12)/4, (8.8pi/48)/4, (19.8pi/108)/4) )
# Offset vector is (1,3,5,7), adding from an origin tile of (0,0,0,0)
#     Tile 1: (0, 0, 0, 0)
#     Tile 2: (1, 3, 5, 7)
#     Tile 3: (2, 6,10,14)
#     Tile 4: (3, 9,15,21)
# Discretized state space: (4, 12, 12, 48, 108)
# Total number of states: 2,875,392 million
def tiling(state_list, tiles, tile_offsets, tile_widths):

    # Process states
    processed_states = process_states(state_list)
    print("processed states:", processed_states)
    # Set cutoffs
    counts = np.zeros((4,4), dtype=np.int32)
    for i in range(4):
        benchmark = np.zeros((4), dtype=np.float32)
        benchmark_high = np.zeros((4), dtype=np.float32)
        # First layer
        benchmark[0] = tile_offsets[i][0]+tile_widths[0]
        while processed_states[0] > benchmark[0] and counts[i,0] < 11:
            benchmark[0] += tile_widths[0]
            counts[i,0] += 1   
        # Second layer
        benchmark[1] = tile_offsets[i][1]+tile_widths[1]
        while processed_states[1] > benchmark[1] and counts[i,1] < 11:
            benchmark[1] += tile_widths[1]
            counts[i,1] += 1
        # Third layer
        benchmark[2] = tile_offsets[i][2]+tile_widths[2]
        while processed_states[2] > benchmark[2] and counts[i,2] < 47:
            benchmark[2] += tile_widths[2]
            counts[i,2] += 1
        # Fourth layer
        benchmark[3] = tile_offsets[i][3]+tile_widths[3]
        while processed_states[3] > benchmark[3] and counts[i,3] < 107:
            benchmark[3] += tile_widths[3]
            counts[i,3] += 1
        tiles[i, counts[i,0], counts[i,1], counts[i,2], counts[i,3]] = 1
        return tiles

        '''
        print()
        print("Processed states:", processed_states)
        print("Tile widths:", tile_widths)
        print("Upper bound:", benchmark)
        print("Indices:", counts)

        i = 0
        print()
        for j in range(12):
            print(j, " | ", tile_widths[0]*j+tile_offsets[i][0], " | ", processed_states[0], "|", counts[i,0])
        print()
        for j in range(12):
            print(j, " | ", tile_widths[1]*j+tile_offsets[i][1], " | ", processed_states[1], "|", counts[i,1])
        print()
        for j in range(48):
            print(j, " | ", tile_widths[2]*j+tile_offsets[i][2], " | ", processed_states[2], "|", counts[i,2])
        print()
        for j in range(104):
            print(j, " | ", tile_widths[3]*j+tile_offsets[i][3], " | ", processed_states[3], "|", counts[i,3])
        '''

# Process raw states
def process_states(state_list):
    processed_states = 4*[0]
    processed_states[0] = np.arctan2(state_list[1], state_list[0])
    processed_states[1] = np.arctan2(state_list[3], state_list[2])
    processed_states[2] = state_list[4]
    processed_states[3] = state_list[5]
    return processed_states

######################
# Learning algorithm #
######################

# Choose an action
def choose_action(q, state, actions):
    if random.random() < e:
        action = random.choice(actions)
    else:
        q = [q.get((state, a), 0.0) for a in actions]
        maxq = max(q)
        if q.count(maxq) > 1:
            best = [i for i in range(len(actions)) if q[i] == maxq]
            i = random.choice(best)
        else:
            i = q.index(maxq)
        action = actions[i]
    return action

# Training
def train(s1, a1, r, s2, a2):
    # Get q of next state action pair
    q2 = q.get((s2, a2), 0.0)
    # Get v of current state action pair
    v = q.get((s1, a1), None)
    # Calculate q for the state action pair
    if v is None:
        q[(s1, a1)] = r
    else:
        q[(s1, a1)] = v+a * (r+g*q2-v)

# Feed this function raw state list
def calculate_reward(state_list):

    processed_states = process_states(state_list)
    reward = math.fabs(processed_states[0])
    reward += math.fabs(processed_states[1]-processed_states[0])
    reward -= math.fabs(processed_states[2])
    reward -= math.fabs(processed_states[3])
    return reward

#########
# Setup #
#########

dim1 = (2.2*math.pi/12)/4
dim2 = dim1
dim3 = (8.8*math.pi/48)/4
dim4 = (19.8*math.pi/104)/4
unit_vector = np.asarray((dim1,dim2,dim3,dim4), dtype=np.float32)

tile_offsets = 4*[0]
tile_offsets[0] = np.asarray((-1*math.pi,-1*math.pi,-4*math.pi,-9*math.pi))
tile_offsets[1] = np.asarray((1,3,5,7))
tile_offsets[1] = tile_offsets[1]*unit_vector*-1+tile_offsets[0]
tile_offsets[2] = np.asarray((2,6,10,14))
tile_offsets[2] = tile_offsets[2]*unit_vector*-1+tile_offsets[0]
tile_offsets[3] = np.asarray((3,9,15,21))
tile_offsets[3] = tile_offsets[3]*unit_vector*-1+tile_offsets[0]

print("tile offsets:")
for i in range(4):
    print(tile_offsets[i])
    
tiles = np.empty((4, 12, 12, 48, 104), dtype=np.int32)
tile_widths = [(2.2*math.pi)/12, (2.2*math.pi)/12, (8.8*math.pi)/48, (19.8*math.pi)/104]

###############
# Actual Main #
###############

print("----- State information -----");
raw_state_list = [-1,0,1,0,0,0]
print("raw_state_list:", raw_state_list)
reward = calculate_reward(raw_state_list)
print("Reward:", reward)
tiled_states = tiling(raw_state_list, tiles, tile_offsets, tile_widths)
print("tiled_states.shape:", tiled_states.shape)
print("4*12*12*48*104:",4*12*12*48*104)

# For data collection
episode = []
trial = []
# Set up environment
env = gym.make('Acrobot-v1')
env.reset()
env.render()
# Learning algorithm
q = {}
e = 0.1
a = 0.2
g = 0.9
actions = [-1, 0, 1]
# Begin experiments
observations = []
observation = None

# Take first random action
#observation = env.step(env.action_space.sample())
#state = observation[0]
#reward = calculate_reward(observation[0])
#tiled_states = tiling(observation[0], tiles, tile_offsets, tile_widths)

for x in range(0, 5):
    print("trial #:", x)
    # Render
    env.render()
    # Take an action
    if observation is None:
        observation = env.step(random.choice(actions))
    else:
        action = choose_action(q, state, actions)
        observation = env.step(action)
    # Record new state 
    state = observation[0]
    print("state[0]:", state[-1]);
    trial.append(state)
    print("state:", state)
    # Calculate reward
    reward = calculate_reward(observation[0])
    trial.append(reward)
    print("reward:", reward)
    episode.append(trial)

print("\n\nepisode:\n")
print(episode)

'''
# Visualization
theta1 = []
theta2 = []
costheta1 = []
sintheta1 = []
costheta2 = []
sintheta2 = []
omega1 = []
omega2 = []
reward = []
done = []
info = []
'''

'''
for x in observations:
    print(x)
    costheta1.append(x[0][0])
    print("costheta1: " + str(x[0][0])) # Delete
    sintheta1.append(x[0][1])
    print("sintheta1: " + str(x[0][1])) # Delete
    costheta2.append(x[0][2])
    sintheta2.append(x[0][3])
    omega1.append(x[0][4])
    print("omega1: " + str(x[0][4])) # Delete
    omega2.append(x[0][5])
    reward.append(x[1])
    done.append(x[2])
    info.append(x[3])
'''

'''
plt.title("Figure 1: Cosine and Sine of Rotational Angles of Pivot Joint")
plt.xlabel("Steps")
plt.plot(X, costheta1, label="Cos", color="blue")
plt.plot(X, sintheta1, label="Sin", color="blue", linestyle="--")
plt.legend()
plt.show()

plt.title("Figure 2: Cosine and Sine of Rotational Angles of Actuated Joint")
plt.xlabel("Steps")
plt.plot(X, costheta2, label="Cos", color="orange")
plt.plot(X, sintheta2, label="Sin", color="orange", linestyle="--")
plt.legend()
plt.show()

plt.title("Figure 3: Angular Velocities of Joints")
plt.xlabel("Steps")
plt.ylabel("Radians Per Second")
plt.plot(X, omega1, label="Pivot Joint", color="blue")
plt.plot(X, omega2, label="Actuated Joint", color="orange")
plt.legend()
plt.show()
'''
