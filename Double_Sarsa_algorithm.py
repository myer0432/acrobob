#!/usr/bin/env python3

# Raw state space: [cos(theta1) sin(theta1) cos(theta2) sin(theta2) dottheta1 dottheta2]
#     •Bounds: -1 <= cos or sin(theta1 or theta2) <= 1; -4pi <= dottheta1 <= 4pi; -9pi <= dottheta2 <= 9pi
# Processed state space: [theta1 theta2 dottheta1 dottheta2]
# Raw action space: [-1, 0, +1] for torque of middle link

import gym
import numpy as np
import math
import copy

def tiling(state_list, tiles, tile_offsets, tile_widths):

    processed_states = process_states(state_list)

    counts = np.zeros( tiles.shape, dtype=np.int32)

    for i in range(4):
        benchmark = np.zeros((4), dtype=np.float32)
        benchmark_high = np.zeros((4), dtype=np.float32)

        benchmark[0] = tile_offsets[i][0]+tile_widths[0]
        while processed_states[0] > benchmark[0] and counts[i,0] < 11:
            benchmark[0] += tile_widths[0]
            counts[i,0] += 1   

        benchmark[1] = tile_offsets[i][1]+tile_widths[1]
        while processed_states[1] > benchmark[1] and counts[i,1] < 11:
            benchmark[1] += tile_widths[1]
            counts[i,1] += 1
                
        benchmark[2] = tile_offsets[i][2]+tile_widths[2]
        while processed_states[2] > benchmark[2] and counts[i,2] < 47:
            benchmark[2] += tile_widths[2]
            counts[i,2] += 1
                    
        benchmark[3] = tile_offsets[i][3]+tile_widths[3]
        while processed_states[3] > benchmark[3] and counts[i,3] < 107:
            benchmark[3] += tile_widths[3]
            counts[i,3] += 1
            
        tiles[i, counts[i,0], counts[i,1], counts[i,2], counts[i,3]] = 1
        features = tiles

        return features

# Feed this function raw state list
def process_states(state_list):

    #print(state_list)

    processed_states = 4*[0]
    processed_states[0] = np.arctan2(state_list[1], state_list[0])
    processed_states[1] = np.arctan2(state_list[3], state_list[2])
    processed_states[2] = state_list[4]
    processed_states[3] = state_list[5]

    return processed_states

def calculate_state_action_value(features, weights):
    weighted_mult = np.multiply(features, weights)
    state_value = np.sum(weighted_mult)
    
    return state_value

# Learning algorithm to implemented: TD, Sarsa, Expected Sarsa
def weight_update(alpha, lamb, gamma, z, reward, 
                  features, weights, state_value, state_value_prime):

    error = reward + gamma*state_value_prime - state_value

    z = gamma*lamb*z+features

    weights = weights + alpha*error*z

    return weights, z
    

# Policy choice
def policy_choice(action_space, env, weights, tiles, tile_offsets, tile_widths, epsilon):
    a = 0

# Training
def train(tiles, tile_offsets, tile_widths,
          alpha, gamma, lamb, weights, epsilon, n_episodes, n_timesteps):
    
    env = gym.make("Acrobot-v1")
    action_space = np.asarray((-1,0,1), dtype=np.int32)
    training_rewards = np.zeros((n_episodes), dtype=np.float32)

    print("HERE WE GOOOOO!!!")
    print(n_episodes)

    for i in range(n_episodes):

        env.reset()
        total_rewards = 0

        z = 0

        timestep_count = 0
        terminal = False
        while terminal is False:

            action = policy_choice(action_space, env, weights, tiles, tile_offsets, tile_widths, epsilon) # Get first action
            #print("First action:", action)

            # TD Prediction: starting in raw state and following vpi thereafter
 #           env.render()
            raw_state, _, _, _ = env.step(action) # Get first state
            features = tiling(raw_state, tiles, tile_offsets, tile_widths) # Get first features
            state_value = calculate_state_value(features, weights) # V(s)
            action = policy_choice(action_space, env, weights, tiles, tile_offsets, tile_widths, epsilon) # Get second action
            #print("Second action:", action)
            raw_state_prime, reward, terminal, _ = env.step(action) # Get second state, Rt+1
            ### reward = calculate_reward(raw_state_prime) # Get Rt+1
            features_prime = tiling(raw_state_prime, tiles, tile_offsets, tile_widths) # Get second features
            state_value_prime = calculate_state_value(features, weights) # V(s')

            weights, z = weight_update(alpha=alpha, lamb=lamb, gamma=gamma, reward=reward, z=z,
                                    features=features, weights=weights, state_value=state_value, state_value_prime=state_value_prime) # Update weights to learn
            total_rewards += reward

            print("\tTimestep:", timestep_count, "Terminal:", terminal)
            timestep_count += 1

        training_rewards[i] = total_rewards
        
        print("Episode:", i)
        print("Reward:", total_rewards)
        print()

    return weights, training_rewards

################
# Program Plan #
################
#
# 1) Initialize feature vector and weight vector
# 2) Train
#     a) Get initial state
#     b) Get feature vector from state
#     c) Get action from policy by feeding feature vector, return s' and r
#     d) Update weights with s' and s
#     e) Return to b
# 3) Get statistics from run

# Dimensions: (theta1, theta2, thetadot1, thetadot2)
# Intervals: (6, 6, 7, 7)
# Range: (-pi to pi, -pi to pi, -4pi to 4pi, -9pi to 9pi)
# Range Length: (2pi, 2pi, 8pi, 18pi)
# Interval Width: (pi/3, pi/3, 8pi/7, 8pi/7)
# Effective Range Length: (2.1pi, 2.1pi, 8.8pi, 19.8pi)
# Effective Interval Width: (2.1pi/6, 2.1pi/6, 8.8pi/7, 19.8pi/7)


### Create offset buckets
# 1 set, 12 tilings, 4 dimensions
offsets_1 = np.zeros((12,4), dtype=np.float32)

# 4 sets, 3 tilings, 3 dimensions
offsets_2 = np.zeros((4,3,3), dtype=np.float32)

# 6 sets, 2 tilings, 2 dimensions
offsets_3 = np.zeros((6,2,2), dtype=np.float32)

# 4 sets, 3 tilings, 1 dimension
offsets_4 = np.zeros((4,3), dtype=np.float32)

offset_list = [offsets_1, offsets_2, offsets_3, offsets_4]

### Create dimension bucket options
offset_options = np.asarray((2.2*math.pi/6, 2.2*math.pi/6, 8.8*math.pi/7, 19.8*math.pi/7), dtype=np.float32)
origin_options = np.asarray((-1.05*math.pi, -1.05*math.pi, -4.4*math.pi, -9.9*math.pi), dtype=np.float32)

# First set of tilings
set_1 = np.asarray((0,1,2,3), dtype=np.float32)

# Second set of tilings
set_2 = np.asarray( ( (0,1,2), (0,1,3), (0,2,3), (1,2,3) ), dtype=np.float32)

# Third set of tilings
set_3 = np.asarray( ( (0,1), (0,2), (0,3), (1,2), (1,3), (2,3) ), dtype=np.float32)

# Fourth set of tilings
set_4 = np.asarray( ( (0), (1), (2), (3) ), dtype=np.float32)

set_list = [set_1, set_2, set_3, set_4]

### Initialize offsets
for i in range(4):
    print(offset_list[i].shape)
    print(set_list[i].shape)
    print()

for i in range(12):
    for j in range(4):

        origin = origin_options[ set_1[j] ]
        base_interval = offset_options[ set_1[j] ]
        actual_offset = -1*np.random.uniform(high=base_interval)

        offsets_1[i,j] = actual_offset

for i in range(4):
    for j in range(3):
        for k in range(3):
            
            origin = origin_options[ set_2[i,k] ]
            base_interval = offset_options[ set_2[i,k ] ]
            actual_offset = -1*np.random.uniform(high=base_interval)

            offsets_2[i,j,k] = actual_offset

for i in range(6):
    for j in range(2):
        for k in range(2):

            origin = origin_options[ set_3[i,k] ]
            base_interval = offset_options[ set_3[i,k] ]
            actual_offset = -1*np.random.uniform(high=base_interval)

            offsets_3[i,j,k] = actual_offset

for i in range(4):
    for j in range(3):

        origin = origin_options[ set_4[i] ]
        base_interval = offset_options[ set_4[i] ]
        actual_offset = -1*np.random.uniform(high=base_interval)

        offsets_4[i,j] = actual_offset

def tiling(state_list, ):
    a = 0
