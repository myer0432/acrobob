#!/usr/bin/env python3

# TODO:
# •Figure out how to return to previous state in env
# •Complete policy method
# •Complete weight update method
# •Complete train method
# •Compile statistics needed for analysis
# •Redo reward function

# Raw state space: [cos(theta1) sin(theta1) cos(theta2) sin(theta2) dottheta1 dottheta2]
#     •Bounds: -1 <= cos or sin(theta1 or theta2) <= 1; -4pi <= dottheta1 <= 4pi; -9pi <= dottheta2 <= 9pi
# Processed state space: [theta1 theta2 dottheta1 dottheta2]
# Tiled state space: ?

# Raw action space: [-1, 0, +1] for torque of middle link

import gym
import numpy as np
import math
import copy

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

    processed_states = process_states(state_list)

    #print(processed_states)
    counts = np.zeros((4,4), dtype=np.int32)
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

# Feed this function raw state list
def calculate_reward(state_list):

    processed_states = process_states(state_list)

    reward = math.fabs(processed_states[0])

    reward += math.fabs(processed_states[1]-processed_states[0])

    reward -= math.fabs(processed_states[2])
    reward -= math.fabs(processed_states[3])

    return reward

def calculate_state_value(features, weights):
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

    action = 0

    max_value = 0
    max_value_index = 0
    for i in range(action_space.shape[0]):
#        env.close()
        dummy_env = copy.deepcopy(env)
        raw_state, _, _, _ = dummy_env.step(action_space[i])
        features = tiling(raw_state, tiles, tile_offsets, tile_widths)
        value = calculate_state_value(features, weights)
        if i == 0:
            max_value = value
            max_value_index = 0
        elif value > max_value:
            max_value = value
            max_value_index = i
        #print("Value:", value, "Index:", i, "Action:", action_space[i])
        processed_state = process_states(raw_state)
        #print("The state:", processed_state)

    # Do actual policy part here
    flip = np.random.binomial(size=1, n=1, p=epsilon)
    if flip == 1:
        #print("Max chosen")
        return action_space[max_value_index]
    else:
        #print("Explore chosen")
        action_index = np.random.randint(action_space.shape[0])
        while action_index == max_value_index:
            action_index = np.random.randint(action_space.shape[0])
            
        return action_space[action_index]


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

###
# 1) Initialize feature vector and weight vector

# Originally
'''
dim1 = (2.2*math.pi/12)/4
dim2 = dim1
dim3 = (8.8*math.pi/48)/4
dim4 = (19.8*math.pi/104)/4
'''

divisor1 = 24
divisor2 = 24
divisor3 = 48
divisor4 = 104

dim1 = (2.2*math.pi/divisor1)/4
dim2 = dim1
dim3 = (8.8*math.pi/divisor3)/4
dim4 = (19.8*math.pi/divisor4)/4

unit_vector = np.asarray((dim1,dim2,dim3,dim4), dtype=np.float32)

tile_offsets = 4*[0]
tile_offsets[0] = np.asarray((-1*math.pi,-1*math.pi,-4*math.pi,-9*math.pi))

tile_offsets[1] = np.asarray((1,3,5,7))
tile_offsets[1] = tile_offsets[1]*unit_vector*-1+tile_offsets[0]

tile_offsets[2] = np.asarray((2,6,10,14))
tile_offsets[2] = tile_offsets[2]*unit_vector*-1+tile_offsets[0]

tile_offsets[3] = np.asarray((3,9,15,21))
tile_offsets[3] = tile_offsets[3]*unit_vector*-1+tile_offsets[0]

for i in range(4):
    print(tile_offsets[i])

tiles = np.empty((4, divisor1, divisor2, divisor3, divisor4), dtype=np.int32)
tile_widths = [(2.2*math.pi)/divisor1, (2.2*math.pi)/divisor2, (8.8*math.pi)/divisor3, (19.8*math.pi)/divisor4]
weights = np.zeros((4, divisor1, divisor2, divisor3, divisor4), dtype=np.float32)

###
# 2) Train
#     a) Get initial state
#     b) Get feature vector from state
#     c) Get action from policy by feeding feature vector, return s' and r
#     d) Update weights with s' and s
#     e) Return to b


env = gym.make("Acrobot-v1")                                                                                                                                                      
env.reset()     

post_weights, post_rewards = train(tiles=tiles, tile_offsets=tile_offsets, tile_widths=tile_widths,
      alpha=1/4, lamb=.8, gamma=.9, weights=weights, epsilon=.9, n_episodes=10, n_timesteps=500)

print(post_weights.shape)
print(post_rewards.shape)
