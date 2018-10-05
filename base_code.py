#!/usr/bin/env python3

# Raw state space: [cos(theta1) sin(theta1) cos(theta2) sin(theta2) dottheta1 dottheta2]
#     •Bounds: -1 <= cos or sin(theta1 or theta2) <= 1; -4pi <= dottheta1 <= 4pi; -9pi <= dottheta2 <= 9pi
# Processed state space: [theta1 theta2 dottheta1 dottheta2]
# Tiled state space: ?

# Raw action space: [-1, 0, +1] for torque of middle link

import gym
import numpy as np
import math

# Tiles state space into distinct partitions. The intersections between those partitions are unique, and each is one element.
# 4 tilings/partitions, calculated by pg 180
# Unit distance is (tile width / # of tiles).
#     # of tiles per dimension: (12,12,48,108)
#     Range of each dimension: (-pi->pi, -pi->pi, -4pi->+4pi, -9pi->+9pi)
#     Length of each dimension: (2pi,2pi,8pi,18pi)
#     Tile width per dimension: (2.4pi/12, 2.4pi/12, 9.6pi/48, 21.6pi/108)
#     Unit distance per dimension: ( (2.4pi/12)/4, (2.4pi/12)/4, (9.6pi/48)/4, (21.6pi/108)/4) )
# Offset vector is (1,3,5,7), adding from an origin tile of (0,0,0,0)
#     Tile 1: (0, 0, 0, 0)
#     Tile 2: (1, 3, 5, 7)
#     Tile 3: (2, 6,10,14)
#     Tile 4: (3, 9,15,21)
#     Tile 5: (4,12,20,28)
# Discretized state space: (48, 48, 192, 432)
# Total number of states: ~191 million

def tiling(state_list, tiles, tile_offsets, tile_widths):

    processed_states = 4*[0]
    processed_states[0] = np.arctan2(state_list[1], state_list[0])
    processed_states[1] = np.arctan2(state_list[3], state_list[2])
    processed_states[2] = state_list[4]
    processed_states[3] = state_list[5]

    print(processed_states)

    for i in range(1):
        counts = np.zeros((4), dtype=np.int32)
        benchmark = np.zeros((4), dtype=np.float32)
        
        benchmark[0] = tile_offsets[i][0]
        while processed_states[0] > benchmark[0] and counts[0] < 11:
            benchmark[0] += tile_widths[0]
            counts[0] += 1   

        benchmark[1] = tile_offsets[i][1]
        while processed_states[1] > benchmark[1] and counts[1] < 11:
            benchmark[1] += tile_widths[1]
            counts[1] += 1
                
        benchmark[2] = tile_offsets[i][2]
        while processed_states[2] > benchmark[2] and counts[2] < 47:
            benchmark[2] += tile_widths[2]
            counts[2] += 1
                    
        benchmark[3] = tile_offsets[i][3]
        while processed_states[3] > benchmark[3] and counts[3] < 107:
            benchmark[3] += tile_widths[3]
            counts[3] += 1
            
        #tiles[i, counts[0], counts[1], counts[2], counts[3]] = 1
        print()
        print(processed_states)
        print(i, benchmark)
        print(i, counts)

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

'''                                                                                                                                                                                                         
env = gym.make("Acrobot-v1")                                                                                                                                                                                
env.reset()                                                                                                                                                                                                 
                                                                                                                                                                                                            
for _ in range(1):                                                                                                                                                                                          
    obs, reward, done, info = env.step(env.action_space.sample())                                                                                                                                           
    print(obs)                                                                                                                                                                                              
'''

#########
# Setup #
#########

dim1 = (2.4*math.pi/12)/4
dim2 = dim1
dim3 = (9.6*math.pi/48)/4
dim4 = (21.6*math.pi/104)/4

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
    
tiles = np.empty((4, 12, 12, 48, 108), dtype=np.int32)
tile_widths = [(2.4*math.pi)/12, (2.4*math.pi)/12, (9.6*math.pi)/48, (21.6*math.pi)/108]
####################
# Actual algorithm #
####################

print()
raw_state_list = [0,1,0,-1,1,1]
reward = calculate_reward(raw_state_list)
print("Reward:", reward)
tiling(raw_state_list, tiles, tile_offsets, tile_widths)
