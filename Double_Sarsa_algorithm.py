#!/usr/bin/env python3

# Raw state space: [cos(theta1) sin(theta1) cos(theta2) sin(theta2) dottheta1 dottheta2]
#     •Bounds: -1 <= cos or sin(theta1 or theta2) <= 1; -4pi <= dottheta1 <= 4pi; -9pi <= dottheta2 <= 9pi
# Processed state space: [theta1 theta2 dottheta1 dottheta2]
# Raw action space: [-1, 0, +1] for torque of middle link

### TODO:
# 1) Modify tiling to support actions as well as states
# 2) Modify weight calculations to happen for all tilings
# 3) Implement Double Sarsa

import gym
import numpy as np
import math
import copy

def tiling():
    a = 0

# Feed this function raw state list
def process_states(state_list):

    #print(state_list)

    processed_states = 4*[0]
    processed_states[0] = np.arctan2(state_list[1], state_list[0])
    processed_states[1] = np.arctan2(state_list[3], state_list[2])
    processed_states[2] = state_list[4]
    processed_states[3] = state_list[5]

    return processed_states

def calculate_state_action_value():
    a = 0

# Learning algorithm to implemented: TD, Sarsa, Expected Sarsa
def weight_update():
    a = 0
    

# Policy choice
def policy_choice():
    a = 0

# Training
def train():
    a = 0

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
offsets_1 = np.zeros((1,12,4), dtype=np.float32)

# 4 sets, 3 tilings, 3 dimensions
offsets_2 = np.zeros((4,3,3), dtype=np.float32)

# 6 sets, 2 tilings, 2 dimensions
offsets_3 = np.zeros((6,2,2), dtype=np.float32)

# 4 sets, 3 tilings, 1 dimension
offsets_4 = np.zeros((4,3,1), dtype=np.float32)

offset_list = np.asarray((offsets_1, offsets_2, offsets_3, offsets_4), dtype=object)

### Create dimension bucket options
interval_options = np.asarray((2.2*math.pi/6, 2.2*math.pi/6, 8.8*math.pi/7, 19.8*math.pi/7), dtype=np.float32)
origin_options = np.asarray((-1.05*math.pi, -1.05*math.pi, -4.4*math.pi, -9.9*math.pi), dtype=np.float32)

# First set of tilings
set_1 = np.asarray((0,1,2,3), dtype=np.int32)
set_1 = np.expand_dims(set_1, axis=0)

# Second set of tilings
set_2 = np.asarray( ( (0,1,2), (0,1,3), (0,2,3), (1,2,3) ), dtype=np.int32)

# Third set of tilings
set_3 = np.asarray( ( (0,1), (0,2), (0,3), (1,2), (1,3), (2,3) ), dtype=np.int32)

# Fourth set of tilings
set_4 = np.asarray( ( (0), (1), (2), (3) ), dtype=np.int32)
set_4 = np.expand_dims(set_4, axis=1)

set_list = np.asarray((set_1, set_2, set_3, set_4), dtype=object)

### Initialize offsets
for i in range(4):
    print("Offset:", offset_list[i].shape)
    print("Count:", counts[i].shape)
    print("Dimension order:", set_list[i].shape)
    print()

# 4 types of tilings, 12 for each type
for i in range(4):

    # Number of sets
    for j in range( offset_list[i].shape[0] ):
        
        # Number of tilings
        for k in range( offset_list[i].shape[1] ):
            
            # Number of dimensions
            for m in range( offset_list[i].shape[2] ):
                
                origin = origin_options[ set_list[i][j,m] ]
                base_interval = interval_options[ set_list[i][j,m] ]
                actual_offset = -1*np.random.uniform(high=base_interval)

                offset_list[i][j,k,m] = actual_offset + origin

                print(origin, base_interval, actual_offset, offset_list[i][j,k,m])

np.save(


def tiling(state_list, set_list, offset_list, interval_options):

    counts = np.empty(offset_list.shape[0], dtype=object)
    for i in range(counts.shape[0]):
        counts[i] = np.zeros(offset_list[i].shape, dtype=np.int32)

    processed_states = process_states(state_list)

    for i in range( offset_list.shape[0] ):
        
        for j in range( offset_list[i].shape[0] ):
            
            for k in range( offset_list[i].shape[1] ):
                
                for m in range( offset_list[i].shape[2] ):
                    
                    base_interval = interval_options[ set_list[i][j,m] ]
                    cursor = offset_list[i][j,k,m] + base_interval
                    while( (count[i][j,k,m] < (set_list[i][j,m]-1)) && (processed_states > cursor) ):
                        counts[i][j,k,m] += 1
                        cursor += base_interval
                  
    return 0
