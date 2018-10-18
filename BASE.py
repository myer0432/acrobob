import sys
import gym
import numpy as np
import matplotlib.pyplot as plt
import bisect
import math
import random
from cycler import cycler

# Configure plotting
plt.style.use('classic')
#np.set_printoptions(precision=3, linewidth=120)

# Static variables
ALPHA = 0.2
GAMMA = 0.9
EPSILON = 0.9
LOWER_BOUNDS = [-1 * math.pi, -1 * math.pi, -12.566, -28.274]
UPPER_BOUNDS = [math.pi, math.pi, 12.566, 28.274]
BINS = [[20, 20, 20, 20], [20, 20, 20, 20], [20, 20, 20, 20]]
# The first grid is not offset
# The second grid is offset to the right by ~ 1/3 * the width of one bin
# The third grid is offset to the right by ~ 2/3 * the width of one bin
OFFSETS = [[0, 0, 0, 0], [0.1, 0.1, 0.418867, 0.942467], [0.2, 0.2, 0.83773, 1.88493]]
ACTIONS = [0, 1, 2] #0 = torque -1 (counter-clockwise), 1 = torque 0, 2 = torque +1 (clockwise)

# Process states function
# Description: Reduces the state space by getting theta
#
# from costheta and sintheta
# @param state: the agent's raw state
# @return processed_state: the agent's processed state
def process_state(state):
    processed_state = 4*[0]
    processed_state[0] = np.arctan2(state[1], state[0])
    processed_state[1] = np.arctan2(state[3], state[2])
    processed_state[2] = state[4]
    processed_state[3] = state[5]
    return processed_state

##########
# Tiling #
##########

# Create one grid
# Description: Creates a grid for tile coding
#
# @param lower_bounds: Lower bounds of state space
# @param upper_bounds:  Upper bounds of state action_space
# @param bins: Dimension specifying the number bins
# @param offsets: Value by which grid layers should be offset from each other
# @return: The tiles
def make_grid(lower_bounds, upper_bounds, bins, offsets):
    # Bounds must be of same dimension
    if (len(lower_bounds) != len(bins) or len(upper_bounds) != len(bins) or len(offsets) != len(bins)):
        return -1;
    # First row - We did this to make grid the correct dimensions to
    # allow us to later use append and there's probably a better way
    grid = [np.linspace(lower_bounds[0], upper_bounds[0], bins[0] + 1)[1:-1]]
    grid[0] = grid[0] + offsets[0]
    # Set rest of rows
    for i in np.arange(1, len(lower_bounds)):
        grid = np.append(grid, [np.linspace(lower_bounds[i], upper_bounds[i], bins[i] + 1)[1:-1]], axis=0)
        grid[i] = grid[i] + offsets[i]
    return grid

# Create the grids
# Description: Creates a grid for tile coding
#
# @param lower_bounds: Lower bounds of state space
# @param upper_bounds:  Upper bounds of state action_space
# @param bin_specs: Dimension specifying the number bins
# @param offsets_specs: Value by which grid layers should be offset from each other
# @return: The tiles
def tile(lower_bounds, upper_bounds, bin_specs, offsets_specs):
    # First grid
    grids = [make_grid(lower_bounds, upper_bounds, bin_specs[0], offsets_specs[0])]
    # Iterate through all subsequent bins
    for i in np.arange(1, len(bin_specs)):
        grids = np.append(grids, [make_grid(lower_bounds, upper_bounds, bin_specs[i], offsets_specs[i])], axis=0)
    return grids

# Grid state function
# Description: Maps the state to a grid and returns the coordinates
#
# @param state: The state to map
# @param grid: The grid to map it to
# @return: The coordinates of the state in the grid
def grid_state(state, grid):
    # Grid and state must be of compatible dimesions
    if (len(state) != (grid.shape)[0]):
        return -1
    # To hold rows of pairs each representing coordinates in a row of the grid
    coordinate = np.array([], dtype="int64")
    # Iterate through each row of the grid
    for i in range(len(state)):
        coordinate = np.append(coordinate, bisect.bisect(grid[i], state[i]))
    return coordinate

# Map state function
# Description: Maps the state to a single spot in the multi-layered grid
#
# @param state: The state to map
# @param grid: The multi-layered grid to map it to
# @return: The coordinates of the state in the grid
def map_state(state, grids):
    # Get coordinates for first grid
    coordinates = [grid_state(state, grids[0])]
    # Get coordinates for the rest of the grids
    for i in np.arange(1, grids.shape[0]):
        coordinates = np.append(coordinates, [grid_state(state, grids[i])], axis=0)
    return coordinates

###########
# Q Table #
###########

# QTable class
#
# Description: Used to hold Q values for the agent's policy.
class QTable:
    # Constructor for a TiledQTable
    #
    # @param lower_bounds: Lower bounds of state space
    # @param upper_bounds:  Upper bounds of state action_space
    # @param bin_specs: Dimension specifying the number bins
    # @param offsets_specs: Value by which grid layers should be offset from each other
    # @param actions: The agent's action action_space
    def __init__(self, lower_bounds, upper_bounds, bin_specs, offsets_specs, actions):
        self.tiles = tile(lower_bounds, upper_bounds, bin_specs, offsets_specs)
        temp = list(self.tiles.shape)
        temp[2] += 1
        tup = tuple(temp)
        self.shape = (len(actions),) + tup
        self.table = np.zeros(self.shape, dtype="float")

    # Get Q value
    #
    # @param state: The state
    # @param action: The action
    # @return: The Q value for that state-action pair
    def getQ(self, state, action):
        # Get coordinates
        coordinates = map_state(state, self.tiles)
        # Find the Q value
        sum = 0
        count = 0
        # Each row of coordinates
        for i in range(len(coordinates)):
            row = coordinates[i]
            # Each index in the row
            for j in range(len(row)):
                index = row[j]
                sum += self.table[action][i][j][index]
                count += 1
        return sum / count

    # Set Q value
    #
    # @param state: The state
    # @param action: The action
    # @param value: The Q value to set for that state-action pair
    def setQ(self, state, action, value):
        # Get coordinates
        coordinates = map_state(state, self.tiles)
        # Each row of coordinates
        for i in range(len(coordinates)):
            row = coordinates[i]
            # Each index in the row
            for j in range(len(row)):
                index = row[j]
                self.table[action][i][j][index] = value

    # Reset the Q table
    def reset(self):
        self.table = np.zeros(self.shape, dtype="float")

    # Set Q value
    #
    # @param state: The state
    # @param action: The action
    # @param epsilon: for epsilon greedy action choice, 0 <= epsilon <= 1
    # @param value: The Q value to set for that state-action pair
    def chooseAction(self, state, epsilon):
        # If random float is greater than epsilon, choose random action
        if (random.random() > epsilon):
            return (random.choice(ACTIONS))
        else:
            max_action = 0
            max_q = self.getQ(state, 0)
            for action in np.arange(1, len(self.table)):
                if self.getQ(state, action) > max_q:
                    max_action = action
            return max_action
        return -1

    # Update policy
    #
    # Description: This function updates the agent's
    # policy given the reward from the last state-action pair.
    #
    # @param previous_state: The previous state
    # @param previous_action: The action taken from the previous state
    # @param state: The state the agent is now in
    # @param reward: The reward given for the last action
    # @param alpha: The learning rate alpha
    # @param gamma: The dicount factor gamma
    def update(self, previous_state, previous_action, state, reward, alpha, gamma):
        previous_Qval = self.getQ(previous_state, previous_action) # Q(s_t, a_t)

        ################## Bellman equation for SARSA ###################
        # action = self.chooseAction(state, EPSILON)
        # current_Qval = self.getQ(state, action) # Q(s_t+1, a_t+1)
        # new_Qval = previous_Qval + alpha * (reward + gamma * current_Qval - previous_Qval)
        #################################################################

        ################## Equation for Q-Learning ######################
        max_Qval = np.argmax(self.getQ(state, a) for a in ACTIONS) # Q(s_t+1, a_t+1)
        new_Qval = previous_Qval + alpha * (reward + gamma * max_Qval - previous_Qval)
        #################################################################
        self.setQ(previous_state, previous_action, new_Qval)

        return new_Qval # delete

########
# Main #
########
def main():
    #########
    # Begin #
    #########
    # Initialize Acrobot-v1 environment
    env = gym.make('Acrobot-v1')
    table = QTable(LOWER_BOUNDS, UPPER_BOUNDS, BINS, OFFSETS, ACTIONS)

    print("State space:")
    print(table.table.shape)

    ############
    # Training #
    ############
    #RENDER = False # Enable to render last trial of last experiment
    data = [] # Data collection
    observation = None # Priming
    steps = range(1, 50001) # Steps per session/trial
    trials = range(1, 11) # Number of trials
    best_trial = -1
    best_table = table
    for trial in trials:
        table.reset() # Reset learning for next agent
        env.reset() # Reset environment
        experiment = [[], [], []] # Data
        rewards = [] # Running reward
        #if RENDER and trial == len(trials):
            #input("Begin rendering?")
        print("### Training trial", trial, "###")
        for step in steps:
            # Render last trial
            #if RENDER and trial == len(trials):
                #env.render()
            # Take an action
            if observation is None: # If this is the first action
                previous_state = [0, 0, 0, 0]
                action = env.action_space.sample()
                observation = env.step(action) # Take a random action
            else: # If this is not the first action
                previous_state = state
                action = table.chooseAction(state, EPSILON)
                observation = env.step(action)
            # Record new state
            state = process_state(observation[0])
            reward = observation[1]
            table.update(previous_state, action, state, reward, ALPHA, GAMMA)
            # Data collection
            rewards.append(reward)
            experiment[0].append(state)
            experiment[1].append(reward)
            experiment[2].append(sum(rewards) / float(step))
        data.append(experiment)
        if (sum(rewards) / float(step)) > best_trial:
            best_trial = sum(rewards) / float(step)
            best_table = table

    print("best table before testing:")
    print(best_table.table)

    ###########
    # Testing #
    ###########
    RENDER = False # Enable to render last trial of last experiment
    tdata = [] # Data collection
    observation = None # Priming
    tsteps = range(1, 20001) # Steps per session/trial
    ttrials = range(1, 11) # Number of trials
    for ttrial in ttrials:
        env.reset() # Reset environment
        test = [[], [], []] # Data
        rewards = [] # Running reward
        if RENDER and ttrial == len(ttrials):
            input("Begin rendering?")
        print("### Trial", ttrial, "###")
        for tstep in tsteps:
            # Render last trial
            if RENDER and ttrial == len(ttrials):
                env.render()
            # Take an action
            if observation is None: # If this is the first action
                previous_state = [0, 0, 0, 0]
                action = env.action_space.sample()
                observation = env.step(action) # Take a random action
            else: # If this is not the first action
                previous_state = state
                action = best_table.chooseAction(state, EPSILON)
                observation = env.step(action)
            # Record new state
            state = process_state(observation[0])
            reward = observation[1]
            # Data collection
            rewards.append(reward)
            test[0].append(state)
            test[1].append(reward)
            test[2].append(sum(rewards) / float(tstep))
        tdata.append(test)
    env.close()

    #################
    # Visualization #
    #################
    # Average of training trials
    averaged_rewards = np.array(data[0][2])
    for i in range(1, len(data)):
        averaged_rewards += data[i][2]
    averaged_rewards /= len(data)
    # Save data
    file = open("Q_training.txt", "w")
    for value in averaged_rewards:
        file.write(str(value) + "\n")
    file.close()

    # Average of testing trials
    taveraged_rewards = np.array(tdata[0][2])
    for i in range(1, len(tdata)):
        taveraged_rewards += tdata[i][2]
    taveraged_rewards /= len(tdata)
    # Save data
    file = open("Q_testing.txt", "w")
    for value in taveraged_rewards:
        file.write(str(value) + "\n")
    file.close()

    # Average of all training trials
    plt.axes([.1,.1,.8,.7])
    plt.figtext(.5,.9,"Q-Learning Training Performance", fontsize=20, ha="center")
    plt.figtext(.5,.85,"with Alpha=" + str(ALPHA) + " and Epsilon=" + str(EPSILON) + " across " + str(len(trials)) + " agents",fontsize=18,ha="center")
    #plt.figtext(.5,.85,"with Alpha=" + str(ALPHA) + " and Epsilon=" + str(EPSILON),fontsize=15,ha="center")
    plt.xlabel("Steps", fontsize=18)
    plt.ylabel("Reward", fontsize=18)
    plt.plot(steps, averaged_rewards, "blue", label="Running Average", linewidth=3)
    plt.legend()
    plt.show()

    # Average of all testing trials
    plt.axes([.1,.1,.8,.7])
    plt.figtext(.5,.9,"Q-Learning Testing Performance", fontsize=20, ha="center")
    plt.figtext(.5,.85,"with Alpha=" + str(ALPHA) + " and Epsilon=" + str(EPSILON) + " across " + str(len(trials)) + " agents",fontsize=18,ha="center")
    #plt.figtext(.5,.85,"with Alpha=" + str(ALPHA) + " and Epsilon=" + str(EPSILON),fontsize=15,ha="center")
    plt.xlabel("Steps", fontsize=18)
    plt.ylabel("Reward", fontsize=18)
    plt.plot(tsteps, taveraged_rewards, "blue", label="Running Average", linewidth=3)
    plt.legend()
    plt.show()

    # # All trials with average
    # plt.axes([.1,.1,.8,.7])
    # plt.figtext(.5,.9,"Q-Learning Performance", fontsize=20, ha="center")
    # plt.figtext(.5,.85,"with Alpha=" + str(ALPHA) + " and Epsilon=" + str(EPSILON) + " across " + str(len(trials)) + " agents",fontsize=18,ha="center")
    # # plt.figtext(.5,.85,"with Alpha=" + str(ALPHA) + " and Epsilon=" + str(EPSILON),fontsize=15,ha="center")
    # plt.xlabel("Steps", fontsize=18)
    # plt.ylabel("Reward", fontsize=18)
    # # plt.rc('axes', prop_cycle=(cycler('color', ['r', 'y', 'g', 'c', 'b', 'm']) +
    # #                            cycler('linestyle', ['-', '-', '-', '-', '-', '-'])))
    # plt.plot(steps, data[0][2], color="blue", label="Agent Trials")
    # for i in range(1, len(data)):
    #     plt.plot(steps, data[i][2], color="blue")
    # plt.plot(steps, averaged_rewards, color="black", label="Running Average of all Agents", linewidth=3)
    # plt.legend()
    # plt.show()

if __name__ == "__main__":
    main()
