import sys
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import bisect
import math
import random
from cycler import cycler

# Configure plotting
plt.style.use('classic')

####################
# Static variables #
####################
# Main
MODE = 0 # 0 = Q-Learning, 1 = SARSA, 2 = Double SARSA
STEPS = 5000
EPISODES = 100
# Hyperparameters
ALPHA = 0.2
GAMMA = 0.9
EPSILON = 0.9
# Tiling
LOWER_BOUNDS = [-1 * math.pi, -1 * math.pi, -12.566, -28.274]
UPPER_BOUNDS = [math.pi, math.pi, 12.566, 28.274]
########################### 5 TILES ####################################
OFFSETS = ([[0, 0, 0, 0], [0.06283184, 0.06283184, 0.25132, 0.56548], [1.2566368, 1.2566368, 0.50264, 1.13096],
    [0.18849552, 0.18849552, 0.75396, 1.69644], [0.25132736, 0.25132736, 1.00528, 2.26192]])
BINS = [[20, 20, 20, 20], [20, 20, 20, 20], [20, 20, 20, 20], [20, 20, 20, 20], [20, 20, 20, 20]]
########################################################################
########################### 3 TILES ####################################
#OFFSETS = [[0, 0, 0, 0], [1.2566368, 1.2566368, 0.50264, 1.13096], [0.25132736, 0.25132736, 1.00528, 2.26192]]
#BINS = [[20, 20, 20, 20], [20, 20, 20, 20], [20, 20, 20, 20]]
########################################################################
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
        self.bin_specs = bin_specs
        # Make shape of Q table (z, a, b, c) where (a, b, c) is the shape of the tiles
        # and z is the size of the action space
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

    # Choose action by policy for Q and SARSA
    #
    # @param state: The state
    # @param epsilon: for epsilon greedy action choice, 0 <= epsilon <= 1
    # @return: The chosen action
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

    # Choose action by policy for double SARSA
    #
    # @param other_table: The second table for double SARSA
    # @param state: The state
    # @param epsilon: for epsilon greedy action choice, 0 <= epsilon <= 1
    # @return: The chosen action
    def chooseDoubleAction(self, other_table, state, epsilon):
        if (random.random() > epsilon):
            return (random.choice(ACTIONS))
        else:
            max_action = 0
            max_q = (self.getQ(state, 0) + other_table.getQ(state,0))/2
            for action in np.arange(1, len(self.table)):
                if (self.getQ(state, action)+other_table.getQ(state,0))/2 > max_q:
                    max_action = action
            return max_action
        return -1

    # Update policy for Q and SARSA
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
        if MODE == 0: # Q
            max_Qval = np.argmax(self.getQ(state, a) for a in ACTIONS) # Q(s_t+1, a_t+1)
            new_Qval = previous_Qval + alpha * (reward + gamma * max_Qval - previous_Qval)
        elif MODE == 1: # SARSA
            action = self.chooseAction(state, EPSILON)
            current_Qval = self.getQ(state, action) # Q(s_t+1, a_t+1)
            new_Qval = previous_Qval + alpha * (reward + gamma * current_Qval - previous_Qval)
        self.setQ(previous_state, previous_action, new_Qval)
        return new_Qval # delete

    # Update policy for double SARSA
    #
    # Description: This function updates the agent's
    # policy given the reward from the last state-action pair.
    #
    # @param target_table: The table to use for choosing an action
    # @param previous_state: The previous state
    # @param previous_action: The action taken from the previous state
    # @param state: The state the agent is now in
    # @param reward: The reward given for the last action
    # @param alpha: The learning rate alpha
    # @param gamma: The dicount factor gamma
    def double_update(self, target_table, previous_state, previous_action, state, reward, alpha, gamma):
        previous_Qval = self.getQ(previous_state, previous_action) # Qx(s_t, a_t)
        action = target_table.chooseAction(state, EPSILON)
        current_Qval = target_table.getQ(state, action) # Qy(s_t+1, a_t+1)
        new_Qval = previous_Qval + alpha * (reward + gamma * current_Qval - previous_Qval)
        self.setQ(previous_state, previous_action, new_Qval)
        return new_Qval

########
# Main #
########
def main():
    #########
    # Begin #
    #########
    env = gym.make('Acrobot-v1')
    table = QTable(LOWER_BOUNDS, UPPER_BOUNDS, BINS, OFFSETS, ACTIONS)
    if MODE == 2: # If double SARSA
        dtable = QTable(LOWER_BOUNDS, UPPER_BOUNDS, BINS, OFFSETS, ACTIONS)
    TILES = len(table.bin_specs)
    ############
    # Training #
    ############
    data = [] # Data collection
    total_steps = [] # Total steps for all trials
    observation = None # Priming
    steps = range(1, STEPS + 1) # Steps per session/trial
    trials = range(1, EPISODES + 1) # Number of trials
    for trial in trials:
        table.reset() # Reset learning for next agent
        if MODE == 2:
            dtable.reset()
        env.reset() # Reset environment
        experiment = [[], [], []] # Data
        rewards = [] # Running reward
        total_step = 0 # Total reward per trial
        print("### Episode", trial, "###")
        for step in steps:
            # Take an action
            if observation is None: # If this is the first action
                previous_state = [0, 0, 0, 0]
                action = env.action_space.sample()
                observation = env.step(action) # Take a random action
            else: # If this is not the first action
                previous_state = state
                if MODE == 0 or MODE == 1:
                    action = table.chooseAction(state, EPSILON)
                elif MODE == 2:
                    action = table.chooseDoubleAction(dtable,state, EPSILON)
                observation = env.step(action)
            # Record new state
            state = process_state(observation[0])
            reward = observation[1]
            # Update
            if MODE == 0 or MODE == 1:
                table.update(previous_state, action, state, reward, ALPHA, GAMMA)
            elif MODE == 2:
                flip = np.random.randint(2)
                if flip == 0:
                    table.double_update(dtable, previous_state, action, state, reward, ALPHA, GAMMA)
                elif flip == 2:
                    dtable.double_update(table, previous_state, action, state, reward, ALPHA, GAMMA)
            # Data collection
            rewards.append(reward)
            # Increment total reward until terminal state is reached
            experiment[0].append(state)
            experiment[1].append(reward)
            experiment[2].append(sum(rewards) / float(step))
            # End episode at terminal state
            if reward == 0.0:
                print("Agent reached terminal state at step", step)
                total_step = step
                break
        # Log terminal step
        if total_step == 0:
            total_steps.append(STEPS)
        else:
            total_steps.append(total_step)
        data.append(experiment)

    ###################
    # Data Collection #
    ###################
    # Save data
    if MODE == 0:
        out = "Q"
    elif MODE == 1:
        out = "S"
    elif MODE == 2:
        out = "D"
    file = open("Data/" + out + "_terminalsteps_t" + str(TILES) + "_a" + str(ALPHA) + "_e" + str(EPSILON) + ".txt", "w")
    file.write("average: " + str(sum(total_steps) / len(total_steps)) + "\n")
    stats = np.array(total_steps)
    file.write("std dev: " + str(np.std(stats)) + "\n")
    file.write("minimum: " + str(min(total_steps)) + "\n")
    file.write("maximum: " + str(max(total_steps)) + "\n")
    file.write("-------------\n")
    count = 1
    for t in total_steps:
        file.write("trial " + str(count) + ": " + str(t) + "\n")
        count += 1
    file.close()

    # This needs to be rewritten
    # # Average of training trials
    # averaged_rewards = np.array(data[0][2])
    # for i in range(1, len(data)):
    #     averaged_rewards += data[i][2]
    # averaged_rewards /= len(data)
    # # Save data
    # file = open("Data/" + out + "_runningavg_t" + str(TILES) + "_a" + str(ALPHA) + "_e" + str(EPSILON) + ".txt", "w")
    # for value in averaged_rewards:
    #     file.write(str(value) + "\n")
    # file.close()

    #################
    # Visualization #
    #################
    # # Set algorithm label
    # if MODE == 0:
    #     algo = "Q-Learning"
    # elif MODE == 1:
    #     algo = "SARSA"
    # elif MODE == 2:
    #     algo = "Double SARSA"

    # Average of testing trials
    # taveraged_rewards = np.array(tdata[0][2])
    # for i in range(1, len(tdata)):
    #     taveraged_rewards += tdata[i][2]
    # taveraged_rewards /= len(tdata)
    # # Save data
    # file = open("Data/" + out + "_testing_a" + str(ALPHA) + "_e" + str(EPSILON) + ".txt", "w")
    # for value in taveraged_rewards:
    #     file.write(str(value) + "\n")
    # file.close()

    # Training
    # plt.axes([.1,.1,.8,.7])
    # plt.figtext(.5,.9,"Training Performance for " + algo + " Across " + str(EPISODES) + " Trials", fontsize=20, ha="center")
    # plt.figtext(.5,.85,"Using Alpha=" + str(ALPHA) + " and Epsilon=" + str(EPSILON),fontsize=18,ha="center")
    # plt.xlabel("Steps", fontsize=18)
    # plt.ylabel("Reward", fontsize=18)
    # plt.plot(range(STEPS), averaged_rewards, "blue", label="Running Average", linewidth=2)
    # plt.legend(loc="lower right")
    # plt.show()

    # Testing
    # plt.axes([.1,.1,.8,.7])
    # plt.figtext(.5,.9,"Testing Performance for " + algo + " Across " + str(T_TRIALS) + " trials", fontsize=20, ha="center")
    # plt.figtext(.5,.85,"Using Alpha=" + str(ALPHA) + " and Epsilon=" + str(EPSILON),fontsize=18,ha="center")
    # plt.xlabel("Steps", fontsize=18)
    # plt.ylabel("Reward", fontsize=18)
    # plt.plot(range(T_STEPS), taveraged_rewards, "orange", label="Running Average", linewidth=2)
    # plt.legend(loc="lower right")
    # plt.show()

if __name__ == "__main__":
    main()
