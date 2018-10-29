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
PRINT = True # If true, prints more frequent progress updates
RANDOM = False
TRAINING_EPISODES = 5
TRAINING_STEPS = 200000 # Max steps per training episode
TESTING_EPISODES = 100
TESTING_STEPS = 5000 # Max testing steps
# TRAINING_EPISODES = 5
# TRAINING_STEPS = 10 # Max steps per training episode
# TESTING_EPISODES = 2
# TESTING_STEPS = 4 # Max testing steps
RENDER = False # Enable to render last trial
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
    observation = None # Priming
    episodes = range(1, TRAINING_EPISODES + 1) # Number of episodes
    training_terminal_steps = [] # Store amount of steps for each agent to terminate
    tot_rewards = []
    min_step = TRAINING_STEPS
    print("Beginning training...")
    for episode in episodes:
        env.reset() # Reset environment
        table.reset()
        if MODE == 2:
            dtable.reset()
        terminal = 0 # Flag used to end episode after terminal state is reached
        training_terminal_step = 0 # Step at which terminal state is reached
        steps = range(1, TRAINING_STEPS + 1) # Steps per session/episode
        rewards = []
        avg_rewards = []
        if PRINT:
            print("### Training: Episode", episode, "###")
        # Run episode
        for step in steps:
            # Take an action
            if observation is None: # If this is the first action
                previous_state = [0, 0, 0, 0]
                action = env.action_space.sample()
            else: # If this is not the first action
                previous_state = state
                if RANDOM:
                    action = env.action_space.sample()
                else:
                    if MODE == 0 or MODE == 1:
                        action = table.chooseAction(state, EPSILON)
                    elif MODE == 2:
                        action = table.chooseDoubleAction(dtable,state, EPSILON)
            observation = env.step(action)
            # Record new state
            state = process_state(observation[0])
            reward = observation[1]
            rewards.append(reward)
            avg_rewards.append(sum(rewards) / step)
            # Update
            if MODE == 0 or MODE == 1:
                table.update(previous_state, action, state, reward, ALPHA, GAMMA)
            elif MODE == 2:
                flip = np.random.randint(2)
                if flip == 0:
                    table.double_update(dtable, previous_state, action, state, reward, ALPHA, GAMMA)
                elif flip == 2:
                    dtable.double_update(table, previous_state, action, state, reward, ALPHA, GAMMA)
            # Print steps until terminal state just for curiosity's sake
            if reward == 0.0 and terminal == 0:
                if PRINT:
                    print("Training agent reached terminal state at step", step)
                training_terminal_step = step
                terminal = 1
        tot_rewards.append(avg_rewards)
        # Log terminal step
        if training_terminal_step == 0:
            training_terminal_steps.append(TRAINING_STEPS)
        else:
            training_terminal_steps.append(training_terminal_step)
        if training_terminal_step < min_step: # DELETE
            best = table
            if MODE == 2:
                bestd = dtable
    # Store trained Q table
    trained_table = table
    if MODE == 2:
        trained_dtable = dtable
    # Get average and standard deviation of running average rewards
    avg_tot_rewards = np.array(tot_rewards[0])
    for i in range(1, len(tot_rewards)):
        avg_tot_rewards += tot_rewards[i]
    avg_tot_rewards = avg_tot_rewards / len(tot_rewards)
    standard_deviation = np.std(avg_tot_rewards)
    print("Training performance standard deviation:", standard_deviation)

    ###########
    # Testing #
    ###########
    data = [] # Data collection
    testing_terminal_steps = [] # Total steps for all episodes
    observation = None # Priming
    episodes = range(1, TESTING_EPISODES + 1) # Number of episodes
    RRENDER = False
    print("Beginning testing...")
    for episode in episodes:
        # Reset each agent to trained table
        # table = trained_table # ENABLE
        table = best # DELETE
        if MODE == 2:
            # dtable = trained_dtable # ENABLE
            dtable = bestd # DELETE
        env.reset() # Reset environment
        experiment = [[], [], []] # Data
        rewards = [] # Running reward
        testing_terminal_step = 0
        terminal = 0 # Flag used to end episodes at terminal state
        steps = range(1, TESTING_STEPS + 1) # Steps per episode
        # Render last full episode
        if RENDER and episode == EPISODES:
            input("Press enter to begin rendering for testing episode.")
            RRENDER = True
        if PRINT:
            print("### Testing: Episode", episode,"###")
        # Run episode
        for step in steps:
            # Render last episoden for testing
            if RRENDER:
                env.render()
            # Take an action
            if observation is None: # If this is the first action
                previous_state = [0, 0, 0, 0]
                action = env.action_space.sample()
            else: # If this is not the first action
                previous_state = state
                if RANDOM:
                    action = env.action_space.sample()
                else:
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
            # End terminal episodes at terminal state
            if reward == 0.0 and terminal == 0:
                if PRINT:
                    print("Testing agent reached terminal state at step", step)
                testing_terminal_step = step
                terminal = 1
                break
        # Log terminal step
        if testing_terminal_step == 0:
            testing_terminal_steps.append(TESTING_STEPS)
        else:
            testing_terminal_steps.append(testing_terminal_step)
    env.close()

    ###################
    # Data Collection #
    ###################
    # Save data
    if RANDOM:
        out = "R"
    elif MODE == 0:
        out = "Q"
    elif MODE == 1:
        out = "S"
    elif MODE == 2:
        out = "D"
    # Save data
    # Training Running average reward
    file = open("../Data/" + out + "_training_rewards_a" + str(ALPHA) + "_e" + str(EPSILON) + ".txt", "w")
    for r in avg_tot_rewards:
        file.write(str(r) + "\n")
    file.close()
    # Testing terminal steps
    file = open("../Data/" + out + "_testing_steps_a" + str(ALPHA) + "_e" + str(EPSILON) + ".txt", "w")
    file.write("average: " + str(sum(testing_terminal_steps) / len(testing_terminal_steps)) + "\n")
    file.write("minimum: " + str(min(testing_terminal_steps)) + "\n")
    file.write("maximum: " + str(max(testing_terminal_steps)) + "\n")
    file.write("-------------\n")
    count = 1
    for t in testing_terminal_steps:
        file.write("episode " + str(count) + ": " + str(t) + "\n")
        count += 1
    file.close()

    #################
    # Visualization #
    #################
    # Set algorithm label
    if MODE == 0:
        algo = "Q-Learning"
    elif MODE == 1:
        algo = "SARSA"
    elif MODE == 2:
        algo = "Double SARSA"

    # Training
    plt.axes([.1,.1,.8,.7])
    plt.figtext(.5,.9,"Average Training Performance for " + algo + " Across " + str(TRAINING_EPISODES) + " Agents", fontsize=20, ha="center")
    plt.figtext(.5,.85,"Using Alpha=" + str(ALPHA) + " and Epsilon=" + str(EPSILON),fontsize=18,ha="center")
    plt.xlabel("Steps", fontsize=18)
    plt.ylabel("Reward", fontsize=18)
    plt.plot(range(TRAINING_STEPS), avg_tot_rewards, "blue", label="Running Average", linewidth=2)
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    main()
