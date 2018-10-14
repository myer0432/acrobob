import sys
import gym
import numpy as np
import matplotlib.pyplot as plt
import bisect
import math

# Configure plotting
plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)

# Static variables
LOWER_BOUNDS = [-1 * math.pi, -1 * math.pi, -12.566, -28.274]
UPPER_BOUNDS = [math.pi, math.pi, 12.566, 28.274]
BINS = [[20, 20, 20, 20], [20, 20, 20, 20], [20, 20, 20, 20]]
OFFSETS = [[0.1, 0.1, 2, 4], [-0.1, -0.1, -2, -4], [0.05, -0.05, -1, 2]] # Offsets should be < bin size which is currently 0.2
ACTIONS = [0, 1, 2] #0 = torque -1 (counter-clockwise), 1 = torque 0, 2 = torque +1 (clockwise)

# Process states function
#
# Description: Reduces the state space by getting theta
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
#
# Description: Creates a grid for tile coding
# @param lower_bounds: Lower bounds of state space
# @param upper_bounds:  Upper bounds of state action_space
# @param bins: Dimension specifying the number bins
# @param offsets: Value by which grid layers should be offset from each other
# @return: The tiles
def make_grid(lower_bounds = [-1.0, -5.0], upper_bounds = [1.0, 5.0],
bins = [10, 10], offsets = [-0.1, 0.5]):
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
#
# Description: Creates a grid for tile coding
# @param lower_bounds: Lower bounds of state space
# @param upper_bounds:  Upper bounds of state action_space
# @param bin_specs: Dimension specifying the number bins
# @param offsets_specs: Value by which grid layers should be offset from each other
# @return: The tiles
def tile(lower_bounds = [-1.0, -5.0], upper_bounds = [1.0, 5.0],
bin_specs = [[10, 10], [10,10]], offsets_specs = [[-0.1, 0.5], [0, 0.4]]):
    # First grid
    grids = [make_grid(lower_bounds, upper_bounds, bin_specs[0], offsets_specs[0])]
    # Iterate through all subsequent bins
    for i in np.arange(1, len(bin_specs)):
        grids = np.append(grids, [make_grid(lower_bounds, upper_bounds, bin_specs[i], offsets_specs[i])], axis=0)
    return grids

# Grid state function
#
# Description: Maps the state to a grid and returns the coordinates
# @param state: The state to map
# @param grid: The grid to map it to
# @return: The coordinates of the state in the grid
def grid_state(state, grid):
    # Grid and state must be of compatible dimesions
    if (len(state) != (grid.shape)[0]):
        return -1
    # To hold rows of pairs each representing coordinates in a row of the grid
    coordinate = np.array([], dtype = "int64")
    # Iterate through each row of the grid
    for i in range(len(state)):
        coordinate = np.append(coordinate, bisect.bisect(grid[i], state[i]))
    return coordinate

# Map state function
#
# Description: Maps the state to a single spot in the multi-layered grid
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

##########################
# Visualization - Delete #
##########################
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
# Visualize tiles
def visualize_tilings(tilings):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    linestyles = ['-', '--', ':']
    legend_lines = []
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, grid in enumerate(tilings):
        for x in grid[0]:
            l = ax.axvline(x=x, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], label=i)
        for y in grid[1]:
            l = ax.axhline(y=y, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)])
        legend_lines.append(l)
    ax.grid(False)
    ax.legend(legend_lines, ["Tiling #{}".format(t) for t in range(len(legend_lines))], facecolor='white', framealpha=0.9)
    ax.set_title("Tilings")
    return ax  # return Axis object to draw on later, if needed
# Visualize with samples
def visualize_encoded_samples(samples, encoded_samples, tilings, low=None, high=None):
    """Visualize samples by activating the respective tiles."""
    samples = np.array(samples)  # for ease of indexing
    # Show tiling grids
    ax = visualize_tilings(tilings)
    # If bounds (low, high) are specified, use them to set axis limits
    if low is not None and high is not None:
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
    else:
        # Pre-render (invisible) samples to automatically set reasonable axis limits, and use them as (low, high)
        ax.plot(samples[:, 0], samples[:, 1], 'o', alpha=0.0)
        low = [ax.get_xlim()[0], ax.get_ylim()[0]]
        high = [ax.get_xlim()[1], ax.get_ylim()[1]]
    # Map each encoded sample (which is really a list of indices) to the corresponding tiles it belongs to
    tilings_extended = [np.hstack((np.array([low]).T, grid, np.array([high]).T)) for grid in tilings]  # add low and high ends
    tile_centers = [(grid_extended[:, 1:] + grid_extended[:, :-1]) / 2 for grid_extended in tilings_extended]  # compute center of each tile
    tile_toplefts = [grid_extended[:, :-1] for grid_extended in tilings_extended]  # compute topleft of each tile
    tile_bottomrights = [grid_extended[:, 1:] for grid_extended in tilings_extended]  # compute bottomright of each tile
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for sample, encoded_sample in zip(samples, encoded_samples):
        for i, tile in enumerate(encoded_sample):
            # Shade the entire tile with a rectangle
            topleft = tile_toplefts[i][0][tile[0]], tile_toplefts[i][1][tile[1]]
            bottomright = tile_bottomrights[i][0][tile[0]], tile_bottomrights[i][1][tile[1]]
            ax.add_patch(Rectangle(topleft, bottomright[0] - topleft[0], bottomright[1] - topleft[1],
                                   color=colors[i], alpha=0.33))
            # In case sample is outside tile bounds, it may not have been highlighted properly
            if any(sample < topleft) or any(sample > bottomright):
                # So plot a point in the center of the tile and draw a connecting line
                cx, cy = tile_centers[i][0][tile[0]], tile_centers[i][1][tile[1]]
                ax.add_line(Line2D([sample[0], cx], [sample[1], cy], color=colors[i]))
                ax.plot(cx, cy, 's', color=colors[i])
    # Finally, plot original samples
    ax.plot(samples[:, 0], samples[:, 1], 'o', color='r')
    ax.margins(x=0, y=0)  # remove unnecessary margins
    ax.set_title("Tile-encoded samples")
    return ax

class TiledQTable:

    def __init__(self, lower_bounds, upper_bounds, bin_specs, offsets_specs, actions):

        self.tiles = tile(lower_bounds, upper_bounds, bin_specs, offsets_specs)
        self.bins = bin_specs
        self.actions = actions
        self.qtables = np.array([])

        qtables = []



########
# Main #
########
def main():
    ####################
    # Testing - Delete #
    ####################
    # States to test
    # states = np.array([np.array([-0.85, -2]), np.array([0, 0]), np.array([1, 1])])
    # tiles = tile()
    # Tile states
    # tiled_states = np.array([map_state(states[0], tiles)])
    # for i in np.arange(1, len(states)):
    #     coordinates = map_state(states[i], tiles)
    #     tiled_states = np.append(tiled_states, [coordinates], 0)
    # print("Coors Shape: ", coordinates.shape)
    # print("Coors: ", coordinates)
    #plt.subplot(visualize_tilings(grids))
    #plt.show()
    #plt.subplot(visualize_encoded_samples(states, encoded_states, grids))
    #plt.show()

    #########
    # Begin #
    #########
    # Initialize Acrobot-v1 environment
    env = gym.make('Acrobot-v1')
    observations = []
    observation = None
    # Explore state (observation) space [delete]
    # print("State space: ", env.observation_space)
    # print("State space min (raw):", env.observation_space.low)
    # print("State space max (raw):", env.observation_space.high)
    # Explore action space
    # print("Action space:", env.action_space)
    # Tile states
    tiles = tile(LOWER_BOUNDS, UPPER_BOUNDS, BINS, OFFSETS)



    print("Tiles:") # Delete
    print(tiles) # Delete

    # For data collection
    trial = []
    episode = []
    # Go
    run = True
    if run:
        for e in range (1):
            env.reset()
            for t in range(0, 10):
                # Render
                env.render()
                # Take an action
                if observation is None: # If this is the first action
                    action = env.action_space.sample()
                    observation = env.step(action) # Take a random action
                else: # If this is not the first action
                    # TODO: Take action by policy
                    action = env.action_space.sample()
                    observation = env.step(action)
                # Record new state
                state = process_state(observation[0])
                coordinates = map_state(state, tiles)
                print("Action: ", action)
                print("State: ", state)
                print("State Coor: ", coordinates)
                trial.append(state)
                print("Trial " + str(t) + ":")
                print(state)
            episode.append(trial)
    env.close()

if __name__ == "__main__":
    main()
