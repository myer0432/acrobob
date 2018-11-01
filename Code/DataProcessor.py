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

# Hyperparameters
ALPHA = 0.2
GAMMA = 0.9
EPSILON = 0.9
steps = 200000
path = "../Data/Archive/"

# Read data
print("Reading random...")
with open(path + "R_training_rewards_a0.2_e0.9.txt") as file:
    r_data = file.readlines()
r_data = [x.strip() for x in r_data]
r_data = list(map(float, r_data))

print("Reading Q...")
with open(path + "Q_training_rewards_a0.2_e0.9.txt") as file:
    q_data = file.readlines()
q_data = [x.strip() for x in q_data]
q_data = list(map(float, q_data))

print("Reading SARSA...")
with open(path + "S_training_rewards_a0.2_e0.9.txt") as file:
    s_data = file.readlines()
s_data = [x.strip() for x in s_data]
s_data = list(map(float, s_data))

print("Reading double SARSA...")
with open(path + "D_training_rewards_a0.2_e0.9.txt") as file:
    d_data = file.readlines()
d_data = [x.strip() for x in d_data]
d_data = list(map(float, d_data))

print("Plotting...")

# Training
plt.axes([.1,.1,.8,.7])
plt.figtext(.5,.9,"Average Learning Curve Across 5 Agents", fontsize=20, ha="center")
plt.figtext(.5,.85,"Using Alpha=" + str(ALPHA) + " , Epsilon=" + str(EPSILON) + ", Gamma=" + str(GAMMA),fontsize=18,ha="center")
plt.xlabel("Steps", fontsize=18)
plt.ylabel("Running Average of Reward", fontsize=18)
print("Plotting random...")
plt.plot(range(steps), r_data, "yellow", label="Random", linewidth=2)
print("Plotting Q...")
plt.plot(range(steps), q_data, "blue", label="Q-Learning", linewidth=2)
print("Plotting SARSA...")
plt.plot(range(steps), s_data, "orange", label="SARSA", linewidth=2)
print("Plotting double SARSA...")
plt.plot(range(steps), d_data, "green", label="Double SARSA", linewidth=2)
print("Legend...")
plt.legend(loc="lower right")
print("Finishing...")
plt.show()
print("End.")
