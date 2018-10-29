steps = 200000
# Read data
with open("R_training_rewards.txt") as file:
    r_data = file.readlines()
r_data = [x.strip() for x in r_data]

with open("Q_training_rewards_a0.2_e0.9.txt") as file:
    q_data = file.readlines()
q_data = [x.strip() for x in q_data]

with open("S_training_rewards_a0.2_e0.9.txt") as file:
    s_data = file.readlines()
s_data = [x.strip() for x in s_data]

with open("D_training_rewards_a0.2_e0.9.txt") as file:
    d_data = file.readlines()
d_data = [x.strip() for x in d_data]

# Training
plt.axes([.1,.1,.8,.7])
plt.figtext(.5,.9,"Average Learning Curve Across 100 Agents", fontsize=20, ha="center")
plt.figtext(.5,.85,"Using Alpha=" + str(ALPHA) + " and Epsilon=" + str(EPSILON),fontsize=18,ha="center")
plt.xlabel("Steps", fontsize=18)
plt.ylabel("Running Average of Reward", fontsize=18)
plt.plot(range(steps), r_data, "yellow", label="Random", linewidth=2)
plt.plot(range(steps), q_data, "blue", label="Q-Learning", linewidth=2)
plt.plot(range(steps), s_data, "orange", label="SARSA", linewidth=2)
plt.plot(range(steps), q_data, "green", label="Double SARSA", linewidth=2)
plt.legend(loc="lower right")
plt.show()
