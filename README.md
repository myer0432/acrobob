# acrobob

**AcroBob** is a reinforcement learning agent being developed in the Open AI Gym Acrobot-v1 environment using tile coding with Q-learning, SARSA, and expected SARSA.

### Table of Contents

|File|Description|
|-------|----------|
|README.md|Read Me|
|AcroBob.py|Current working version|
|Base.py|Development base|

### Recent Notes
Updated Base.py to AcroBob.py reworked to be directly comparable to the Open AI Gym benchmarks. Executes over 100 episodes with 5,000 steps each as rarely does an agent need more than this many steps. Records amount of steps it takes before each agent reaches a terminal state and outputs a file containing statistics about the experiment.

I expanded to 5 and then 9 tiles and fixed our offsets which were jacked but it seemed to make no difference in performance. This version is set with 5 tiles.

All of my experiments so far have been run on Q. The data output can be found in the Data folder on this repo. We need to run the same experiments for ~~SARSA and~~ Double SARSA. You will not need to edit any code in order to generate output files for your different algorithms as it will generate a file with a name starting with the first letter of your algorithm, but if you run the program twice without moving your old output then it will overwrite it.

**Question**: Do we need to run experiments on varying hyperparameters for our report or do we just need one experiment each on the same hyperparameters and then n works to compare to?

### Current Results

This shows on average how long 100 agents took to reach a terminal state in each algorithm.

|Algo|Steps|
|----|-----|
|Q|1753.6|
|S|1532.24|
|D|?|

### Presentation

[Working version](https://docs.google.com/presentation/d/10INKYFpmIKXP7GfELKWysKBvLX78ijg27G1ittbkTAM/edit#slide=id.p)

[Test](Project Specs.pdf)
