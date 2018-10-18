# acrobob

**AcroBob** is a reinforcement learning agent being developed in the Open AI Gym Acrobot-v1 environment using tile coding with Q-learning, SARSA, and expected SARSA.

### Updates
**10.13.18:** Just added the Base.py file that Mel and I have been working on. Contains new tiling implementation. Tiling works but needs to be expanded to more dimensions (currently at 2, we have 4). Mapping of states needs to be more robustly tested and tiling has not been tested with actual agent actions, only simulated samples. After that we have a plan for implemention of learning. We plan on tackling all of these today. -S

**10.18.18:** AcroBob version 1.0 complete and functional. All three learning algorithms, Q, SARSA, and double SARSA, are integrated into the same BASE.py file. One can ctrl+f "Enable for Q and SARSA" to find all lines of code that must be enabled for Q and SARSA and "Enable for double SARSA" to find all lines of code that must be enabled for double SARSA. Code is currently set on Q and SARSA. Hyperparamaters are currently set to a=0.2, e=0.9, g=0.9. Number of tiles=3. This is the current working version as of long checkpoint 2.
