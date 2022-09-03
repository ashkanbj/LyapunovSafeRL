# LyapunovSafeRL

## Installation
- Pytorch 1.12.0
- OpenAI Gym 0.15.7
- DMLab
- MuJoCo
- OpenAI Safety Gym
## Instructions
- Install the dependencies by running `python setup.py install`
## Hyperparameters
| Hyperparameter (Experiments) | Value | Hyperparameter (Transformers) | Value |
| ----------- | ----------- | ----------- | ----------- |
| Total Steps |  6E6 | Embedding dimension | 256 |
| Batch Size | 16  | Attention heads | 4 |
| Action Repeats | 1  | Dimension of attention heads | 64 |
| Entropy Cost | 0.01  | Feed-forward size | 1024| 
| Baseline Cost | 0.5  | Memory length | 100 |
| Discount Factor | 0.99  | Dropout | 0.1 |
| Reward Clipping | [-1,1]  | Attention span | 400 |
| Optimizer | RMSProp  | Ramp length | 32 |
| Weight Decay | 0  |  |  |
| Smoothing Constant | 0.99  |  |  |
| Momentum | 0  |  |  |
| epsilon | 0.01  |  |  |
| Warmup Steps | 0  |  |  |
| Gradient Norm Clipping | 40  |  |  |
| Scheduler | cosine  |  |  |
| Steps in scheduler | 1,000  |  |  |
| Constraint probability | 1E-4 |  |  |

## Running the code
`python lyapunovrl/algos/pytorch/{algorithm}.py --env <env_name> --exp_name <experiment name>`
## Environments
OpenAI Safety Gym: Safexp-{robot}{task}{difficulty}-v0     
Choose robot from {Point, Car, Doggo}, task from {Goal, Push} and difficulty from {1,2}.     
DMLab-30: dmlab-{level_name}     
Choose level from {rooms_select_nonmatching_object, rooms_watermaze, explore_obstructed_goals_small, explore_goal_locations_small, explore_object_rewards_few, explore_obstructed_goals_large, explore_goal_locations_large, and explore_object_rewards_many}
## Results
- To test the learned policy, run `python scripts/test_policy.py <path to saved model>`
- To plot the results, run `python scripts/plot.py <path to saved model>`