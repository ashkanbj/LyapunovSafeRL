from copy import deepcopy

import gym
import safety_gym

from lyapunovrl.utils.gridnavigation import GridNavigationEnv
from lyapunovrl.utils.mpi_tools import mpi_fork
from lyapunovrl.utils.run_utils import setup_logger_kwargs

# defining the algorithm, task, and robot
algo = "ppo"
task = "goal1"
robot = "point"
# set seed for reproducibility
seed = 1234

cpu = 1

algo = algo.lower()
task = task.capitalize()
robot = robot.capitalize()

# Hyperparameters
exp_name = algo + "_" + robot + task
if robot == "Doggo":
    num_steps = 1e8
    steps_per_epoch = 60000
else:
    num_steps = 1e7
    steps_per_epoch = 30000
epochs = int(num_steps / steps_per_epoch)
save_freq = 50
target_kl = 0.01
cost_lim = 25

# Prepare Logger
exp_name = algo + "_" + robot.lower() + task.lower()
logger_kwargs = setup_logger_kwargs(exp_name, seed)

# Env name
env_name = "Safexp-" + robot + task + "-v0"

# Fork for parallelizing
mpi_fork(cpu)

from lyapunovrl.algos import ppo

# run the algorithm
ppo(
    env_fn=lambda: gym.make(env_name),
    ac_kwargs=dict(
        hidden_sizes=(256, 256),
    ),  # hidden sizes for the policy and value function: This is different from the spinup implementation
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    save_freq=save_freq,  # save frequency: This is not in the spinup implementation
    target_kl=target_kl,  # target KL divergence: This is not in the spinup implementation
    cost_lim=cost_lim,  # cost limit: This is not in the spinup implementation
    seed=seed,
    logger_kwargs=logger_kwargs,
)
