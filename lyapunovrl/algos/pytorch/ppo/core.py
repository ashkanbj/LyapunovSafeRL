from copy import deepcopy

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from gym.spaces import Box, Discrete
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from lyapunovrl.utils.mpi_tools import mpi_avg

EPS = 1e-8


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def keys_as_sorted_list(dict):
    return sorted(list(dict.keys()))


def values_as_sorted_list(dict):
    return [dict[k] for k in keys_as_sorted_list(dict)]


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(
            axis=-1
        )  # Last axis sum needed for Torch Normal distribution


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(
            self.v_net(obs), -1
        )  # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):
    def __init__(
        self, observation_space, action_space, hidden_sizes=(64, 64), activation=nn.Tanh
    ):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(
                obs_dim, action_space.shape[0], hidden_sizes, activation
            )
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(
                obs_dim, action_space.n, hidden_sizes, activation
            )

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

        # build cost value function
        self.vc = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
            vc = self.vc(obs)
        return a.numpy(), v.numpy(), vc.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]


class Agent:
    def __init__(self, **kwargs):
        self.params = deepcopy(kwargs)

    def set_logger(self, logger):
        self.logger = logger

    def prepare_update(self, training_package):
        # training_package is a dict with everything we need (and more)
        # to train.
        self.training_package = training_package

    def prepare_session(self, sess):
        self.sess = sess

    def update_pi(self, inputs):
        raise NotImplementedError

    def log(self):
        pass

    def ensure_satisfiable_penalty_use(self):
        reward_penalized = self.params.get("reward_penalized", False)
        objective_penalized = self.params.get("objective_penalized", False)
        assert not (reward_penalized and objective_penalized), (
            "Can only use either reward_penalized OR objective_penalized, "
            + "not both."
        )

        if not (reward_penalized or objective_penalized):
            learn_penalty = self.params.get("learn_penalty", False)
            assert not (learn_penalty), (
                "If you are not using a penalty coefficient, you should "
                + "not try to learn one."
            )

    def ensure_satisfiable_optimization(self):
        first_order = self.params.get("first_order", False)
        trust_region = self.params.get("trust_region", False)
        assert not (first_order and trust_region), (
            "Can only use either first_order OR trust_region, " + "not both."
        )

    @property
    def cares_about_cost(self):
        return self.use_penalty or self.constrained

    @property
    def clipped_adv(self):
        return self.params.get("clipped_adv", False)

    @property
    def constrained(self):
        return self.params.get("constrained", False)

    @property
    def first_order(self):
        self.ensure_satisfiable_optimization()
        return self.params.get("first_order", False)

    @property
    def learn_penalty(self):
        # Note: can only be true if "use_penalty" is also true.
        self.ensure_satisfiable_penalty_use()
        return self.params.get("learn_penalty", False)

    @property
    def penalty_param_loss(self):
        return self.params.get("penalty_param_loss", False)

    @property
    def objective_penalized(self):
        self.ensure_satisfiable_penalty_use()
        return self.params.get("objective_penalized", False)

    @property
    def reward_penalized(self):
        self.ensure_satisfiable_penalty_use()
        return self.params.get("reward_penalized", False)

    @property
    def save_penalty(self):
        # Essentially an override for CPO so it can save a penalty coefficient
        # derived in its inner-loop optimization process.
        return self.params.get("save_penalty", False)

    @property
    def trust_region(self):
        self.ensure_satisfiable_optimization()
        return self.params.get("trust_region", False)

    @property
    def use_penalty(self):
        return self.reward_penalized or self.objective_penalized


class PPOAgent(Agent):
    def __init__(
        self, clip_ratio=0.2, pi_lr=3e-4, pi_iters=80, kl_margin=1.2, **kwargs
    ):
        super().__init__(**kwargs)
        self.clip_ratio = clip_ratio
        self.pi_lr = pi_lr
        self.pi_iters = pi_iters
        self.kl_margin = kl_margin
        self.params.update(dict(clipped_adv=True, first_order=True, constrained=False))

    def update_pi(self, inputs):

        # Things we need from training package
        train_pi = self.training_package["train_pi"]
        d_kl = self.training_package["d_kl"]
        target_kl = self.training_package["target_kl"]

        # Run the update
        for i in range(self.pi_iters):
            _, kl = self.sess.run([train_pi, d_kl], feed_dict=inputs)
            kl = mpi_avg(kl)
            if kl > self.kl_margin * target_kl:
                self.logger.log("Early stopping at step %d due to reaching max kl." % i)
                break
        self.logger.store(StopIter=i)

    def log(self):
        self.logger.log_tabular("StopIter", average_only=True)
