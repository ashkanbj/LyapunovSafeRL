from math import gamma
import time

import gym
import safety_gym
import numpy as np
import torch
from torch.optim import Adam
from torch.autograd import Variable

import lyapunovrl.algos.pytorch.ppo.core as core
from lyapunovrl.utils.logx import EpochLogger
from lyapunovrl.utils.mpi_pytorch import (
    mpi_avg_grads,
    setup_pytorch_for_mpi,
    sync_params,
)
from lyapunovrl.utils.mpi_tools import (
    mpi_sum,
    mpi_avg,
    mpi_fork,
    mpi_statistics_scalar,
    num_procs,
    proc_id,
)


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        size,
        # pi_info_shapes,
        gamma=0.99,
        lam=0.95,
        cost_gamma=0.99,
        cost_lam=0.95,
    ):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.cadv_buf = np.zeros(size, dtype=np.float32)  # cost advantage
        self.cost_buf = np.zeros(size, dtype=np.float32)  # costs
        self.cret_buf = np.zeros(size, dtype=np.float32)  # cost return
        self.cval_buf = np.zeros(size, dtype=np.float32)  # cost value
        self.logp_buf = np.zeros(size, dtype=np.float32)
        # self.pi_info_bufs = {
        #     k: np.zeros([size] + list(v), dtype=np.float32)
        #     for k, v in pi_info_shapes.items()
        # }
        # self.sorted_pi_info_keys = core.keys_as_sorted_list(self.pi_info_bufs)
        self.gamma, self.lam = gamma, lam
        self.cost_gamma, self.cost_lam = cost_gamma, cost_lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(
        self,
        obs,
        act,
        rew,
        val,
        cost,
        cval,
        logp,
        # pi_info,
    ):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.cost_buf[self.ptr] = cost
        self.cval_buf[self.ptr] = cval
        self.logp_buf[self.ptr] = logp
        # for k in self.sorted_pi_info_keys:
        #     self.pi_info_bufs[k][self.ptr] = pi_info[k]
        self.ptr += 1

    def finish_path(self, last_val=0, last_cval=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        costs = np.append(self.cost_buf[path_slice], last_cval)
        cvals = np.append(self.cval_buf[path_slice], last_cval)
        cdeltas = costs[:-1] + self.gamma * cvals[1:] - cvals[:-1]
        self.cadv_buf[path_slice] = core.discount_cumsum(
            cdeltas, self.cost_gamma * self.cost_lam
        )
        self.cret_buf[path_slice] = core.discount_cumsum(costs, self.cost_gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + core.EPS)

        # Center, but do NOT rescale advantages for cost gradient
        cadv_mean, _ = mpi_statistics_scalar(self.cadv_buf)
        self.cadv_buf -= cadv_mean

        data = dict(
            obs=self.obs_buf,
            act=self.act_buf,
            ret=self.ret_buf,
            adv=self.adv_buf,
            cadv=self.cadv_buf,
            cret=self.cret_buf,
            logp=self.logp_buf,
        )
        return {
            k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()
        }  # + core.values_as_sorted_list(self.pi_info_bufs)


def alsrl(
    env_fn,
    agent=core.PPOAgent(),
    actor_critic=core.MLPActorCritic,
    ac_kwargs=dict(),
    seed=0,
    render=False,
    steps_per_epoch=4000,
    epochs=50,
    # Discount factors:
    gamma=0.99,
    cost_gamma=0.99,
    clip_ratio=0.2,
    pi_lr=3e-4,
    vf_lr=1e-3,
    train_pi_iters=80,
    train_v_iters=80,
    # Discount factors:
    lam=0.97,
    cost_lam=0.97,
    # Policy learning:
    ent_reg=0.0,
    # Cost constraints / penalties:
    cost_lim=25,
    penalty_init=1.0,
    penalty_lr=5e-2,
    max_ep_len=1000,
    # KL divergence:
    target_kl=0.01,
    logger_kwargs=dict(),
    save_freq=10,
):
    """
    Proximal Policy Optimization (by clipping),

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
            module. The ``step`` method should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing
                                           | the log probability, according to
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical:
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while
            still profiting (improving the objective function)? The new policy
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`.

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # agent = core.PPOAgent(**PPO_kwargs)

    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    agent.set_logger(logger)

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Organize placeholders for zipping with data from buffer on updates
    # buf_phs = [
    #     ac.x_ph,
    #     ac.a_ph,
    #     ac.adv_ph,
    #     ac.cadv_ph,
    #     ac.ret_ph,
    #     ac.cret_ph,
    #     ac.logp_old_ph,
    # ]
    # buf_phs += core.values_as_sorted_list(ac.pi_info_phs)

    # Organize symbols we have to compute at each step of acting in env
    # get_action_ops = dict(pi=ac.pi, v=ac.v, logp_pi=ac.logp_pi, pi_info=ac.pi_info)

    # If agent is reward penalized, it doesn't use a separate value function
    # for costs and we don't need to include it in get_action_ops; otherwise we do.
    # if not (agent.reward_penalized):
    #     get_action_ops["vc"] = ac.vc

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v, ac.vc])
    logger.log("\nNumber of parameters: \t pi: %d, \t v: %d, \t vc: %d\n" % var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    # pi_info_shapes = {k: v.shape.as_list()[1:] for k, v in ac.pi_info_phs.items()}
    buf = PPOBuffer(
        obs_dim,
        act_dim,
        local_steps_per_epoch,
        # pi_info_shapes,
        gamma,
        lam,
        cost_gamma,
        cost_lam,
    )

    # Set up penalty params
    def compute_loss_penalty(data):
        """
        Compute the loss penalty for the given data.
        """
        cur_cost = data["cur_cost"]

        penalty_param = Variable(
            torch.exp(torch.Tensor([penalty_init])) - 1, requires_grad=True
        )

        # penalty in tf is calculated as tf.nn.softplus(penalty_param)
        penalty = torch.nn.Softplus()(penalty_param)

        if agent.learn_penalty:
            if agent.penalty_param_loss:
                penalty_loss = -penalty_param * (cur_cost - cost_lim)
            else:
                penalty_loss = -penalty * (cur_cost - cost_lim)

        return penalty, penalty_loss, penalty_param

    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, cadv, logp_old = (
            data["obs"],
            data["act"],
            data["adv"],
            data["cadv"],
            data["logp"],
        )

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
        if agent.clipped_adv:
            surr_adv = (torch.min(ratio * adv, clip_adv)).mean()
        else:
            surr_adv = (ratio * adv).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        # surrogate cost
        surr_cost = (ratio * cadv).mean()

        # Create policy objective function, including entropy regularization
        pi_objective = surr_adv + ent_reg * ent
        if agent.use_penalty:
            penalty, _, _ = compute_loss_penalty(data)
            # Possibly include surr_cost in pi_objective
            if agent.objective_penalized:
                pi_objective -= penalty.item() * surr_cost
                pi_objective /= 1 + penalty.item()

        # Loss function for pi is negative of pi_objective
        loss_pi = -pi_objective

        if agent.trust_region:
            raise NotImplementedError("Trust region not implemented.")

        elif agent.first_order:
            return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data["obs"], data["ret"]
        return ((ac.v(obs) - ret) ** 2).mean()

    # Set up function for computing cost loss
    def compute_loss_vc(data):
        obs, cret = data["obs"], data["cret"]
        return ((ac.vc(obs) - cret) ** 2).mean()

    # If agent uses penalty directly in reward function, don't train a separate
    # value function for predicting cost returns. (Only use one vf for r - p*c.)
    # if agent.reward_penalized:
    #     total_value_loss = compute_loss_v
    # else:
    #     total_value_loss = compute_loss_v + compute_loss_vc

    # Set up optimizers for policy, value function, and penalty
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    vcf_optimizer = Adam(ac.vc.parameters(), lr=vf_lr)
    if agent.use_penalty:
        # pen, _, penalty_param = compute_loss_penalty(data)
        penalty_param = Variable(
            torch.exp(torch.Tensor([penalty_init])) - 1, requires_grad=True
        )
        penalty_optimizer = Adam([penalty_param], lr=penalty_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        cur_cost = logger.get_stats("EpCost")[0]
        data["cur_cost"] = cur_cost
        # data["cur_penalty"] = penalty_param
        c = cur_cost - cost_lim
        if c > 0 and agent.cares_about_cost:
            logger.log("Warning! Safety constraint is already violated.", "red")

        penalty_loss = -penalty_param * c
        # penalty learning
        if agent.use_penalty:
            # compute_loss_penalty(data)
            penalty_optimizer.zero_grad()
            penalty_loss.backward()
            mpi_avg_grads(penalty_param)
            penalty_optimizer.step()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()
        vc_l_old = compute_loss_vc(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info["kl"])
            if kl > 1.5 * target_kl:
                logger.log("Early stopping at step %d due to reaching max kl." % i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)  # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)  # average grads across MPI processes
            vf_optimizer.step()

            # cost value function learning
            vcf_optimizer.zero_grad()
            loss_vc = compute_loss_vc(data)
            loss_vc.backward()
            mpi_avg_grads(ac.vc)  # average grads across MPI processes
            vcf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info["kl"], pi_info_old["ent"], pi_info["cf"]
        logger.store(
            LossPi=pi_l_old,
            LossV=v_l_old,
            LossVC=vc_l_old,
            KL=kl,
            Entropy=ent,
            ClipFrac=cf,
            DeltaLossPi=(loss_pi.item() - pi_l_old),
            DeltaLossV=(loss_v.item() - v_l_old),
            DeltaLossVC=(loss_vc.item() - vc_l_old),
        )

    # Prepare for interaction with environment
    start_time = time.time()
    o, r, d, c, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0, 0
    cur_penalty = 0
    cum_cost = 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):

        for t in range(local_steps_per_epoch):

            # Possibly render
            if render and proc_id() == 0 and t < 1000:
                env.render()

            a, v_t, vc_t, logp_t = ac.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, d, info = env.step(a)

            # Include penalty on cost
            c = info.get("cost", 0)

            # Track cumulative cost over training
            cum_cost += c

            # save and log
            if agent.reward_penalized:
                r_total = r - cur_penalty * c
                r_total = r_total / (1 + cur_penalty)
                buf.store(o, a, r_total, v_t, 0, 0, logp_t)
            else:
                buf.store(o, a, r, v_t, c, vc_t, logp_t)
            logger.store(VVals=v_t, CostVVals=vc_t)

            # Update obs (critical!)
            o = next_o
            ep_ret += r
            ep_len += 1
            ep_cost += c

            # save and log
            # buf.store(o, a, r, v, logp)
            # logger.store(VVals=v)

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not (terminal):
                    print(
                        "Warning: trajectory cut off by epoch at %d steps." % ep_len,
                        flush=True,
                    )
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, vc, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                    vc = 0
                buf.finish_path(last_val=v, last_cval=vc)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len, EpCost=ep_cost)
                o, ep_ret, ep_cost, ep_len = env.reset(), 0, 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({"env": env}, None)

        # Perform PPO update!
        update()

        cumulative_cost = mpi_sum(cum_cost)
        cost_rate = cumulative_cost / ((epoch + 1) * steps_per_epoch)

        # Log info about epoch
        logger.log_tabular("Epoch", epoch)

        # Performance stats
        logger.log_tabular("EpRet", with_min_and_max=True)
        logger.log_tabular("EpCost", with_min_and_max=True)
        logger.log_tabular("EpLen", average_only=True)
        logger.log_tabular("CumulativeCost", cumulative_cost)
        logger.log_tabular("CostRate", cost_rate)

        # value function values
        logger.log_tabular("VVals", with_min_and_max=True)
        logger.log_tabular("CostVVals", with_min_and_max=True)

        # Vc loss and change, if applicable (reward_penalized agents don't use vc)
        if not (agent.reward_penalized):
            logger.log_tabular("LossVC", average_only=True)
            logger.log_tabular("DeltaLossVC", average_only=True)

        if agent.use_penalty or agent.save_penalty:
            # logger.log_tabular("Penalty", average_only=True)
            # logger.log_tabular("DeltaPenalty", average_only=True)
            pass
        else:
            logger.log_tabular("Penalty", 0)
            logger.log_tabular("DeltaPenalty", 0)

        # Anything from the agent?
        agent.log()

        logger.log_tabular("LossPi", average_only=True)
        logger.log_tabular("DeltaLossPi", average_only=True)

        logger.log_tabular("LossV", average_only=True)
        logger.log_tabular("DeltaLossV", average_only=True)

        logger.log_tabular("Entropy", average_only=True)
        logger.log_tabular("KL", average_only=True)
        logger.log_tabular("ClipFrac", average_only=True)

        # Time and steps elapsed
        logger.log_tabular("TotalEnvInteracts", (epoch + 1) * steps_per_epoch)
        # logger.log_tabular("StopIter", average_only=True)
        logger.log_tabular("Time", time.time() - start_time)
        logger.dump_tabular()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="alsrl")
    parser.add_argument("--env", type=str, default="Safexp-PointGoal1-v0")
    parser.add_argument("--hid", type=int, default=64)
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--cost_gamma", type=float, default=0.99)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--len", type=int, default=1000)
    parser.add_argument("--cost_lim", type=int, default=10)
    parser.add_argument("--exp_name", type=str, default="runagent")
    parser.add_argument("--kl", type=float, default=0.01)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--reward_penalized", action="store_true")
    parser.add_argument("--objective_penalized", action="store_true")
    parser.add_argument("--learn_penalty", action="store_true")
    parser.add_argument("--penalty_param_loss", action="store_true")
    parser.add_argument("--entreg", type=float, default=0.0)
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from lyapunovrl.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # PPO_kwargs for considering safety
    ppo_kwargs = dict(
        reward_penalized=False,
        objective_penalized=True,
        learn_penalty=True,
        penalty_param_loss=True,
    )

    # Prepare agent
    # agent_kwargs = dict(
    #     reward_penalized=args.reward_penalized,
    #     objective_penalized=args.objective_penalized,
    #     learn_penalty=args.learn_penalty,
    #     penalty_param_loss=args.penalty_param_loss,
    # )

    if args.agent == "alsrl":
        agent = core.PPOAgent(**ppo_kwargs)
    else:
        raise NotImplementedError

    alsrl(
        lambda: gym.make(args.env),
        agent=agent,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        seed=args.seed,
        render=args.render,
        # experience collection
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        max_ep_len=args.len,
        # discount factors
        gamma=args.gamma,
        cost_gamma=args.cost_gamma,
        # policy learning
        ent_reg=args.entreg,
        # KL divergence
        target_kl=args.kl,
        # cost function
        cost_lim=args.cost_lim,
        # Logging
        logger_kwargs=logger_kwargs,
        save_freq=1,
    )
