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


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, dim):
        super(PositionalEmbedding, self).__init__()

        self.dim = dim
        inv_freq = 1 / (10000 ** (torch.arange(0.0, dim, 2.0) / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, positions):
        sinusoid_inp = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb[:, None, :]


class PositionwiseFF(torch.nn.Module):
    def __init__(self, d_input, d_inner, dropout):
        super(PositionwiseFF, self).__init__()

        self.d_input = d_input
        self.d_inner = d_inner
        self.dropout = dropout
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_input, d_inner),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_inner, d_input),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input_):
        ff_out = self.ff(input_)
        return ff_out


class GatingMechanism(torch.nn.Module):
    def __init__(self, d_input, bg=0.1):
        super(GatingMechanism, self).__init__()
        self.Wr = torch.nn.Linear(d_input, d_input)
        self.Ur = torch.nn.Linear(d_input, d_input)
        self.Wz = torch.nn.Linear(d_input, d_input)
        self.Uz = torch.nn.Linear(d_input, d_input)
        self.Wg = torch.nn.Linear(d_input, d_input)
        self.Ug = torch.nn.Linear(d_input, d_input)
        self.bg = bg

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x, y):
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))
        g = torch.mul(1 - z, x) + torch.mul(z, h)
        return g


class MultiHeadAttentionXL(torch.nn.Module):
    def __init__(self, d_input, d_inner, n_heads=4, dropout=0.1, dropouta=0.0):
        super(MultiHeadAttentionXL, self).__init__()

        self.d_input = d_input
        self.d_inner = d_inner
        self.n_heads = n_heads

        # Linear transformation for keys & values for all heads at once for efficiency.
        # 2 for keys & values.
        self.linear_kv = torch.nn.Linear(d_input, (d_inner * n_heads * 2), bias=False)
        # for queries (will not be concatenated with memorized states so separate).
        self.linear_q = torch.nn.Linear(d_input, d_inner * n_heads, bias=False)

        # for positional embeddings.
        self.linear_p = torch.nn.Linear(d_input, d_inner * n_heads, bias=False)
        self.scale = 1 / (d_inner**0.5)  # for scaled dot product attention
        self.dropa = torch.nn.Dropout(dropouta)

        self.lout = torch.nn.Linear(d_inner * n_heads, d_input, bias=False)
        self.dropo = torch.nn.Dropout(dropout)

    def _rel_shift(self, x):
        # x shape: [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        zero_pad = torch.zeros(
            (x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype
        )
        return (
            torch.cat([zero_pad, x], dim=1)
            .view(x.size(1) + 1, x.size(0), *x.size()[2:])[1:]
            .view_as(x)
        )

    def forward(self, input_, pos_embs, memory, u, v, mask=None):
        """
        + pos_embs: positional embeddings passed separately to handle relative positions.
        + Arguments
            - input: torch.FloatTensor, shape - (seq, bs, self.d_input) = (20, 5, 8)
            - pos_embs: torch.FloatTensor, shape - (seq + prev_seq, bs, self.d_input) = (40, 1, 8)
            - memory: torch.FloatTensor, shape - (prev_seq, b, d_in) = (20, 5, 8)
            - u: torch.FloatTensor, shape - (num_heads, inner_dim) = (3 x )
            - v: torch.FloatTensor, shape - (num_heads, inner_dim)
            - mask: torch.FloatTensor, Optional = (20, 40, 1)

        + Returns
            - output: torch.FloatTensor, shape - (seq, bs, self.d_input)

        + symbols representing shape of the tensors
            - cs: current sequence length, b: batch, H: no. of heads
            - d: inner dimension, ps: previous sequence length
        """
        cur_seq = input_.shape[0]
        prev_seq = memory.shape[0]
        H, d = self.n_heads, self.d_inner
        # concat memory across sequence dimension
        # input_with_memory = [seq + prev_seq x B x d_input] = [40 x 5 x 8]
        input_with_memory = torch.cat([memory, input_], dim=0)

        # k_tfmd, v_tfmd = [seq + prev_seq x B x n_heads.d_head_inner], [seq + prev_seq x B x n_heads.d_head_inner]
        k_tfmd, v_tfmd = torch.chunk(
            self.linear_kv(input_with_memory),
            2,
            dim=-1,
        )
        # q_tfmd = [seq x B x n_heads.d_head_inner] = [20 x 5 x 96]
        q_tfmd = self.linear_q(input_)

        _, bs, _ = q_tfmd.shape
        assert bs == k_tfmd.shape[1]

        # content_attn = [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        content_attn = torch.einsum(
            "ibhd,jbhd->ijbh",
            (
                (q_tfmd.view(cur_seq, bs, H, d) + u),
                k_tfmd.view(cur_seq + prev_seq, bs, H, d),
            ),
        )

        # p_tfmd: [seq + prev_seq x 1 x n_heads.d_head_inner] = [40 x 1 x 96]
        p_tfmd = self.linear_p(pos_embs)
        # position_attn = [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        position_attn = torch.einsum(
            "ibhd,jhd->ijbh",
            (
                (q_tfmd.view(cur_seq, bs, H, d) + v),
                p_tfmd.view(cur_seq + prev_seq, H, d),
            ),
        )

        position_attn = self._rel_shift(position_attn)
        # attn = [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        attn = content_attn + position_attn

        if mask is not None and mask.any().item():
            # fills float('-inf') where mask is True.
            attn = attn.masked_fill(mask[..., None], -float("inf"))
        # rescale to prevent values from exploding.
        # normalize across the value sequence dimension.
        attn = torch.softmax(attn * self.scale, dim=1)
        # attn = [curr x curr+prev x B x n_heads] = [20 x 40 x 5 x 3]
        attn = self.dropa(attn)

        # attn_weighted_values = [curr x B x n_heads.d_inner] = [20 x 5 x 96]
        attn_weighted_values = (
            torch.einsum(
                "ijbh,jbhd->ibhd",
                (
                    attn,  # (cs, cs + ps, b, H)
                    v_tfmd.view(cur_seq + prev_seq, bs, H, d),  # (cs + ps, b, H, d)
                ),
            )  # (cs, b, H, d)
            .contiguous()  # we need to change the memory layout to make `view` work
            .view(cur_seq, bs, H * d)
        )  # (cs, b, H * d)

        # output = [curr x B x d_input] = [20 x 5 x 8]
        output = self.dropo(self.lout(attn_weighted_values))
        return output


class StableTransformerEncoderLayerXL(torch.nn.Module):
    def __init__(
        self,
        n_heads,
        d_input,
        d_head_inner,
        d_ff_inner,
        dropout,
        gating=True,
        dropouta=0.0,
    ):
        super(StableTransformerEncoderLayerXL, self).__init__()

        self.gating = gating
        self.gate1 = GatingMechanism(d_input)
        self.gate2 = GatingMechanism(d_input)
        self.mha = MultiHeadAttentionXL(
            d_input,
            d_head_inner,
            n_heads=n_heads,
            dropout=dropout,
            dropouta=dropouta,
        )
        self.ff = PositionwiseFF(d_input, d_ff_inner, dropout)
        self.norm1 = torch.nn.LayerNorm(d_input)
        self.norm2 = torch.nn.LayerNorm(d_input)

    def forward(self, input_, pos_embs, u, v, mask=None, mems=None):
        src2 = self.norm1(input_)
        src2 = self.mha(src2, pos_embs, mems, u, v, mask=mask)
        src = self.gate1(input_, src2) if self.gating else input_ + src2
        src2 = self.ff(self.norm2(src))
        src = self.gate2(src, src2) if self.gating else src + src2
        return src


class StableTransformerXL(torch.nn.Module):
    def __init__(
        self,
        d_input,
        n_layers,
        n_heads,
        d_head_inner,
        d_ff_inner,
        dropout=0.1,
        dropouta=0.0,
    ):
        super(StableTransformerXL, self).__init__()

        (
            self.n_layers,
            self.n_heads,
            self.d_input,
            self.d_head_inner,
            self.d_ff_inner,
        ) = (n_layers, n_heads, d_input, d_head_inner, d_ff_inner)

        self.pos_embs = PositionalEmbedding(d_input)
        self.drop = torch.nn.Dropout(dropout)
        self.layers = torch.nn.ModuleList(
            [
                StableTransformerEncoderLayerXL(
                    n_heads,
                    d_input,
                    d_head_inner=d_head_inner,
                    d_ff_inner=d_ff_inner,
                    dropout=dropout,
                    dropouta=dropouta,
                )
                for _ in range(n_layers)
            ]
        )

        # u and v are global parameters: maybe changing these to per-head parameters might help performance?
        self.u, self.v = (
            # [n_heads x d_head_inner] = [3 x 32]
            torch.nn.Parameter(torch.Tensor(self.n_heads, self.d_head_inner)),
            torch.nn.Parameter(torch.Tensor(self.n_heads, self.d_head_inner)),
        )

    def init_memory(self, device=torch.device("cpu")):
        return [
            # torch.empty(0, dtype=torch.float).to(device)
            torch.zeros(20, 5, 8, dtype=torch.float).to(device)
            for _ in range(self.n_layers + 1)
        ]

    def update_memory(self, previous_memory, hidden_states):
        """
        + Arguments
            - previous_memory: List[torch.FloatTensor],
            - hidden_states: List[torch.FloatTensor]
        """
        assert len(hidden_states) == len(previous_memory)
        mem_len, seq_len = previous_memory[0].size(0), hidden_states[0].size(0)
        # mem_len, seq_len = 3, hidden_states[0].size(0)
        # print(mem_len, seq_len)

        with torch.no_grad():
            new_memory = []
            end_idx = mem_len + seq_len
            beg_idx = max(0, end_idx - mem_len)
            for m, h in zip(previous_memory, hidden_states):
                cat = torch.cat([m, h], dim=0)
                new_memory.append(cat[beg_idx:end_idx].detach())
        return new_memory

    def forward(self, inputs, memory=None):
        """
        + Arguments
            - inputs - torch.FloatTensor = [T x B x d_inner] = [20 x 5 x 8]
            - memory - Optional, list[torch.FloatTensor] = [[T x B x d_inner] x 5]
        """
        if memory is None:
            memory = self.init_memory(inputs.device)
        assert len(memory) == len(self.layers) + 1

        cur_seq, bs = inputs.shape[:2]
        prev_seq = memory[0].size(0)

        # dec_attn_mask = [curr x curr + prev x 1] = [20 x 40 x 1]
        dec_attn_mask = (
            torch.triu(
                torch.ones((cur_seq, cur_seq + prev_seq)),
                diagonal=1 + prev_seq,
            )
            .bool()[..., None]
            .to(inputs.device)
        )

        pos_ips = torch.arange(cur_seq + prev_seq - 1, -1, -1.0, dtype=torch.float).to(
            inputs.device
        )
        # pos_embs = [curr + prev x 1 x d_input] = [40 x 1 x 8]
        pos_embs = self.drop(self.pos_embs(pos_ips))
        if self.d_input % 2 != 0:
            pos_embs = pos_embs[:, :, :-1]

        hidden_states = [inputs]
        layer_out = inputs
        for mem, layer in zip(memory, self.layers):
            # layer_out = [curr x B x d_inner] = [20 x 5 x 8]
            layer_out = layer(
                layer_out,
                pos_embs,
                self.u,
                self.v,
                mask=dec_attn_mask,
                mems=mem,
            )
            hidden_states.append(layer_out)

        # Memory is treated as a const., don't propagate through it
        # new_memory = [[T x B x d_inner] x 4]
        memory = self.update_memory(memory, hidden_states)
        return {"logits": layer_out, "memory": memory}


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


class TransformerGaussianPolicy(torch.nn.Module):
    def __init__(self, state_dim, act_dim, n_transformer_layers=4, n_attn_heads=3):
        """
        NOTE - I/P Shape : [seq_len, batch_size, state_dim]
        """
        super(TransformerGaussianPolicy, self).__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim

        self.transformer = StableTransformerXL(
            d_input=state_dim,
            n_layers=n_transformer_layers,
            n_heads=n_attn_heads,
            d_head_inner=32,
            d_ff_inner=64,
        )
        self.memory = None

        self.head_sate_value = torch.nn.Linear(state_dim, 1)
        self.head_act_mean = torch.nn.Linear(state_dim, act_dim)
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()

    def _distribution(self, trans_state):
        mean = self.tanh(self.head_act_mean(trans_state))
        std = torch.exp(self.log_std)
        return Normal(mean, std)

    def _log_prob_from_distribution(self, policy, action):
        return policy.log_prob(action).sum(axis=-1)

    def forward(self, state, action=None):
        trans_state = self.transformer(state, self.memory)
        trans_state, self.memory = trans_state["logits"], trans_state["memory"]

        policy = self._distribution(trans_state)
        state_value = self.head_sate_value(trans_state)

        logp_a = None
        if action is not None:
            logp_a = self._log_prob_from_distribution(policy, action)

        return policy, logp_a, state_value

    def step(self, state):
        if state.shape[0] == self.state_dim:
            state = state.reshape(1, 1, -1)
        with torch.no_grad():
            trans_state = self.transformer(state, self.memory)
            trans_state, self.memory = trans_state["logits"], trans_state["memory"]

            policy = self._distribution(trans_state)
            action = policy.sample()
            logp_a = self._log_prob_from_distribution(policy, action)
            state_value = self.head_sate_value(trans_state)

        return action.numpy(), logp_a.numpy(), state_value.numpy()
