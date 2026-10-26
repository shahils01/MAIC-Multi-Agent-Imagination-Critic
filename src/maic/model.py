"""Attention-based team behavior, physical world, and imagined-return models.

Action draws and generated trajectories are detached in this first version.
Forecast loss trains the context/scene encoders and tail-value head, while
the action/world components have their own real-data supervised losses.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .attention import AttentionBlock, AttentionPool, DecoderBlock
from .batch import HistoryBatch, TaskBatch, validate_actions
from .config import MAICConfig


@dataclass
class SceneEncoding:
    agents: Tensor
    team: Tensor


@dataclass
class WorldPrediction:
    next_states: Tensor
    reward: Tensor
    termination_logit: Tensor


@dataclass
class ImaginationOutput:
    mean: Tensor                        # [B]
    variance: Tensor                    # [B], population rollout variance
    returns: Tensor                     # [B, M]
    states: Tensor                      # [B, M, H+1, N, S]
    actions: Tensor                     # [B, M, H, N, C]
    rewards: Tensor                     # [B, M, H], zero after termination
    terminated: Tensor                  # [B, M, H], sampled terminal events
    policy_context: Tensor              # [B, D], inferred ONCE from real prefix


class ActionEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(k, config.d_model) for k in config.action_bins)

    def forward(self, actions):
        return sum(embedding(actions[..., i]) for i, embedding in enumerate(self.embeddings)) / math.sqrt(len(self.embeddings))


class TaskEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.identity = nn.Embedding(config.num_tasks, config.d_model)
        self.parameters = nn.Linear(config.task_parameter_dim, config.d_model, bias=False) if config.task_parameter_dim else None
        self.agent_parameters = nn.Linear(config.agent_task_dim, config.d_model, bias=False) if config.agent_task_dim else None
        self.roles = nn.Embedding(config.num_roles, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, task, agent_mask):
        global_task = self.identity(task.ids)
        if self.parameters is not None:
            global_task = global_task + self.parameters(task.parameters)
        global_task = self.norm(global_task)
        agent_task = self.roles(task.role_ids(agent_mask))
        if self.agent_parameters is not None:
            clean = task.agent_parameters.masked_fill(~agent_mask[..., None], 0)
            agent_task = agent_task + self.agent_parameters(clean)
        return global_task, agent_task.masked_fill(~agent_mask[..., None], 0)


class HistoryEncoder(nn.Module):
    """Spatial self-attention followed by per-agent causal temporal attention."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.d_model
        self.state_projection = nn.Linear(config.state_dim, d)
        self.action_embedding = ActionEmbedding(config)
        self.time_embedding = nn.Embedding(config.history_length, d)
        self.temporal_start = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.spatial = nn.ModuleList(AttentionBlock(d, config.num_heads, config.dropout) for _ in range(config.spatial_layers))
        self.temporal = nn.ModuleList(AttentionBlock(d, config.num_heads, config.dropout) for _ in range(config.temporal_layers))
        self.pool = AttentionPool(d, config.num_heads, config.dropout)

    def forward(self, history, global_task, agent_task):
        history = history.crop(self.config.history_length)
        b, t, n, _ = history.states.shape
        valid = history.valid_times
        state_valid = valid[:, :, None] & history.agent_mask[:, None]
        states = history.states.masked_fill(~state_valid[..., None], 0)
        previous = states.new_zeros((b, t, n, self.config.d_model))
        if t > 1:
            action_valid = state_valid[:, 1:] & valid[:, :-1, None]
            actions = history.actions.masked_fill(~action_valid[..., None], 0)
            previous[:, 1:] = self.action_embedding(actions).masked_fill(~action_valid[..., None], 0)
        positions = (valid.long().cumsum(dim=1) - 1).clamp_min(0)
        tokens = (self.state_projection(states) + previous + self.time_embedding(positions)[:, :, None]
                  + global_task[:, None, None] + agent_task[:, None])
        tokens = tokens.masked_fill(~state_valid[..., None], 0)
        nodes = tokens.reshape(b*t, n, -1)
        spatial_padding = (~history.agent_mask[:, None].expand(b, t, n)).reshape(b*t, n)
        for layer in self.spatial:
            nodes = layer(nodes, padding_mask=spatial_padding)
        temporal = nodes.reshape(b, t, n, -1).permute(0, 2, 1, 3).reshape(b*n, t, -1)
        # An always-visible start token prevents fully masked causal queries on
        # left-padded histories, including during the backward pass.
        temporal = torch.cat((self.temporal_start.expand(b*n, -1, -1), temporal), dim=1)
        padding = (~valid[:, None].expand(b, n, t)).reshape(b*n, t)
        padding = torch.cat((torch.zeros(b*n, 1, dtype=torch.bool, device=states.device), padding), dim=1)
        for layer in self.temporal:
            temporal = layer(temporal, padding_mask=padding, causal=True)
        last = torch.arange(t, device=states.device).expand(b, t).masked_fill(~valid, -1).amax(dim=1) + 1
        last = last[:, None].expand(b, n).reshape(-1)
        nodes = temporal[torch.arange(b*n, device=states.device), last].reshape(b, n, -1)
        nodes = nodes.masked_fill(~history.agent_mask[..., None], 0)
        return SceneEncoding(nodes, self.pool(nodes, history.agent_mask))


class JointActionDecoder(nn.Module):
    """Causal categorical decoding over agent/component slots.

No current/future logged action reaches its own prediction. BOS stays visible
even for interspersed padded agents. Agent order is the chosen autoregressive
factorization; unlike the scene/world encoders it is not permutation invariant.
"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.d_model
        self.action_embedding = ActionEmbedding(config)
        self.component = nn.Embedding(config.action_components, d)
        self.start = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.query_projection = nn.Linear(3*d, d)
        self.layers = nn.ModuleList(DecoderBlock(d, config.num_heads, config.dropout) for _ in range(config.decoder_layers))
        self.output = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, max(config.action_bins)))
        allowed = torch.arange(max(config.action_bins))[None] < torch.tensor(config.action_bins)[:, None]
        self.register_buffer("allowed", allowed)

    def forward(self, scene, context, global_task, agent_mask, actions, *, limit=None):
        b, n, d = scene.agents.shape
        c = self.config.action_components
        if actions.shape != (b, n, c):
            raise ValueError("decoder actions must be [B, N, C]")
        validate_actions(actions, agent_mask, self.config)
        actions = actions.masked_fill(~agent_mask[..., None], 0)
        length = n*c if limit is None else limit
        components = torch.arange(c, device=actions.device).repeat(n)
        # One embedding per actual action component, then a causal right shift.
        embedded = torch.stack([embedding(actions[..., i]) for i, embedding in enumerate(self.action_embedding.embeddings)], dim=2).reshape(b, n*c, d)
        valid = agent_mask[..., None].expand(b, n, c).reshape(b, n*c)
        indices = torch.arange(n*c, device=actions.device).expand(b, -1)
        previous_valid = indices.masked_fill(~valid, -1).cummax(dim=1).values
        previous_valid = torch.cat((indices.new_full((b, 1), -1), previous_valid[:, :-1]), dim=1)
        previous = embedded.gather(1, previous_valid.clamp_min(0)[..., None].expand(b, n*c, d))
        previous = torch.where((previous_valid < 0)[..., None], self.start.expand(b, n*c, d), previous)
        query = self.query_projection(torch.cat((scene.agents, context[:, None].expand(b, n, d), global_task[:, None].expand(b, n, d)), dim=-1))
        query = query[:, :, None].expand(b, n, c, d).reshape(b, n*c, d)
        tokens = query + self.component(components)[None] + previous
        tokens = torch.cat((self.start.expand(b, -1, -1), tokens[:, :length]), dim=1)
        padding = torch.cat((torch.zeros(b, 1, dtype=torch.bool, device=actions.device), ~valid[:, :length]), dim=1)
        memory = torch.cat((context[:, None], global_task[:, None], scene.agents), dim=1)
        memory_padding = torch.cat((torch.zeros(b, 2, dtype=torch.bool, device=actions.device), ~agent_mask), dim=1)
        for layer in self.layers:
            tokens = layer(tokens, memory, padding, memory_padding)
        logits = self.output(tokens[:, 1:]).masked_fill(~self.allowed[components[:length]][None], -torch.inf)
        return logits.reshape(b, n, c, -1) if limit is None else logits

    @torch.no_grad()
    def sample(self, scene, context, global_task, agent_mask, *, generator=None):
        b, n = agent_mask.shape
        c = self.config.action_components
        actions = torch.zeros(b, n, c, dtype=torch.long, device=scene.agents.device)
        for slot in range(n*c):
            logits = self(scene, context, global_task, agent_mask, actions, limit=slot+1)[:, -1]
            sampled = torch.multinomial(logits.softmax(dim=-1), 1, generator=generator)[:, 0]
            agent, component = divmod(slot, c)
            actions[:, agent, component] = torch.where(agent_mask[:, agent], sampled, 0)
        return actions


class WorldModel(nn.Module):
    """Attention over agents; physical dynamics do not use policy context."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.d_model
        self.actions = ActionEmbedding(config)
        self.roles = nn.Embedding(config.num_roles, d)
        self.state_projection = nn.Linear(config.state_dim, d)
        self.dynamics = nn.ModuleList(AttentionBlock(d, config.num_heads, config.dropout) for _ in range(config.world_layers))
        self.delta = nn.Linear(d, config.state_dim)
        self.transition_projection = nn.Linear(2*config.state_dim+d, d)
        self.outcome = nn.ModuleList(AttentionBlock(d, config.num_heads, config.dropout) for _ in range(config.world_layers))
        self.pool = AttentionPool(d, config.num_heads, config.dropout)
        self.reward = nn.Linear(d, 1)
        self.termination = nn.Linear(d, 1)

    def forward(self, states, actions, agent_mask, roles, global_task, agent_task):
        states = states.masked_fill(~agent_mask[..., None], 0)
        actions = actions.masked_fill(~agent_mask[..., None], 0)
        action = self.actions(actions)
        nodes = self.state_projection(states) + action + self.roles(roles)
        for layer in self.dynamics:
            nodes = layer(nodes, padding_mask=~agent_mask)
        following = (states + self.delta(nodes)).masked_fill(~agent_mask[..., None], 0)
        # Real next states can also supervise outcome heads independently of
        # dynamics error; see outcome_prediction below.
        reward, termination = self.outcome_prediction(states, actions, following, agent_mask, global_task, agent_task)
        return WorldPrediction(following, reward, termination)

    def outcome_prediction(self, states, actions, following, agent_mask, global_task, agent_task):
        states = states.masked_fill(~agent_mask[..., None], 0)
        following = following.masked_fill(~agent_mask[..., None], 0)
        actions = actions.masked_fill(~agent_mask[..., None], 0)
        transition = self.transition_projection(torch.cat((states, following, self.actions(actions)), dim=-1))
        transition = transition + global_task[:, None] + agent_task
        for layer in self.outcome:
            transition = layer(transition, padding_mask=~agent_mask)
        team = self.pool(transition, agent_mask)
        return self.reward(team)[:, 0], self.termination(team)[:, 0]


class MAIC(nn.Module):
    def __init__(self, config: MAICConfig):
        super().__init__()
        self.config = config
        self.task_encoder = TaskEncoder(config)
        self.scene_encoder = HistoryEncoder(config)
        self.policy_encoder = HistoryEncoder(config)
        self.policy_projection = nn.Sequential(nn.Linear(config.d_model, config.d_model), nn.LayerNorm(config.d_model))
        self.action_decoder = JointActionDecoder(config)
        self.world_model = WorldModel(config)
        self.value_head = nn.Sequential(nn.Linear(3*config.d_model, 2*config.d_model), nn.GELU(), nn.Linear(2*config.d_model, 1))

    def encode(self, history: HistoryBatch, task: TaskBatch):
        history.validate(self.config)
        task.validate(self.config, history.agent_mask)
        global_task, agent_task = self.task_encoder(task, history.agent_mask)
        scene = self.scene_encoder(history, global_task, agent_task)
        context = self.policy_projection(self.policy_encoder(history, global_task, agent_task).team)
        return scene, context, global_task, agent_task

    def value(self, scene, context, global_task):
        return self.value_head(torch.cat((scene.team, context, global_task), dim=-1))[:, 0]

    def forward(self, history, task, actions=None):
        scene, context, global_task, _ = self.encode(history, task)
        result = {"value": self.value(scene, context, global_task), "policy_context": context}
        if actions is not None:
            result["action_logits"] = self.action_decoder(scene, context, global_task, history.agent_mask, actions)
        return result

    def imagine(self, history, task, *, horizon=5, num_rollouts=8, generator=None):
        """Sample continuations; only the bootstrap has a gradient path.

        Use model.eval() and torch.no_grad() for deployment. The variance is
        rollout spread, not a calibrated epistemic confidence interval.
        """
        if horizon < 1 or num_rollouts < 1:
            raise ValueError("horizon and num_rollouts must be positive")
        _, context, _, _ = self.encode(history, task)
        b, _, n, s = history.states.shape
        config = self.config
        fixed_context = context.repeat_interleave(num_rollouts, dim=0)
        repeated_task = task.repeat(num_rollouts)
        current = history.crop(config.history_length).repeat(num_rollouts)
        mask = current.agent_mask
        total = history.states.new_zeros(b*num_rollouts)
        survival = torch.ones_like(total, dtype=torch.bool)
        rewards, actions, terminations = [], [], []
        states = [current.last_states()]
        with torch.no_grad():
            global_task, agent_task = self.task_encoder(repeated_task, mask)
            for k in range(horizon):
                scene = self.scene_encoder(current, global_task, agent_task)
                joint = self.action_decoder.sample(scene, fixed_context, global_task, mask, generator=generator)
                joint = joint.masked_fill(~survival[:, None, None], 0)
                physical = current.last_states()
                predicted = self.world_model(physical, joint, mask, repeated_task.role_ids(mask), global_task, agent_task)
                reward = torch.where(survival, predicted.reward, torch.zeros_like(total))
                done = (torch.rand(total.shape, device=total.device, generator=generator) < predicted.termination_logit.sigmoid()) & survival
                total = total + config.gamma**k * reward
                following = torch.where(survival[:, None, None], predicted.next_states, physical)
                # Keep a valid repeated terminal state on stopped branches.
                # Their return/bootstrap remain masked, so no post-terminal
                # prediction contributes to utility.
                current = HistoryBatch(torch.cat((current.states, following[:, None]), dim=1),
                    torch.cat((current.actions, joint[:, None]), dim=1), mask,
                    torch.cat((current.valid_times, torch.ones_like(survival[:, None])), dim=1)).crop(config.history_length)
                states.append(following)
                actions.append(joint)
                rewards.append(reward)
                terminations.append(done)
                survival = survival & ~done
        # Re-encode detached final histories, retaining a supervised gradient to
        # the scene encoder, task encoder, fixed context, and tail-value head.
        global_task, agent_task = self.task_encoder(repeated_task, mask)
        final_scene = self.scene_encoder(current, global_task, agent_task)
        tail = self.value(final_scene, fixed_context, global_task)
        returns = (total + config.gamma**horizon * survival.to(tail.dtype) * tail).reshape(b, num_rollouts)
        return ImaginationOutput(returns.mean(dim=1), returns.var(dim=1, unbiased=False), returns,
            torch.stack(states, dim=1).reshape(b, num_rollouts, horizon+1, n, s),
            torch.stack(actions, dim=1).reshape(b, num_rollouts, horizon, n, config.action_components),
            torch.stack(rewards, dim=1).reshape(b, num_rollouts, horizon),
            torch.stack(terminations, dim=1).reshape(b, num_rollouts, horizon), context)
