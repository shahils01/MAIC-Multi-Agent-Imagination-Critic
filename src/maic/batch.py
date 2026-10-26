"""Tensor contracts only; no dataset storage or loading assumptions."""

from dataclasses import dataclass, fields
import torch
from torch import Tensor

from .config import MAICConfig


class TensorBatch:
    def to(self, device):
        return type(self)(**{
            f.name: getattr(self, f.name).to(device) if getattr(self, f.name) is not None else None
            for f in fields(self)
        })

    def select(self, index):
        return type(self)(**{
            f.name: getattr(self, f.name)[index] if getattr(self, f.name) is not None else None
            for f in fields(self)
        })

    def repeat(self, count: int):
        return type(self)(**{
            f.name: getattr(self, f.name).repeat_interleave(count, dim=0)
            if getattr(self, f.name) is not None else None for f in fields(self)
        })


def _shape(value, shape, name):
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name}: expected {tuple(shape)}, got {tuple(value.shape)}")


def _bool_mask(value, shape, name):
    _shape(value, shape, name)
    if value.dtype != torch.bool:
        raise ValueError(f"{name} must be boolean")


def _integer(value, name):
    if value.dtype != torch.long:
        raise ValueError(f"{name} must use torch.long categorical indices")


def _finite_real(value, valid, name):
    if not value.is_floating_point() or not torch.isfinite(value[valid]).all():
        raise ValueError(f"{name} must contain finite floating-point values at valid entries")


def validate_actions(actions, valid, config):
    _integer(actions, "actions")
    for c, bins in enumerate(config.action_bins):
        selected = actions[..., c][valid]
        if ((selected < 0) | (selected >= bins)).any():
            raise ValueError(f"action component {c} outside [0, {bins})")


@dataclass
class TaskBatch(TensorBatch):
    ids: Tensor                         # [B]
    parameters: Tensor | None = None     # [B, P]
    agent_parameters: Tensor | None = None  # [B, N, Q]
    roles: Tensor | None = None          # [B, N]

    def validate(self, config: MAICConfig, agent_mask: Tensor):
        b, n = agent_mask.shape
        _shape(self.ids, (b,), "task ids")
        _integer(self.ids, "task ids")
        if ((self.ids < 0) | (self.ids >= config.num_tasks)).any():
            raise ValueError("task id outside configured vocabulary")
        for value, dim, shape, valid, name in (
            (self.parameters, config.task_parameter_dim, (b, config.task_parameter_dim),
             torch.ones(b, dtype=torch.bool, device=agent_mask.device), "task parameters"),
            (self.agent_parameters, config.agent_task_dim, (b, n, config.agent_task_dim),
             agent_mask, "agent task parameters"),
        ):
            if dim and value is None:
                raise ValueError(f"{name} are required by the model configuration")
            if value is not None:
                _shape(value, shape, name)
                _finite_real(value, valid, name)
        if self.roles is not None:
            _shape(self.roles, (b, n), "roles")
            _integer(self.roles, "roles")
            real = self.roles[agent_mask]
            if ((real < 0) | (real >= config.num_roles)).any():
                raise ValueError("role id outside configured vocabulary")

    def role_ids(self, agent_mask):
        return torch.zeros_like(agent_mask, dtype=torch.long) if self.roles is None else self.roles.masked_fill(~agent_mask, 0)


@dataclass
class HistoryBatch(TensorBatch):
    states: Tensor                      # [B, T, N, S]
    actions: Tensor                     # [B, T-1, N, C]; never a_t
    agent_mask: Tensor                  # [B, N], true for real agents
    time_mask: Tensor | None = None      # [B, T], contiguous real observations

    @property
    def valid_times(self):
        return self.time_mask if self.time_mask is not None else torch.ones(
            self.states.shape[:2], dtype=torch.bool, device=self.states.device)

    def validate(self, config):
        if self.states.ndim != 4:
            raise ValueError("states must be [B, T, N, S]")
        b, t, n, s = self.states.shape
        if min(b, t, n) < 1 or s != config.state_dim:
            raise ValueError("empty history or incorrect state_dim")
        _bool_mask(self.agent_mask, (b, n), "agent_mask")
        if not self.agent_mask.any(dim=1).all():
            raise ValueError("each sample needs at least one real agent")
        _bool_mask(self.valid_times, (b, t), "time_mask")
        if not self.valid_times.any(dim=1).all():
            raise ValueError("each sample needs an observed state")
        positions = torch.arange(t, device=self.states.device).expand(b, t)
        first = positions.masked_fill(~self.valid_times, t).amin(dim=1)
        last = positions.masked_fill(~self.valid_times, -1).amax(dim=1)
        if not (self.valid_times.sum(dim=1) == last - first + 1).all():
            raise ValueError("time_mask must contain one contiguous observed segment")
        _shape(self.actions, (b, t-1, n, config.action_components), "history actions")
        state_valid = self.valid_times[:, :, None] & self.agent_mask[:, None, :]
        _finite_real(self.states, state_valid, "states")
        action_valid = state_valid[:, 1:] & self.valid_times[:, :-1, None]
        validate_actions(self.actions, action_valid, config)

    def last_states(self):
        b, t = self.states.shape[:2]
        index = torch.arange(t, device=self.states.device).expand(b, t).masked_fill(~self.valid_times, -1).amax(dim=1)
        return self.states[torch.arange(b, device=self.states.device), index].masked_fill(~self.agent_mask[..., None], 0)

    def crop(self, maximum):
        # Independently keep each sample's most recent observations; pack left
        # padding so short episodes do not disappear next to longer episodes.
        b, t, n, s = self.states.shape
        lengths = self.valid_times.sum(dim=1).clamp(max=maximum)
        width = min(maximum, int(lengths.max()))
        last = torch.arange(t, device=self.states.device).expand(b, t).masked_fill(~self.valid_times, -1).amax(dim=1)
        index = last[:, None] - torch.arange(width-1, -1, -1, device=self.states.device)
        valid = torch.arange(width, device=self.states.device)[None] >= width - lengths[:, None]
        states = self.states.gather(1, index.clamp_min(0)[..., None, None].expand(b, width, n, s))
        states = states.masked_fill(~(valid[..., None] & self.agent_mask[:, None])[..., None], 0)
        if width > 1:
            action_index = (index[:, 1:] - 1).clamp(0, t-2)
            actions = self.actions.gather(1, action_index[..., None, None].expand(b, width-1, n, self.actions.shape[-1]))
            action_valid = valid[:, 1:] & valid[:, :-1]
            actions = actions.masked_fill(~(action_valid[..., None] & self.agent_mask[:, None])[..., None], 0)
        else:
            actions = self.actions[:, :0]
        return HistoryBatch(states, actions, self.agent_mask, valid)


@dataclass
class TrajectoryBatch:
    states: Tensor                      # [B, L+1, N, S]
    actions: Tensor                     # [B, L, N, C]
    rewards: Tensor                     # [B, L]
    terminated: Tensor                  # [B, L]
    truncated: Tensor                   # [B, L]
    agent_mask: Tensor                  # [B, N]
    task: TaskBatch
    transition_mask: Tensor | None = None  # [B, L], right padding only

    @property
    def valid_transitions(self):
        return self.transition_mask if self.transition_mask is not None else torch.ones_like(self.terminated)

    def to(self, device):
        return type(self)(**{f.name: getattr(self, f.name).to(device)
                            if getattr(self, f.name) is not None else None for f in fields(self)})

    def select(self, index):
        return type(self)(**{f.name: (self.task.select(index) if f.name == "task" else getattr(self, f.name)[index])
                            if getattr(self, f.name) is not None else None for f in fields(self)})

    def history(self, step):
        valid = torch.cat((torch.ones_like(self.valid_transitions[:, :1]), self.valid_transitions), dim=1)
        return HistoryBatch(self.states[:, :step+1], self.actions[:, :step], self.agent_mask, valid[:, :step+1])

    def validate(self, config):
        if self.actions.ndim != 4:
            raise ValueError("trajectory actions must be [B, L, N, C]")
        b, length, n, c = self.actions.shape
        if length < 1:
            raise ValueError("trajectory needs at least one transition")
        _shape(self.states, (b, length+1, n, config.state_dim), "trajectory states")
        _shape(self.actions, (b, length, n, config.action_components), "trajectory actions")
        for value, name in ((self.terminated, "terminated"), (self.truncated, "truncated"), (self.valid_transitions, "transition_mask")):
            _bool_mask(value, (b, length), name)
        mask = self.valid_transitions
        if not mask[:, 0].all() or (mask[:, 1:] & ~mask[:, :-1]).any():
            raise ValueError("trajectory padding must be on the right")
        boundary = self.terminated | self.truncated
        if (self.terminated & self.truncated & mask).any():
            raise ValueError("a transition cannot be both terminated and truncated")
        if (boundary[:, :-1] & mask[:, 1:]).any():
            raise ValueError("a trajectory cannot cross an episode boundary")
        _shape(self.rewards, (b, length), "rewards")
        _finite_real(self.rewards, mask, "rewards")
        self.history(length).validate(config)
        self.task.validate(config, self.agent_mask)
