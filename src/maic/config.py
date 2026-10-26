"""Small, serializable architecture and training configurations."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class MAICConfig:
    state_dim: int
    action_bins: tuple[int, ...] = (5,)
    num_tasks: int = 2
    task_parameter_dim: int = 0
    agent_task_dim: int = 0
    num_roles: int = 1
    d_model: int = 128
    num_heads: int = 4
    spatial_layers: int = 2
    temporal_layers: int = 2
    decoder_layers: int = 2
    world_layers: int = 2
    history_length: int = 16
    dropout: float = 0.0
    gamma: float = 0.99

    def __post_init__(self):
        object.__setattr__(self, "action_bins", tuple(self.action_bins))
        for name in ("state_dim", "num_tasks", "num_roles", "d_model", "num_heads",
                     "spatial_layers", "temporal_layers", "decoder_layers",
                     "world_layers", "history_length"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("task_parameter_dim", "agent_task_dim"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not self.action_bins or any(not isinstance(k, int) or k < 2 for k in self.action_bins):
            raise ValueError("action_bins must contain >=2 categories per component")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if not math.isfinite(self.gamma) or not 0 <= self.gamma < 1:
            raise ValueError("gamma must be finite and in [0, 1)")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")

    @property
    def action_components(self) -> int:
        return len(self.action_bins)


@dataclass(frozen=True)
class LossWeights:
    action: float = 1.0
    dynamics: float = 1.0
    reward: float = 1.0
    termination: float = 1.0
    tail: float = 1.0
    imagination: float = 1.0
    multi_step: float = 0.5

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in vars(self).values()):
            raise ValueError("loss weights must be finite and nonnegative")
