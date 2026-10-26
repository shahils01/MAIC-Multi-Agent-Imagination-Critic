"""One optimizer step over tensor trajectories, without a data loader."""

from copy import deepcopy
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .config import LossWeights
from .returns import discounted_returns


@dataclass
class LossOutput:
    total: torch.Tensor
    terms: dict[str, torch.Tensor]

    def metrics(self):
        return {"total": float(self.total.detach()), **{k: float(v.detach()) for k, v in self.terms.items()}}


class MAICTrainer:
    def __init__(self, model, optimizer=None, *, weights=None, learning_rate=3e-4,
                 target_decay=0.995, gradient_clip=1.0):
        if not 0 <= target_decay < 1 or not math.isfinite(gradient_clip) or gradient_clip <= 0:
            raise ValueError("invalid EMA decay or gradient clip")
        self.model = model
        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.weights = weights or LossWeights()
        self.target_decay = target_decay
        self.gradient_clip = gradient_clip
        self.target = deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update_target(self):
        # Named parameters come from the same unwrapped model structure; do not
        # mix DDP-prefixed names with an unwrapped EMA dictionary.
        online = dict(self.model.named_parameters())
        for name, parameter in self.target.named_parameters():
            parameter.mul_(self.target_decay).add_(online[name], alpha=1-self.target_decay)
        for target_buffer, online_buffer in zip(self.target.buffers(), self.model.buffers()):
            target_buffer.copy_(online_buffer)

    @torch.no_grad()
    def targets(self, batch, context_step):
        _, context, _, _ = self.target.encode(batch.history(context_step), batch.task)
        global_task, agent_task = self.target.task_encoder(batch.task, batch.agent_mask)
        last_scene = self.target.scene_encoder(batch.history(batch.actions.shape[1]), global_task, agent_task)
        bootstrap = self.target.value(last_scene, context, global_task)
        return discounted_returns(batch.rewards, batch.terminated, bootstrap,
                                  self.model.config.gamma, batch.valid_transitions)

    def loss(self, batch, *, context_step=1, horizon=3, num_rollouts=4, multi_step_horizon=3,
             generator=None):
        batch.validate(self.model.config)
        length = batch.actions.shape[1]
        if not isinstance(context_step, int) or context_step < 0 or context_step >= int(batch.valid_transitions.sum(dim=1).min()):
            raise ValueError("context_step must precede a real transition in every sample")
        if multi_step_horizon < 1:
            raise ValueError("multi_step_horizon must be positive")
        targets = self.targets(batch, context_step)
        prefix = batch.history(context_step)
        _, context, _, _ = self.model.encode(prefix, batch.task)
        names = ("action", "dynamics", "reward", "termination", "tail", "multi_step")
        # Every term remains differentiable even if a particular batch has only
        # one valid context/continuation point.
        sums = {name: context.sum()*0 for name in names}
        counts = {name: 0 for name in names}
        for step in range(context_step, length):
            active = batch.valid_transitions[:, step]
            if not active.any():
                continue
            current = batch.select(active)
            task = current.task
            mask = current.agent_mask
            c = context[active]
            global_task, agent_task = self.model.task_encoder(task, mask)
            scene = self.model.scene_encoder(current.history(step), global_task, agent_task)
            actions = current.actions[:, step]
            logits = self.model.action_decoder(scene, c, global_task, mask, actions)
            labels = actions[mask].reshape(-1)
            sums["action"] = sums["action"] + F.cross_entropy(logits[mask].reshape(-1, logits.shape[-1]), labels, reduction="sum")
            counts["action"] += labels.numel()
            states = current.states[:, step]
            following = current.states[:, step+1]
            world = self.model.world_model(states, actions, mask, task.role_ids(mask), global_task, agent_task)
            sums["dynamics"] = sums["dynamics"] + F.mse_loss(world.next_states[mask], following[mask], reduction="sum")
            counts["dynamics"] += following[mask].numel()
            reward, term = self.model.world_model.outcome_prediction(states, actions, following, mask, global_task, agent_task)
            sums["reward"] = sums["reward"] + F.mse_loss(reward, current.rewards[:, step], reduction="sum")
            sums["termination"] = sums["termination"] + F.binary_cross_entropy_with_logits(term, current.terminated[:, step].float(), reduction="sum")
            value = self.model.value(scene, c, global_task)
            sums["tail"] = sums["tail"] + F.mse_loss(value, targets[active, step], reduction="sum")
            for name in ("reward", "termination", "tail"):
                counts[name] += int(active.sum())

        # Multi-step physical prediction uses logged actions, isolating world
        # accuracy from errors in the learned behavior decoder.
        global_task, agent_task = self.model.task_encoder(batch.task, batch.agent_mask)
        predicted = batch.states[:, context_step].masked_fill(~batch.agent_mask[..., None], 0)
        for step in range(context_step, min(length, context_step+multi_step_horizon)):
            active = batch.valid_transitions[:, step]
            actions = batch.actions[:, step].masked_fill(~(active[:, None] & batch.agent_mask)[..., None], 0)
            world = self.model.world_model(predicted, actions, batch.agent_mask, batch.task.role_ids(batch.agent_mask), global_task, agent_task)
            predicted = world.next_states
            valid = active[:, None] & batch.agent_mask
            if valid.any():
                sums["multi_step"] = sums["multi_step"] + F.mse_loss(predicted[valid], batch.states[:, step+1][valid], reduction="sum")
                counts["multi_step"] += batch.states[:, step+1][valid].numel()
        terms = {name: sums[name] / max(counts[name], 1) for name in names}
        forecast = self.model.imagine(prefix, batch.task, horizon=horizon, num_rollouts=num_rollouts, generator=generator)
        terms["imagination"] = F.mse_loss(forecast.mean, targets[:, context_step])
        total = sum(getattr(self.weights, name)*term for name, term in terms.items())
        return LossOutput(total, terms)

    def train_step(self, batch, **kwargs):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        output = self.loss(batch, **kwargs)
        if not torch.isfinite(output.total):
            raise FloatingPointError("nonfinite training loss")
        output.total.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip, error_if_nonfinite=True)
        self.optimizer.step()
        self.update_target()
        return {**output.metrics(), "gradient_norm": float(norm)}
