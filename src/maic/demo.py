"""Synthetic sanity check, not a trajectory dataloader or benchmark."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from .batch import TaskBatch, TrajectoryBatch
from .config import MAICConfig
from .model import MAIC
from .training import MAICTrainer


def synthetic_batch(config, batch_size=4, length=6, num_agents=3, seed=0):
    """Two explicit task objectives, mixed competent/poor toy controllers.

    Two state features are 2-D position. Actions are stay/right/left/up/down.
    This fixture intentionally has no relationship to a real dataset schema.
    """
    if config.state_dim != 2 or config.action_bins != (5,) or config.num_tasks != 2 or config.agent_task_dim != 2:
        raise ValueError("synthetic fixture needs 2-D positions, five actions, two tasks and 2-D goals")
    generator = torch.Generator().manual_seed(seed)
    state = torch.rand(batch_size, num_agents, 2, generator=generator)*1.5-0.75
    task_ids = torch.arange(batch_size) % 2
    goal = torch.zeros(batch_size, num_agents, 2)
    goal[task_ids == 0, :, 0] = 0.8
    # Formation: known per-role offsets about a fixed center.
    goal[task_ids == 1, :, 0] = torch.linspace(-0.5, 0.5, num_agents)
    goal[task_ids == 1, :, 1] = 0.5
    steps = torch.tensor([[0., 0.], [0.1, 0.], [-0.1, 0.], [0., 0.1], [0., -0.1]])
    states, actions, rewards = [state], [], []
    for _ in range(length):
        distance = ((state[:, :, None] + steps[None, None] - goal[:, :, None])**2).sum(dim=-1)
        competent = distance.argmin(dim=-1)
        random = torch.randint(5, (batch_size, num_agents), generator=generator)
        good = (torch.arange(batch_size) % 4 < 2)[:, None]
        action = torch.where(good, competent, random)
        state = state + steps[action]
        reward = -((state-goal)**2).sum(dim=-1).mean(dim=-1)
        states.append(state)
        actions.append(action[..., None])
        rewards.append(reward)
    terminated = torch.zeros(batch_size, length, dtype=torch.bool)
    terminated[:, -1] = True
    return TrajectoryBatch(torch.stack(states, dim=1), torch.stack(actions, dim=1), torch.stack(rewards, dim=1),
        terminated, torch.zeros_like(terminated), torch.ones(batch_size, num_agents, dtype=torch.bool),
        TaskBatch(task_ids, agent_parameters=goal))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    config = MAICConfig(state_dim=2, agent_task_dim=2, d_model=32, num_heads=4,
                        spatial_layers=1, temporal_layers=1, decoder_layers=1,
                        world_layers=1, history_length=8)
    model = MAIC(config).to(args.device)
    trainer = MAICTrainer(model)
    batch = synthetic_batch(config, seed=args.seed).to(args.device)
    for step in range(args.steps):
        metrics = trainer.train_step(batch, context_step=2, horizon=3, num_rollouts=4)
        if step == 0 or step == args.steps-1:
            print(json.dumps({"step": step+1, **metrics}))
    model.eval()
    with torch.no_grad():
        forecast = model.imagine(batch.history(2), batch.task, horizon=3, num_rollouts=8)
    print(json.dumps({"predicted_returns": forecast.mean.cpu().tolist(),
                      "rollout_std": forecast.variance.sqrt().cpu().tolist(),
                      "note": "Synthetic execution check; these are not calibrated performance estimates."}))
    if args.checkpoint:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(config), "model": model.state_dict(),
                    "target_model": trainer.target.state_dict(), "optimizer": trainer.optimizer.state_dict(),
                    "steps": args.steps, "seed": args.seed, "synthetic_demo": True}, args.checkpoint)


if __name__ == "__main__":
    main()
