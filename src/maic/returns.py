"""Aligned return arithmetic; true terminals and truncations are distinct."""

import torch


def discounted_returns(rewards, terminated, bootstrap, gamma, mask=None):
    """Backward n-step/MC targets for [B, L] real transitions.

    Bootstrap is V(s_L), not V(s_(L-1)). For a true terminal its contribution
    is removed by terminated. Administrative truncations do not set terminated.
    Right-padded entries are skipped; they do not consume a discount step.
    """
    if rewards.ndim != 2 or rewards.shape != terminated.shape:
        raise ValueError("rewards/terminated must have matching [B, L] shapes")
    if bootstrap.shape != rewards.shape[:1] or not 0 <= gamma < 1:
        raise ValueError("incorrect bootstrap shape or gamma")
    if terminated.dtype != torch.bool:
        raise ValueError("terminated must be boolean")
    if mask is None:
        mask = torch.ones_like(terminated)
    if mask.shape != rewards.shape or mask.dtype != torch.bool:
        raise ValueError("mask must be boolean with shape [B, L]")
    if not mask[:, 0].all() or (mask[:, 1:] & ~mask[:, :-1]).any():
        raise ValueError("mask must have real transitions followed by right padding")
    if not torch.isfinite(rewards[mask]).all() or not torch.isfinite(bootstrap).all():
        raise ValueError("return inputs must be finite at valid positions")
    following = bootstrap
    targets = []
    for step in range(rewards.shape[1]-1, -1, -1):
        reward = torch.where(mask[:, step], rewards[:, step], torch.zeros_like(following))
        value = reward + gamma * (~terminated[:, step]).to(reward.dtype) * following
        following = torch.where(mask[:, step], value, following)
        targets.append(torch.where(mask[:, step], following, torch.zeros_like(following)))
    return torch.stack(targets[::-1], dim=1)
