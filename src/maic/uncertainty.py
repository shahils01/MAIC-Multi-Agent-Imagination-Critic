"""Empirical rollout spread and ensemble disagreement, without calibration claims."""

from dataclasses import dataclass

import torch


@dataclass
class EnsemblePrediction:
    mean: torch.Tensor
    variance: torch.Tensor
    rollout_variance: torch.Tensor
    model_variance: torch.Tensor
    member_means: torch.Tensor


@torch.no_grad()
def predict_ensemble(models, history, task, **imagination_options):
    if not models:
        raise ValueError("provide at least one model")
    if any(model.config != models[0].config for model in models):
        raise ValueError("ensemble members must use the same configuration")
    means, variances = [], []
    for model in models:
        # Restore the caller's exact submodule modes, including deliberately
        # frozen/eval submodules in a training model.
        modes = {module: module.training for module in model.modules()}
        try:
            model.eval()
            output = model.imagine(history, task, **imagination_options)
            means.append(output.mean)
            variances.append(output.variance)
        finally:
            for module, training in modes.items():
                module.training = training
    means = torch.stack(means)
    within = torch.stack(variances).mean(dim=0)
    between = means.var(dim=0, unbiased=False)
    return EnsemblePrediction(means.mean(dim=0), within+between, within, between, means)
