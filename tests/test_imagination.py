from copy import deepcopy
from dataclasses import replace

import torch

from maic import MAIC, predict_ensemble


def constant_outputs(model, reward, terminal_logit, tail):
    with torch.no_grad():
        model.world_model.reward.weight.zero_()
        model.world_model.reward.bias.fill_(reward)
        model.world_model.termination.weight.zero_()
        model.world_model.termination.bias.fill_(terminal_logit)
        model.value_head[-1].weight.zero_()
        model.value_head[-1].bias.fill_(tail)


def test_imagination_has_exact_discount_and_endpoint_bootstrap(config, batch):
    model = MAIC(replace(config, gamma=.9)).eval()
    constant_outputs(model, 2., -100., 3.)
    output = model.imagine(batch.history(2), batch.task, horizon=3, num_rollouts=4)
    torch.testing.assert_close(output.returns, torch.full((2, 4), 2+1.8+1.62+.9**3*3))
    assert not output.terminated.any()


def test_true_terminal_masks_every_future_reward_and_tail(model, batch):
    constant_outputs(model, 2., 100., 999.)
    output = model.imagine(batch.history(2), batch.task, horizon=3, num_rollouts=4)
    torch.testing.assert_close(output.returns, torch.full((2, 4), 2.))
    assert output.terminated[:, :, 0].all()
    assert not output.terminated[:, :, 1:].any()
    assert (output.rewards[:, :, 1:] == 0).all()
    torch.testing.assert_close(output.states[:, :, 1], output.states[:, :, 3])


def test_policy_context_is_inferred_once_and_held_fixed(model, batch):
    inferred, used = [], []
    handle1 = model.policy_encoder.register_forward_hook(lambda module, inputs, output: inferred.append(output.team))
    handle2 = model.action_decoder.register_forward_pre_hook(lambda module, inputs: used.append(inputs[1].detach().clone()))
    output = model.imagine(batch.history(2), batch.task, horizon=3, num_rollouts=2)
    handle1.remove()
    handle2.remove()
    assert len(inferred) == 1
    expected = output.policy_context.repeat_interleave(2, dim=0)
    assert used
    for context in used:
        torch.testing.assert_close(context, expected)


def test_forecast_gradients_train_tail_and_context_but_not_sampled_world(model, batch):
    with torch.no_grad():
        model.world_model.termination.weight.zero_()
        model.world_model.termination.bias.fill_(-100.)
    forecast = model.imagine(batch.history(2), batch.task, horizon=2, num_rollouts=2)
    forecast.mean.square().mean().backward()
    assert model.value_head[-1].weight.grad is not None
    assert model.policy_projection[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.action_decoder.parameters())
    assert all(p.grad is None for p in model.world_model.parameters())


def test_ensemble_variance_decomposition_and_mode_restoration(model, batch):
    second = deepcopy(model)
    constant_outputs(model, 0., -100., 1.)
    constant_outputs(second, 0., -100., 3.)
    model.train()
    model.scene_encoder.eval()
    result = predict_ensemble([model, second], batch.history(2), batch.task, horizon=2, num_rollouts=2)
    scale = model.config.gamma**2
    torch.testing.assert_close(result.mean, torch.full((2,), 2*scale))
    torch.testing.assert_close(result.rollout_variance, torch.zeros(2))
    torch.testing.assert_close(result.model_variance, torch.full((2,), scale**2))
    torch.testing.assert_close(result.variance, result.model_variance+result.rollout_variance)
    assert model.training and not model.scene_encoder.training and not second.training


def test_sampling_is_reproducible_with_a_generator(model, batch):
    first = model.imagine(batch.history(2), batch.task, horizon=2, num_rollouts=2, generator=torch.Generator().manual_seed(12))
    second = model.imagine(batch.history(2), batch.task, horizon=2, num_rollouts=2, generator=torch.Generator().manual_seed(12))
    torch.testing.assert_close(first.returns, second.returns)
    assert torch.equal(first.actions, second.actions)
