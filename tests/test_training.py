from dataclasses import replace

import pytest
import torch

from maic import LossWeights, MAICConfig, MAICTrainer


def test_all_objectives_backward_and_optimizer_ema_update(model, batch):
    trainer = MAICTrainer(model, target_decay=.5)
    old = {name: p.detach().clone() for name, p in model.named_parameters()}
    metrics = trainer.train_step(batch, context_step=1, horizon=2, num_rollouts=2, multi_step_horizon=2)
    assert set(("action", "dynamics", "reward", "termination", "tail", "multi_step", "imagination")) <= metrics.keys()
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert any(not torch.equal(parameter, old[name]) for name, parameter in model.named_parameters())
    for name, parameter in trainer.target.named_parameters():
        torch.testing.assert_close(parameter, .5*old[name]+.5*dict(model.named_parameters())[name])
        assert not parameter.requires_grad and parameter.grad is None
    for prefix in ("policy_encoder", "scene_encoder", "action_decoder", "world_model", "value_head"):
        gradients = [p.grad for name, p in model.named_parameters() if name.startswith(prefix) and p.grad is not None]
        assert gradients and any(g.abs().sum() > 0 for g in gradients)
        assert all(torch.isfinite(g).all() for g in gradients)


def test_real_continuation_uses_prefix_context(model, batch):
    trainer = MAICTrainer(model)
    expected = model.encode(batch.history(1), batch.task)[1]
    contexts = []
    hook = model.action_decoder.register_forward_pre_hook(lambda module, inputs: contexts.append(inputs[1]))
    trainer.loss(batch, context_step=1, horizon=1, num_rollouts=1)
    hook.remove()
    for context in contexts[:3]:
        torch.testing.assert_close(context, expected)


def test_padded_episodes_and_agents_are_ignored_in_all_losses(model, batch):
    batch.transition_mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    batch.terminated[0] = torch.tensor([False, True, False, False])
    batch.agent_mask[0, -1] = False
    state_valid = batch.history(4).valid_times[..., None] & batch.agent_mask[:, None]
    action_valid = batch.valid_transitions[..., None] & batch.agent_mask[:, None]
    batch.states = batch.states.masked_fill(~state_valid[..., None], float("nan"))
    batch.actions = batch.actions.masked_fill(~action_valid[..., None], -999)
    batch.rewards = batch.rewards.masked_fill(~batch.valid_transitions, float("nan"))
    batch.task.agent_parameters = batch.task.agent_parameters.masked_fill(~batch.agent_mask[..., None], float("nan"))
    trainer = MAICTrainer(model)
    metrics = trainer.train_step(batch, context_step=1, horizon=2, num_rollouts=2)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())


def test_episode_boundaries_cannot_be_crossed(model, batch):
    batch.terminated[0, 1] = True
    with pytest.raises(ValueError, match="boundary"):
        MAICTrainer(model).loss(batch)


def test_truncation_bootstraps_target_at_last_recorded_state(model, batch):
    trainer = MAICTrainer(model)
    with torch.no_grad():
        trainer.target.value_head[-1].weight.zero_()
        trainer.target.value_head[-1].bias.fill_(10.)
    batch.terminated.zero_()
    batch.truncated[:, -1] = True
    targets = trainer.targets(batch, 1)
    torch.testing.assert_close(targets[:, -1], batch.rewards[:, -1]+model.config.gamma*10)


@pytest.mark.parametrize("kwargs", [dict(d_model=15), dict(action_bins=(1,)), dict(history_length=0),
                                    dict(gamma=1.), dict(dropout=float("nan")), dict(state_dim=-1)])
def test_invalid_architecture_config(kwargs):
    with pytest.raises(ValueError):
        MAICConfig(**{"state_dim": 2, **kwargs})


def test_negative_loss_weight_rejected():
    with pytest.raises(ValueError):
        LossWeights(imagination=-1)
