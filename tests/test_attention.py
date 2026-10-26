from dataclasses import replace

import pytest
import torch

from maic import HistoryBatch, MAIC, TaskBatch


def test_shapes_and_multiple_action_components(config, batch):
    config = replace(config, action_bins=(3, 5))
    model = MAIC(config).eval()
    history = batch.history(2)
    history.actions = torch.cat((history.actions % 3, history.actions), dim=-1)
    labels = torch.zeros(2, 3, 2, dtype=torch.long)
    output = model(history, batch.task, labels)
    assert output["value"].shape == (2,)
    assert output["action_logits"].shape == (2, 3, 2, 5)
    assert torch.isneginf(output["action_logits"][:, :, 0, 3:]).all()
    prediction = model.imagine(history, batch.task, horizon=2, num_rollouts=3)
    assert prediction.actions.shape == (2, 3, 2, 3, 2)
    assert (prediction.actions[..., 0] < 3).all()
    assert (prediction.actions[..., 1] < 5).all()


def test_logged_current_or_future_action_cannot_change_earlier_logits(model, batch):
    scene, context, task, _ = model.encode(batch.history(2), batch.task)
    first = torch.zeros(2, 3, 1, dtype=torch.long)
    changed = first.clone()
    changed[:, 1:] = 4
    logits1 = model.action_decoder(scene, context, task, batch.agent_mask, first)
    logits2 = model.action_decoder(scene, context, task, batch.agent_mask, changed)
    torch.testing.assert_close(logits1[:, :2], logits2[:, :2])
    # Changing the first action must not change its OWN prediction.
    changed[:, 0] = 3
    logits3 = model.action_decoder(scene, context, task, batch.agent_mask, changed)
    torch.testing.assert_close(logits1[:, 0], logits3[:, 0])


def test_scene_and_world_are_permutation_equivariant(model, batch):
    history = batch.history(2)
    order = torch.tensor([2, 0, 1])
    permuted = HistoryBatch(history.states[:, :, order], history.actions[:, :, order], history.agent_mask[:, order])
    task = TaskBatch(batch.task.ids, agent_parameters=batch.task.agent_parameters[:, order])
    scene, context, global_task, agents = model.encode(history, batch.task)
    pscene, pcontext, ptask, pagents = model.encode(permuted, task)
    torch.testing.assert_close(pscene.agents, scene.agents[:, order], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(pscene.team, scene.team, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(pcontext, context, atol=2e-6, rtol=2e-5)
    actions = batch.actions[:, 2]
    world = model.world_model(history.last_states(), actions, history.agent_mask, batch.task.role_ids(history.agent_mask), global_task, agents)
    pworld = model.world_model(permuted.last_states(), actions[:, order], permuted.agent_mask, task.role_ids(permuted.agent_mask), ptask, pagents)
    torch.testing.assert_close(pworld.next_states, world.next_states[:, order], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(pworld.reward, world.reward, atol=2e-6, rtol=2e-5)


def test_nan_padding_and_invalid_padded_labels_do_not_change_real_outputs(model, batch):
    history = batch.history(2)
    baseline = model(history, batch.task, batch.actions[:, 2])
    b, t, n, s = history.states.shape
    padded = HistoryBatch(torch.cat((history.states, torch.full((b, t, 1, s), float("nan"))), dim=2),
        torch.cat((history.actions, torch.full((b, t-1, 1, 1), -999, dtype=torch.long)), dim=2),
        torch.cat((history.agent_mask, torch.zeros(b, 1, dtype=torch.bool)), dim=1))
    task = TaskBatch(batch.task.ids, agent_parameters=torch.cat((batch.task.agent_parameters, torch.full((b, 1, 2), float("nan"))), dim=1))
    labels = torch.cat((batch.actions[:, 2], torch.full((b, 1, 1), -999, dtype=torch.long)), dim=1)
    output = model(padded, task, labels)
    torch.testing.assert_close(output["value"], baseline["value"], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(output["action_logits"][:, :n], baseline["action_logits"], atol=2e-6, rtol=2e-5)
    sum(output["value"]).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_left_time_padding_is_invariant_and_has_finite_backward(model, batch):
    history = batch.history(2)
    baseline = model(history, batch.task)["value"]
    b, t, n, s = history.states.shape
    padded = HistoryBatch(torch.cat((torch.full((b, 1, n, s), float("nan")), history.states), dim=1),
        torch.cat((torch.full((b, 1, n, 1), -999, dtype=torch.long), history.actions), dim=1),
        history.agent_mask, torch.cat((torch.zeros(b, 1, dtype=torch.bool), torch.ones(b, t, dtype=torch.bool)), dim=1))
    output = model(padded, batch.task)["value"]
    torch.testing.assert_close(output, baseline)
    output.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_variable_history_lengths_crop_independently(model, batch):
    history = batch.history(4)
    valid = torch.tensor([[True, True, False, False, False], [True, True, True, True, True]])
    history.time_mask = valid
    history.states = history.states.masked_fill(~valid[..., None, None], float("nan"))
    history.actions = history.actions.masked_fill(~(valid[:, :-1] & valid[:, 1:])[..., None, None], -999)
    result = model(history, batch.task)["value"]
    first = HistoryBatch(history.states[:1, :2], history.actions[:1, :1], history.agent_mask[:1])
    expected = model(first, batch.task.select(slice(0, 1)))["value"]
    torch.testing.assert_close(result[:1], expected, atol=2e-6, rtol=2e-5)


def test_single_agent_single_observation(config):
    config = replace(config, agent_task_dim=0)
    model = MAIC(config)
    history = HistoryBatch(torch.zeros(1, 1, 1, 2), torch.zeros(1, 0, 1, 1, dtype=torch.long), torch.ones(1, 1, dtype=torch.bool))
    output = model.imagine(history, TaskBatch(torch.zeros(1, dtype=torch.long)), horizon=1, num_rollouts=1)
    assert torch.isfinite(output.mean).all()
    assert output.variance.item() == 0


def test_role_and_parameter_inputs_are_required_and_checked(config, batch):
    model = MAIC(config)
    with pytest.raises(ValueError, match="agent task parameters"):
        model(batch.history(2), TaskBatch(batch.task.ids))
    with pytest.raises(ValueError, match="task id"):
        model(batch.history(2), TaskBatch(torch.full((2,), 99, dtype=torch.long), agent_parameters=batch.task.agent_parameters))
