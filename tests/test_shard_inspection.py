import io
import json
import tarfile

import pytest

np = pytest.importorskip("numpy")
from maic.shard_inspection import inspect_shard


def npy(value):
    out = io.BytesIO()
    np.save(out, value)
    return out.getvalue()


def write_shard(path, *, nonfinite_regular_action=False, object_array=False, duplicate=False):
    def state(x, goals):
        poses = np.array([[x, 0.], [x, 1.]], dtype=np.float32)
        angle = np.arctan2(goals[:, 1]-poses[:, 1], goals[:, 0]-poses[:, 0])
        blocks = np.zeros((2, 8), dtype=np.float32)
        blocks[:, :2] = poses
        blocks[:, 3] = np.linalg.norm(goals-poses, axis=1)
        blocks[:, 4], blocks[:, 5] = np.sin(angle), np.cos(angle)
        return np.tile(blocks.reshape(1, -1), (2, 1))

    old_goals = np.array([[3., 0.], [3., 1.]])
    new_goals = np.array([[10., 0.], [10., 1.]])
    reset_state = state(7., new_goals)
    rows = [(0, 0, True, False, False, state(0., old_goals), old_goals),
            (0, 1, False, False, False, state(1., old_goals), old_goals),
            (0, 2, False, True, True, reset_state, old_goals),
            (1, 0, True, False, False, reset_state, new_goals),
            (1, 1, False, False, False, state(8., new_goals), new_goals)]
    with tarfile.open(path, "w") as archive:
        for episode, step, reset, done, truncated, shared_state, goals in rows:
            key = f"{episode:06d}_{step:06d}"
            actions = np.full((2, 2), np.nan if reset else .02, dtype=np.float32)
            if nonfinite_regular_action and episode == 0 and step == 1:
                actions[0, 0] = np.nan
            infos = {"phase": "reset"} if reset else [
                {"goal": [*goal, 0.], "alive": not done, "truncated": truncated, "min_separation": float("inf")}
                for goal in goals]
            metadata = {"episode_id": episode, "step_id": step, "done": done,
                        "reward_mean": 0. if reset else -1., "infos": infos}
            payloads = {"state.npy": npy(shared_state), "obs.npy": npy(shared_state[:, :5]),
                        "actions.npy": npy(actions), "env_actions.npy": npy(100*actions),
                        "rewards.npy": npy(np.full((2, 1), 0. if reset else -1.)),
                        "dones.npy": npy(np.full((2,), done, dtype=bool)),
                        "info.json": json.dumps(metadata).encode(), "caption.txt": b"reward_mean -1",
                        "image.png": shared_state.tobytes()}
            if object_array:
                payloads["obs.npy"] = npy(np.array([{"untrusted": "not executed"}], dtype=object))
            for suffix, payload in payloads.items():
                member = tarfile.TarInfo(f"{key}.{suffix}")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
                if duplicate and suffix == "info.json":
                    archive.addfile(member, io.BytesIO(payload))


def test_audit_identifies_reset_placeholders_scaling_and_terminal_alias(tmp_path):
    path = tmp_path/"sample.tar"
    write_shard(path)
    original = path.read_bytes()
    report = inspect_shard(path)
    assert report["records"] == 5
    assert len(report["reset_keys"]) == 2
    assert report["nonfinite_actions_outside_reset"] == []
    assert report["arrays"]["actions.npy"]["nonfinite_elements"] == 8
    assert report["action_scale_check"]["max_absolute_error"] == 0
    assert report["duplicated_global_state_rows"]["matching_records"] == 5
    assert report["terminal_records"] == [{"key": "000000_000002", "truncated": True}]
    assert report["terminal_reset_aliases"] == [{"terminal_key": "000000_000002", "reset_key": "000001_000000",
                                                  "state_equal": True, "obs_equal": True, "image_equal": True}]
    assert report["nonterminal_consecutive_pairs"] == 2
    assert report["inferred_goal_geometry"]["nonterminal"]["max_distance_error"] < 1e-6
    assert report["inferred_goal_geometry"]["terminal"]["max_distance_error"] == pytest.approx(1.)
    assert report["captions_containing_reward"] == 5
    json.dumps(report, allow_nan=False)
    assert path.read_bytes() == original


def test_real_nonfinite_action_is_not_mistaken_for_reset(tmp_path):
    path = tmp_path/"bad-action.tar"
    write_shard(path, nonfinite_regular_action=True)
    report = inspect_shard(path)
    assert report["nonfinite_actions_outside_reset"] == ["000000_000001"]
    assert report["nonterminal_consecutive_pairs"] == 1


def test_object_arrays_are_rejected_without_unpickling(tmp_path):
    path = tmp_path/"object.tar"
    write_shard(path, object_array=True)
    with pytest.raises(ValueError, match="Invalid numeric array"):
        inspect_shard(path)


def test_duplicate_sample_members_are_rejected(tmp_path):
    path = tmp_path/"duplicate.tar"
    write_shard(path, duplicate=True)
    with pytest.raises(ValueError, match="Duplicate member"):
        inspect_shard(path)
