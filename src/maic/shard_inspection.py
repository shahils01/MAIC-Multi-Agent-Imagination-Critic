"""Read-only schema/transition audit. Does not implement a training dataloader.

Archives are read in place; nothing is extracted or executed. NumPy object
arrays are rejected. The geometry checks are explicitly inferred from these
sample shards, not a substitute for a producer's schema specification.
"""

import argparse
from collections import Counter, OrderedDict
import hashlib
import io
import json
from pathlib import Path
import tarfile


def inspect_shard(path):
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("Shard inspection requires numpy; install maic[inspect].") from exc

    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""):
            digest.update(chunk)
    specs = {}
    episodes = OrderedDict()
    reset_keys, terminal_records, aliases = [], [], []
    bad_actions, root_keys, agent_keys = [], set(), set()
    geometry = {label: {"records": 0, "max_distance_error": 0., "max_sin_cos_error": 0.}
                for label in ("nonterminal", "terminal")}
    info_types, alive_counts = Counter(), Counter()
    field_counts = Counter()
    scale_error, scale_elements = 0., 0
    duplicate_states, state_records, caption_rewards = 0, 0, 0
    any_disagreements, all_disagreements, infinite_info_values = 0, 0, 0
    safe_pairs = 0
    previous = None

    with tarfile.open(path, "r:*") as archive:
        samples = OrderedDict()
        for member in archive:
            if not member.isfile():
                continue
            if "." not in member.name:
                raise ValueError(f"Member has no sample/field separator: {member.name}")
            key, suffix = member.name.split(".", 1)
            fields = samples.setdefault(key, {})
            if suffix in fields:
                raise ValueError(f"Duplicate member for {key}.{suffix}")
            fields[suffix] = member
            field_counts[suffix] += 1

        for key, fields in samples.items():
            if "info.json" not in fields:
                raise ValueError(f"Missing info.json for {key}")
            meta = json.load(archive.extractfile(fields["info.json"]))
            root_keys.update(meta)
            episode, step = meta["episode_id"], meta["step_id"]
            info = meta.get("infos")
            info_types[type(info).__name__] += 1
            is_reset = isinstance(info, dict) and info.get("phase") == "reset"
            done = bool(meta.get("done", False))
            agents = info if isinstance(info, list) else []
            for agent in agents:
                agent_keys.update(agent)
                infinite_info_values += sum(isinstance(value, (int, float)) and not np.isfinite(value)
                                            for value in agent.values())
            truncated = any(bool(agent.get("truncated", False)) for agent in agents)
            if agents:
                alive_counts[sum(bool(agent.get("alive", False)) for agent in agents)] += 1
            if is_reset:
                reset_keys.append(key)
            if done:
                terminal_records.append({"key": key, "truncated": truncated})

            arrays = {}
            for suffix, member in fields.items():
                if not suffix.endswith(".npy"):
                    continue
                try:
                    value = np.load(io.BytesIO(archive.extractfile(member).read()), allow_pickle=False)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"Invalid numeric array {key}.{suffix}: {exc}") from exc
                if value.dtype.kind not in "biuf":
                    raise ValueError(f"Unsupported numeric dtype in {key}.{suffix}: {value.dtype}")
                arrays[suffix] = value
                spec = specs.setdefault(suffix, {"variants": Counter(), "nonfinite_elements": 0,
                    "nonfinite_records": 0, "finite_min": None, "finite_max": None})
                spec["variants"][(tuple(value.shape), str(value.dtype))] += 1
                finite = np.isfinite(value)
                invalid = int((~finite).sum())
                spec["nonfinite_elements"] += invalid
                spec["nonfinite_records"] += int(invalid > 0)
                if finite.any():
                    low, high = float(value[finite].min()), float(value[finite].max())
                    spec["finite_min"] = low if spec["finite_min"] is None else min(spec["finite_min"], low)
                    spec["finite_max"] = high if spec["finite_max"] is None else max(spec["finite_max"], high)

            raw, env = arrays.get("actions.npy"), arrays.get("env_actions.npy")
            if raw is not None and not np.isfinite(raw).all() and not is_reset:
                bad_actions.append(key)
            if raw is not None and env is not None and raw.shape == env.shape:
                finite = np.isfinite(raw) & np.isfinite(env)
                if finite.any():
                    scale_error = max(scale_error, float(np.abs(env[finite]-100*raw[finite]).max()))
                    scale_elements += int(finite.sum())
            state = arrays.get("state.npy")
            duplicated = state is not None and state.ndim == 2 and np.array_equal(state, np.broadcast_to(state[:1], state.shape))
            state_records += int(state is not None)
            duplicate_states += int(duplicated)
            dones = arrays.get("dones.npy")
            if dones is not None:
                any_disagreements += int(bool(dones.any()) != done)
                all_disagreements += int(bool(dones.all()) != done)
            if "caption.txt" in fields:
                caption = archive.extractfile(fields["caption.txt"]).read().decode("utf-8", errors="replace")
                caption_rewards += int("reward" in caption.lower())

            # Test the inferred N-agent x 8-feature layout against stored goals.
            # Large errors are retained as evidence; they never alter the data.
            if duplicated and agents and state.shape[1] == len(agents)*8:
                blocks = state[0].reshape(len(agents), 8)
                if all(isinstance(agent.get("goal"), list) and len(agent["goal"]) >= 2 for agent in agents):
                    goals = np.asarray([agent["goal"][:2] for agent in agents])
                    distance = np.linalg.norm(goals-blocks[:, :2], axis=1)
                    angle = np.arctan2(goals[:, 1]-blocks[:, 1], goals[:, 0]-blocks[:, 0])-blocks[:, 2]
                    item = geometry["terminal" if done else "nonterminal"]
                    item["records"] += 1
                    item["max_distance_error"] = max(item["max_distance_error"], float(np.abs(distance-blocks[:, 3]).max()))
                    trig = np.stack((np.sin(angle), np.cos(angle)), axis=-1)
                    item["max_sin_cos_error"] = max(item["max_sin_cos_error"], float(np.abs(trig-blocks[:, 4:6]).max()))

            segment = episodes.setdefault(episode, {"episode_id": episode, "steps": [], "has_reset": False,
                                                   "ending_done": False, "ending_truncated": False})
            segment["steps"].append(step)
            segment["has_reset"] |= is_reset
            segment["ending_done"], segment["ending_truncated"] = done, truncated
            image = archive.extractfile(fields["image.png"]).read() if "image.png" in fields and (is_reset or done) else None
            if previous is not None:
                if previous["done"] and is_reset and previous["episode"] != episode:
                    aliases.append({"terminal_key": previous["key"], "reset_key": key,
                        "state_equal": state is not None and previous["state"] is not None and bool(np.array_equal(previous["state"], state)),
                        "obs_equal": previous["obs"] is not None and arrays.get("obs.npy") is not None and bool(np.array_equal(previous["obs"], arrays["obs.npy"])),
                        "image_equal": previous["image"] is not None and image is not None and previous["image"] == image})
                consecutive = previous["episode"] == episode and step == previous["step"]+1
                if consecutive and not previous["done"] and not done and not is_reset and raw is not None and np.isfinite(raw).all():
                    safe_pairs += 1
            previous = {"key": key, "episode": episode, "step": step, "done": done,
                        "state": state, "obs": arrays.get("obs.npy"), "image": image}

    for spec in specs.values():
        spec["variants"] = [{"shape": list(shape), "dtype": dtype, "records": count}
                            for (shape, dtype), count in spec["variants"].items()]
    segments = []
    for segment in episodes.values():
        steps = segment.pop("steps")
        segments.append({**segment, "first_step": min(steps), "last_step": max(steps), "records": len(steps),
                         "contiguous": all(b == a+1 for a, b in zip(steps, steps[1:]))})
    return {"filename": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest(),
        "records": len(samples), "field_counts": dict(field_counts), "arrays": specs,
        "episode_segments": segments, "reset_keys": reset_keys, "terminal_records": terminal_records,
        "info_container_types": dict(info_types), "root_metadata_fields": sorted(root_keys),
        "agent_metadata_fields": sorted(agent_keys), "live_agent_counts": dict(alive_counts),
        "nonfinite_actions_outside_reset": bad_actions, "nonfinite_top_level_agent_metadata_values": infinite_info_values,
        "duplicated_global_state_rows": {"matching_records": duplicate_states, "state_records": state_records},
        "action_scale_check": {"candidate_env_scale": 100, "finite_elements_checked": scale_elements, "max_absolute_error": scale_error},
        "done_reduction_disagreements": {"any_agent": any_disagreements, "all_agents": all_disagreements},
        "captions_containing_reward": caption_rewards, "terminal_reset_aliases": aliases,
        "inferred_goal_geometry": geometry, "nonterminal_consecutive_pairs": safe_pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"schema_version": 1, "scope": "read-only sample audit; transition timing is not producer-verified",
              "shards": [inspect_shard(path) for path in args.shards]}
    text = json.dumps(report, indent=2, allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
