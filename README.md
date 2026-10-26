# MAIC — Multi-Agent Imagination Critic

A task-conditioned multi-agent critic that infers team behavior from short
trajectories and imagines continuations to predict discounted return.

This first implementation uses **multi-head attention and numerical states**.
It includes imagination in training and inference. It evaluates continuation of
the observed controller; it does not optimize a new actor.

Read [PROJECT.md](PROJECT.md) for the mathematical formulation and research scope.

## What is implemented

- Separate scene and policy-context encoders: attention between agents at each
  timestep, followed by causal attention over each agent's history.
- Learned attention pooling for a team-level representation; no fixed team size.
- An autoregressive attention decoder for coordinated joint actions, including
  multiple categorical control components per agent.
- An attention-based physical state-delta model, task-conditioned reward and
  termination heads, and a learned tail-value head.
- Multiple sampled imagined continuations with discounted rewards and a
  surviving-branch tail bootstrap. Policy context is inferred once per real
  prefix and held fixed during imagination.
- Joint supervised training, multi-step physical prediction, detached return
  targets, and an EMA target model.
- Masking for padded agents/histories/episodes, distinct terminal and truncation
  handling, and optional ensemble prediction.

No dataloader, real-dataset adapter, vision encoder, or pretrained checkpoint is
included. The data interface is tensors; dataset integration is deliberately
deferred. Action bins describe categorical indices, not assumed physical units.

## Setup and synthetic execution check

Python 3.10+ and PyTorch 1.12+ are required. In your preferred environment:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
maic-demo --steps 10
```

You can also run from the checkout without installing the package:

```bash
PYTHONPATH=src python -m maic.demo --steps 10
PYTHONPATH=src python -m pytest -q
```

The demo generates a small synthetic 2-D system, runs every training objective,
and forecasts returns from short prefixes. It is an execution check, not evidence
of performance on formation-control data. It defaults to CPU. Set `--device cuda`
when an appropriate CUDA runtime is available.

An optional checkpoint is written only when requested:

```bash
maic-demo --steps 10 --checkpoint outputs/synthetic.pt
```

Checkpoints contain the configuration, model/target/optimizer state, and a flag
identifying synthetic training. No training jobs or downloads run on import.

## Tensor interface

`B` = batch size, `T` = observed states, `N` = padded agent count, `S` = state
features, and `C` = action components.

| Input | Shape / meaning |
| --- | --- |
| `HistoryBatch.states` | `[B, T, N, S]`, floating-point numerical observations |
| `HistoryBatch.actions` | `[B, T-1, N, C]`, `torch.long` executed-action indices |
| `HistoryBatch.agent_mask` | `[B, N]`, boolean; at least one real agent per sample |
| `HistoryBatch.time_mask` | Optional `[B, T]`, one contiguous observed segment |
| `TaskBatch.ids` | `[B]`, `torch.long` task identity |
| `TaskBatch.parameters` | Optional `[B, P]`, global structured task parameters |
| `TaskBatch.agent_parameters` | Optional `[B, N, Q]`, assigned goals/formation offsets |
| `TaskBatch.roles` | Optional `[B, N]`, `torch.long` role IDs; defaults to role 0 |

A history ends at state `s_t` and contains actions only through `a_(t-1)`.
The action being predicted is not an input to its own prediction. During
teacher-forced joint-action decoding, earlier-agent/component labels are visible
only to later action slots.

If parameter dimensions in the configuration are nonzero, their tensors are
required. Normalize real state/task features consistently before passing them to
MAIC. Physical parameters needed to predict dynamics belong in the state.

## Predict a return

```python
import torch
from maic import MAIC, MAICConfig, HistoryBatch, TaskBatch

config = MAICConfig(
    state_dim=8,
    action_bins=(11, 11),  # e.g. two discretized controls; choose bins for your data
    num_tasks=3,
    agent_task_dim=2,      # e.g. assigned goal or formation offset
    num_roles=2,
    d_model=128,
    num_heads=4,
)
model = MAIC(config).eval()  # randomly initialized until trained

history = HistoryBatch(
    states=torch.randn(2, 4, 3, 8),
    actions=torch.randint(11, (2, 3, 3, 2)),
    agent_mask=torch.ones(2, 3, dtype=torch.bool),
)
task = TaskBatch(
    ids=torch.tensor([0, 1]),
    agent_parameters=torch.zeros(2, 3, 2),
    roles=torch.zeros(2, 3, dtype=torch.long),
)
with torch.no_grad():
    prediction = model.imagine(history, task, horizon=5, num_rollouts=8)

print(prediction.mean)      # [B], mean imagined return
print(prediction.variance)  # [B], sampled rollout spread
print(prediction.actions.shape)  # [B, M, H, N, C]
```

`model(history, task)` exposes the learned tail value, while `model.imagine(...)`
is the imagination-based estimate. `predict_ensemble([...], history, task, ...)`
adds between-model disagreement to within-model rollout variance. Use generators
on the same device as the model for reproducible sampling.

## Train from tensors

`TrajectoryBatch` adds `rewards`, `terminated`, and `truncated`, all `[B, L]`,
alongside states `[B, L+1, N, S]`, actions `[B, L, N, C]`, task inputs, and agent
masks. Optional boolean `transition_mask` permits right-padded episodes. A sample
cannot cross a terminal/truncation boundary.

```python
from maic import MAICTrainer

trainer = MAICTrainer(model, learning_rate=3e-4)
metrics = trainer.train_step(
    trajectory_batch,
    context_step=2,       # infer behavior from states 0..2 and actions 0..1
    horizon=5,
    num_rollouts=8,
    multi_step_horizon=3,
)
```

All samples need a real transition at `context_step`. The action and tail-value
heads are supervised at subsequent real continuation points with the **same
prefix context**. Return targets bootstrap from the final recorded state, with
zero bootstrap on true termination and retained bootstrap on administrative
truncation. If a time limit is part of the task objective, mark it terminal and
include remaining time in the state.

Initial loss weights are configurable through `LossWeights`; they are starting
settings, not validated hyperparameters. Good and bad controllers both supervise
action prediction. The physical multi-step loss rolls forward using logged
actions, allowing dynamics errors to be evaluated separately from behavior errors.

## Gradient and uncertainty boundaries

Generated actions, states, rewards, and terminal draws are detached in this
version. The imagination loss differentiates through the final scene encoder,
fixed policy context, task encoder, and tail-value head. Action prediction learns
through cross-entropy; dynamics/reward/termination learn from real transitions.
There is no actor reward-maximization objective or gradient estimator through
categorical draws. This boundary is explicit and covered by tests.

The initial policy context is deterministic; a stochastic controller posterior
and distributional tail head remain extensions. Rollout spread and ensemble
disagreement are **not calibrated confidence bounds**. Accurate action prediction
or low synthetic losses alone do not establish return-prediction quality.

The scene/world encoders share parameters across agents and are permutation
equivariant, with invariant team pooling when roles/task features travel with
their agents. The autoregressive action decoder uses the supplied agent order;
its learned joint distribution is not guaranteed to be order invariant.

Real evaluation should hold out entire episodes and controller checkpoints,
compare policies from matched initial conditions, and measure return error,
ranking accuracy, selection regret, and uncertainty coverage.

## Sample-shard audit and collector contract

The two supplied shards were inspected in place: `shard-000033.tar` contains
1,000 records; `shard-000285.tar` contains 436. The machine-readable findings are
in [docs/sample_shard_audit.json](docs/sample_shard_audit.json). A read-only
inspection utility is available separately from the deferred dataloader:

```bash
python -m pip install -e '.[inspect]'
maic-inspect /path/shard-000033.tar /path/shard-000285.tar --output /tmp/audit.json
```

The matching collector is `AERO-MARL/mat/runner/shared/mujoco_runner_wds.py`.
The local source was checked at commit `ca3cd38e53cb50d27e6b62ee63da31d79df7f077`;
the exact revision that originally generated the shards is not recorded.

- Each sample has five agents. Policy actions are continuous `[5, 2]`, and
  `env_actions = 100 * actions` exactly on finite entries. Categorical MAIC heads
  need explicit discretization before these actions can be used as labels.
- `state.npy` is `[5, 40]` with five identical rows containing the joint state.
  A candidate per-agent representation is `state[0].reshape(5, 8)`, rather than
  treating the repeated rows as distinct per-agent observations. Goal-geometry
  checks support the first six fields as x/y, heading, goal distance, sine and
  cosine of relative goal bearing. The external `ManyAgent_GoTOGoal` package,
  which defines the environment itself, is not included in AERO-MARL; remaining
  field/physics semantics are not yet producer-verified.
- **Record `k` is post-step.** Pair the state in record `k-1` with the actions
  and rewards in record `k` to predict the state in record `k`. Reset records
  have deliberately missing (NaN) actions and zero rewards: keep their initial
  state, but never use those action placeholders as labels.
- Global `done` is the reduction **all agents done**, not any agent done.
  Per-agent inactivity is distinct from padding: an inactive robot remains part
  of the scene. Time-limit flags live in per-agent `info.json` metadata.
- **Terminal next observations are lost in this collection path.**
  `ShareDummyVecEnv.step_wait` resets before returning observations, while
  retaining the previous episode's rewards/dones/info. All 11 terminal/reset
  pairs available for direct comparison match in state, observation, and image.
  Do not train terminal state deltas or timeout bootstraps against those reset
  observations. Terminal rewards remain useful labels; a future adapter needs
  an explicit missing-final-state policy, not silently fabricated endpoints.
- There are 1,411 consecutive nonterminal pairs within these samples, using the
  confirmed record timing. This is a structural count, not a training-quality
  claim. Shard boundaries may split episodes; stitch only within a known data
  source and matching episode/step IDs. One shard is not one trajectory.
- Captions contain reward labels and episode/step metadata; they are not task
  instructions. Task identity and controller/checkpoint provenance require an
  external collection manifest. Do not infer policy quality from shard numbers.

The sample audit does not alter the archives or the AERO-MARL collector, and it
does not yet make the current trainer compatible with missing terminal states
or continuous raw action labels.

## Source map

| File | Responsibility |
| --- | --- |
| `src/maic/attention.py` | Multi-head self-attention, cross-attention, pooling |
| `src/maic/model.py` | History encoders, joint-action decoder, world model, imagination |
| `src/maic/batch.py` | Tensor contracts, masks, causal history construction |
| `src/maic/returns.py` | Terminal-aware, aligned discounted targets |
| `src/maic/training.py` | Supervised objectives, forecast loss, optimizer, EMA |
| `src/maic/uncertainty.py` | Ensemble prediction and variance decomposition |
| `src/maic/demo.py` | Synthetic execution check |
| `src/maic/shard_inspection.py` | Read-only sample schema, scaling, boundary, and reset audit |

The project is inspired by world-model and policy-evaluation research; it makes
no claim that extending world models to multiple agents is itself new. See the
research framing in [PROJECT.md](PROJECT.md).
