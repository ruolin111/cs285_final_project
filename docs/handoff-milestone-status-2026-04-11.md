# Milestone Handoff Status

## Branch

- Working branch: `codex/dynamic-prompt-routing-bootstrap`
- Repository root for implementation: `cs285_final_project/`
- `x-teaming/` remains reference-only

## Scope

This branch is now scoped to the **Phase 1 milestone only**:

- bootstrap/tooling
- canonical replayable episode schema
- synthetic sample data
- replayable routing environment
- formatter/tokenizer
- small LSTM policy/value model
- vanilla PPO
- one training curve and basic evaluation

Deferred for later:

- supervised baselines
- transformer encoder
- benign-KL regularization
- history-aware shaping
- broad ablation matrix

## Completed

### Task 1: Bootstrap

Implemented:

- `pyproject.toml`
- `configs/base.yaml`
- `scripts/train.py`
- `src/utils/{config,seeding,logging,types}.py`
- bootstrap tests in `tests/test_integration_debug.py`

Verified:

- `uv run pytest tests -q`
- `uv run python scripts/train.py --config configs/base.yaml`
- `uv run python -m scripts.train --config configs/base.yaml`

Important conventions established:

- Config loader returns a raw mapping.
- CLI-specific validation is separate in `require_bootstrap_config`.
- CLI errors must fail cleanly with `error: ...`, not raw tracebacks.

### Task 2: Canonical Replayable Schema

Implemented:

- `src/datasets/schema.py`
- `src/datasets/io.py`
- `src/datasets/preprocess.py`
- `src/datasets/__init__.py`
- `sample_data/episodes.jsonl`
- `tests/test_schema.py`

Verified:

- `uv run pytest tests/test_schema.py -v`

Important conventions established:

- Canonical turn fields are `role`, `text`, `turn_index`
- Canonical episode fields are:
  - `episode_id`
  - `source`
  - `split`
  - `label`
  - `harmful`
  - `jailbreak_success`
  - `success_turn`
  - `turns`
  - `metadata`
- Allowed turn roles are only `user`, `assistant`, `system`
- Canonical JSONL loading is strict and must reject unexpected keys
- `iter_episodes()` is intended to be the lazy streaming primitive
- `load_episodes()` is the eager wrapper

## Critical Semantic Rule

`success_turn` has one explicit meaning:

- it is the **raw conversation `turn_index` of the first harmful assistant response** in the stored trajectory

It is **not**:

- a user decision-step index
- a prefix index
- an arbitrary success marker

Required invariants:

- if `harmful` is `false`, `success_turn` must be `null`
- if `jailbreak_success` is `false`, `success_turn` must be `null`
- if `jailbreak_success` is `true`, `success_turn` must be non-null
- `success_turn` must point to an assistant turn

Task 3 and beyond should preserve this exact interpretation.

## Next Task

### Task 3: Replayable Environment

Implement next:

- `src/envs/conversation_routing.py`
- `src/envs/rollout.py`
- `tests/test_environment.py`

Environment contract:

- decisions happen on `user` turns only
- actions:
  - `0 = ALLOW`
  - `1 = WARN`
  - `2 = BLOCK`
- if blocked before the assistant response at `success_turn`, reward is positive on harmful episodes
- if the trajectory reaches the assistant response at `success_turn` without blocking, reward is negative
- benign block is negative
- benign full allow is small positive or zero

## Recommended Continuation Order

1. Finish Task 3 environment and tests
2. Implement formatter/tokenizer/LSTM model
3. Implement vanilla PPO core
4. Add minimal evaluation and save one training curve

## Commands To Resume

From the feature worktree:

```bash
uv run pytest tests/test_schema.py -v
uv run pytest tests/test_integration_debug.py -v
git status
```

If continuing Task 3:

```bash
uv run pytest tests/test_environment.py -v
```

## Notes For Teammates

- Do not re-open the schema design. Build the environment around the existing canonical fields.
- Do not reinterpret `success_turn`.
- Do not move code back under `x-teaming/`.
- Keep the milestone narrow until PPO plus one curve is working.
