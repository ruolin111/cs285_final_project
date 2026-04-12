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

### Task 3: Replayable Environment

Implemented:

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

Verified:

- `uv run pytest tests/test_environment.py -v`

Important conventions established:

- the environment only exposes decisions on `user` turns
- terminal harmful misses return the stored assistant success turn as the final observation
- `WARN` updates `num_warnings` in observation state without injecting synthetic turns into the raw conversation history

### Task 4: Formatter, Tokenizer, and LSTM Router

Implemented:

- `src/models/formatting.py`
- `src/models/tokenizer.py`
- `src/models/encoders.py`
- `src/models/router.py`
- `src/models/__init__.py`
- `tests/test_tokenizer_and_formatter.py`
- `tests/test_models.py`

Verified:

- `uv run pytest tests/test_tokenizer_and_formatter.py tests/test_models.py -q`

Important conventions established:

- formatted observations include a `[ROUTING]` header with `decision_turn_index` and `num_warnings`
- truncation limits must apply to the final formatted observation, including routing metadata
- tokenizer padding must fail loudly if `pad_to` is smaller than the longest sequence

### Task 5: Minimal Vanilla PPO Baseline

Implemented:

- `src/algorithms/ppo.py`
- `src/algorithms/__init__.py`
- `scripts/train_ppo.py`
- `configs/ppo_debug.yaml`
- `tests/test_ppo.py`

Verified:

- `uv run pytest tests/test_ppo.py -q`
- `uv run pytest -q`
- `uv run python scripts/train_ppo.py --config configs/ppo_debug.yaml`

Important conventions established:

- rollout collection uses shuffled episode coverage rather than always starting from dataset prefix episodes
- `eval_every > 1` is supported; final update is always evaluated so summary and best checkpoint exist
- training writes `metrics.jsonl`, `summary.json`, checkpoints, tokenizer artifact, config snapshot, and `plots/training_curves.png`
- runtime outputs live under `outputs/` and are gitignored

## Current Milestone State

Working milestone path is now:

1. load canonical replayable episodes
2. interact through `ConversationRoutingEnv`
3. format observations into deterministic text
4. tokenize with a small vocab tokenizer
5. train an LSTM policy/value model with in-house PPO
6. save metrics and one training-curve plot

Current branch commits for this path:

- `2eaf938` bootstrap
- `d4eed95` schema/data layer
- `8d906be` replayable environment
- `31f9603` formatter/tokenizer/LSTM router
- `c6bd433` milestone PPO baseline

## Recommended Continuation Order

1. Add a compact evaluator script for loading a saved PPO checkpoint and exporting episode-level metrics
2. Add one short README run section for `train.py` and `train_ppo.py`
3. Decide whether the milestone needs one committed sample curve image in `docs/` or whether generated `outputs/` artifacts are enough
4. Only after that, start supervised baselines or broader final-project scaffolding

## Commands To Resume

From the feature worktree:

```bash
uv run pytest -q
uv run python scripts/train.py --config configs/base.yaml
uv run python scripts/train_ppo.py --config configs/ppo_debug.yaml
git status
```

If continuing from the PPO milestone:

```bash
ls outputs/
cat outputs/<latest-run>/summary.json
```

## Notes For Teammates

- Do not re-open the schema design. Build the environment around the existing canonical fields.
- Do not reinterpret `success_turn`.
- Do not move code back under `x-teaming/`.
- Keep the milestone narrow until checkpoint loading/eval is clean.
- `uv.lock` is still intentionally untracked.
