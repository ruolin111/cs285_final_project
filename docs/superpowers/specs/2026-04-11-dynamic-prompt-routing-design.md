# Dynamic Prompt Routing Design

## Goal

Build a CS285 final project codebase in PyTorch for turn-level defense against multi-turn LLM jailbreaks. The defender observes conversation history and emits one discrete routing action per decision step: `ALLOW`, `WARN`, or `BLOCK`.

The project must remain faithful to a replayable offline-RL framing:
- no token-level RL
- no online attacker generation in the main loop
- no external RL library for PPO core logic
- small, explainable models suitable for multiple ablations and report-ready analysis

## Scope Boundaries

- The repository root will contain the new CS285 project.
- [`x-teaming/`](/Users/yuchenzhang/Desktop/berkeleyMeng/cs285/final_proj/cs285_final_project/x-teaming) remains reference-only and is not the project package root.
- Decision points occur on `user` turns only.
- The environment uses stored future outcomes as counterfactual continuation labels.
- The initial default model stack is deterministic formatter + custom vocabulary tokenizer + lightweight LSTM.
- Transformer, benign-KL regularization, and other full-method components are scaffolded from the start but do not block the first milestone implementation.
- The project uses `uv` with `pyproject.toml` as the primary environment/tooling path.

## Recommended Approach

Use a full final-architecture scaffold from day one, but implement the lowest-risk vertical slice first.

This keeps the repo stable while the project grows from:
1. data schema and preprocessing
2. replayable environment
3. formatting/tokenization and small encoder
4. supervised baselines
5. vanilla PPO
6. evaluation pipeline
7. history-aware shaping and benign-KL ablations

This approach is preferred over a narrow milestone-only scaffold because it makes baseline-to-method comparisons cleaner and reduces refactoring churn later.

## Repository Structure

```text
cs285_final_project/
├── configs/
│   ├── base.yaml
│   ├── supervised_memoryless.yaml
│   ├── supervised_history.yaml
│   ├── ppo.yaml
│   ├── ppo_shaping.yaml
│   ├── ppo_benign_kl.yaml
│   └── full_method.yaml
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   └── superpowers/
│       └── specs/
├── outputs/
├── sample_data/
│   ├── harmful_immediate_success.jsonl
│   ├── harmful_late_success.jsonl
│   └── benign_episode.jsonl
├── scripts/
│   ├── preprocess_data.py
│   ├── train.py
│   ├── train_supervised.py
│   ├── train_ppo.py
│   └── eval_model.py
├── src/
│   ├── algorithms/
│   ├── datasets/
│   ├── envs/
│   ├── eval/
│   ├── models/
│   ├── trainers/
│   └── utils/
├── tests/
├── x-teaming/
├── pyproject.toml
└── README.md
```

## Top-Level Responsibilities

- `configs/`: experiment definitions for baselines, PPO, ablations, and the full method
- `data/raw/`: untracked source-specific data drops from X-Teaming/XGuard or similar exporters
- `data/processed/`: normalized canonical JSONL episodes produced by preprocessing
- `sample_data/`: tiny committed synthetic episodes for tests and debug runs
- `scripts/`: thin CLI entry points for preprocessing, training, and evaluation
- `src/datasets/`: canonical schema, validation, loaders, split handling, preprocessing
- `src/envs/`: replayable conversation routing environment and rollout helpers
- `src/models/`: formatter, tokenizer, LSTM/transformer encoders, policy/value and supervised heads
- `src/algorithms/`: PPO core, GAE, rollout buffers, credit shaping, benign-KL logic
- `src/trainers/`: supervised and RL orchestration
- `src/eval/`: episode-level evaluation, exports, metrics, plots
- `src/utils/`: config loading, seeding, shared types, logging helpers
- `tests/`: unit and integration checks targeting schema, environment, PPO, and evaluation correctness
- `outputs/`: run artifacts, plots, config snapshots, metrics logs, checkpoints

## Data Design

### Canonical Episode Schema

Each replayable conversation episode will normalize to a strict JSONL schema with fields equivalent to:

- `episode_id: str`
- `source: str`
- `split: str`
- `label: str`
- `harmful: bool`
- `jailbreak_success: bool`
- `success_turn: int | null`
- `turns: list[Turn]`
- `metadata: dict[str, Any]`

`success_turn` is the raw conversation `turn_index` of the first harmful assistant response in the stored trajectory. It is not a user decision-step index. For benign episodes or harmful episodes without jailbreak success, `success_turn` must be `null`.

Each `Turn` will include:

- `role: "user" | "assistant" | "system"`
- `text: str`
- `turn_index: int`

Expected conversation metadata includes:

- `category`
- `attack_family`
- `verifier_score`
- `target_model`
- `dataset_name`

### Data Storage Convention

- `sample_data/` is committed and contains tiny synthetic episodes only.
- `data/raw/` is gitignored and stores raw source exports.
- `data/processed/` is gitignored by default and stores normalized JSONL outputs.
- Conversion logic lives in source-controlled scripts/modules rather than notebooks.

### Validation Requirements

- fail loudly on malformed fields
- no silent field dropping
- clear error messages for missing labels, invalid roles, inconsistent `turn_index`, or impossible `success_turn`
- source-specific preprocessing should map into the canonical schema without hardcoding training logic to one dataset format

## Environment Design

### Environment Name

`ConversationRoutingEnv`

### Core Semantics

- One episode equals one stored conversation.
- Observations are structured conversation prefixes plus routing metadata.
- The harmful label is never directly exposed as observation.
- Each decision step occurs on a `user` turn before the defender would permit or deny the next assistant response.
- Actions are:
  - `0 = ALLOW`
  - `1 = WARN`
  - `2 = BLOCK`

### Transition Logic

- `ALLOW`: advance to the next decision point
- `WARN`: record warning count and advance
- `BLOCK`: terminate the episode immediately
- episode also terminates when the stored conversation ends
- optional config may allow early terminal handling at `success_turn`

### Reward Design

Rewards are configurable and mostly terminal.

For harmful episodes:
- block before the assistant response at `success_turn`: positive reward, larger when earlier
- allow the conversation to reach the assistant response at `success_turn` without blocking: large negative reward
- harmful episode with no recorded success: neutral or small positive depending on config

For benign episodes:
- fully allow to end: small positive or zero
- warn: small negative
- block: large negative

The environment returns an `info` dict containing:

- `episode_id`
- `harmful`
- `jailbreak_success`
- `blocked`
- `block_turn`
- `success_turn`
- `num_warnings`
- `final_outcome`

The first version uses stored continuation labels only and does not simulate adaptive attackers.

## Model Design

### Formatting Layer

Conversation history is rendered deterministically with role markers such as:

- `[SYSTEM]`
- `[USER]`
- `[ASSISTANT]`

The formatter supports truncation strategies:
- `last_k_turns`
- `max_chars`
- `max_tokens_estimate`

### Tokenization Layer

Use a small custom vocabulary tokenizer first:
- special tokens for padding and role markers
- vocabulary built from training data
- save/load support for reproducibility

### Encoder Layer

Initial default:
- lightweight LSTM encoder

Scaffolded alternative:
- small transformer encoder

Both encoder paths feed:
- policy logits over 3 actions
- value estimate for PPO
or
- class logits for supervised routing baselines

## Learning Methods

### Supervised Baselines

Two baselines are required:

1. memoryless baseline
Uses only the latest user/current-turn text.

2. history-based supervised baseline
Uses the full revealed prefix through the formatter + encoder stack.

Since oracle routing actions do not exist in raw data, supervised training uses simple configurable pseudo-label rules. These labels are clearly documented as heuristic baselines only.

### Vanilla PPO Baseline

The PPO implementation is written from scratch and includes:

- rollout collection
- GAE
- clipped policy objective
- value loss
- entropy bonus
- minibatch updates
- gradient clipping
- checkpoint save/load

The PPO code must support variable-length episodes and discrete actions of size 3.

### Main Method

The project-specific method extends PPO with:

1. history-aware credit shaping
Redistribute sparse terminal signal backward or scale reward for earlier correct intervention.

2. benign-support regularization
Add KL regularization toward a frozen reference policy on benign-like states to reduce overblocking.

Both extensions are independently toggleable for fair ablations:
- `ppo`
- `ppo_plus_shaping`
- `ppo_plus_kl`
- `full_method`

## Evaluation Design

Evaluation is episode-level and independent from training loops where possible.

Metrics include:

- attack success rate after defense
- benign false positive rate
- average intervention turn
- average return
- block precision/recall
- warn rate
- average number of warnings
- harmful episodes blocked before `success_turn`
- benign episodes fully allowed
- stratification by conversation length
- stratification by attack horizon

Exports include:

- metrics JSON/CSV
- per-episode predictions
- console summary
- plots in `outputs/` using matplotlib only

## Testing Strategy

Unit tests should cover:

- schema validation
- tokenizer/formatter correctness
- environment transitions
- reward computation
- PPO buffer behavior
- GAE correctness
- model forward pass shapes

Integration tests should cover:

- supervised training on tiny synthetic data
- PPO running for a few updates on synthetic episodes
- evaluation on a saved checkpoint

Debugging utilities should include:

- printing a rollout with actions and rewards
- episode inspection for missed/blocked cases
- deterministic mini-run mode
- overfit-on-tiny-data mode

## Implementation Order

1. Bootstrap `uv` + `pyproject.toml`, base config, shared utils, and a minimal train CLI.
2. Implement canonical schema, loaders, validation, synthetic sample data, and preprocessing.
3. Implement `ConversationRoutingEnv` and rollout helpers with tests.
4. Implement formatter, tokenizer, LSTM path, and transformer scaffold.
5. Implement supervised baselines and prefix-expansion datasets.
6. Implement vanilla PPO.
7. Implement evaluation pipeline and plotting.
8. Implement shaping and benign-KL ablations.
9. Harden with integration tests and debug utilities.

## Risks To Watch

- accidental leakage of `harmful` or future success information into the observation path
- off-by-one mistakes in `success_turn` and block timing
- converting the task into token-level or generation-time RL by mistake
- reward shaping that makes comparison against vanilla PPO unfair
- benign-KL applied to states that were not selected by a defensible rule
- evaluation that reports per-step classification quality but misses episode-level defense behavior

## Out of Scope

- live attacker generation in the training loop
- large pretrained LLM fine-tuning
- distributed training infrastructure
- cloud-only execution paths
- replacing turn-level routing with token-level control
