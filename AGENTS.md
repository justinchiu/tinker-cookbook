# Tinker Cookbook Agent Guide

Quick reference for agents working on `tinker-cookbook`. Full documentation is in `docs/`.

`tinker-cookbook` is a client library with training and eval code built on the Tinker service (hosted by Thinking Machines Lab) and the Tinker SDK (a separate repo with just the API). You author training/eval loops that run on a CPU machine; Tinker executes the heavy GPU work.

**Start here:** `docs/training-sampling.mdx` - Complete walkthrough of training and sampling basics.

## Documentation Map (`docs/`)

**API Fundamentals:**
- `index.mdx` - Tinker overview, division of responsibilities
- `install.mdx` - Installation, API key setup
- `training-sampling.mdx` - **Starter guide**: data prep, forward_backward, sampling, vision inputs
- `losses.mdx` - Loss functions (cross_entropy, importance_sampling, ppo, cispo, dro, forward_backward_custom)
- `save-load.mdx` - Checkpointing (save_weights_for_sampler vs save_state)
- `async.mdx` - Sync/async APIs, futures, overlapping requests
- `model-lineup.mdx` - Available models
- `under-the-hood.mdx` - Clock cycles, worker pools

**API Reference (`api-reference/`):**
- `types.md` - **All API types** (Datum, ModelInput, TensorData, SamplingParams, etc.)
- `trainingclient.md`, `samplingclient.md`, `serviceclient.md`, `restclient.md` - Client APIs

**Supervised Learning (`supervised-learning/`):**
- `../supervised-learning.mdx` - SL overview
- `sl-basic.mdx` - First SL run
- `sl-hyperparams.mdx` - LR formula, batch size
- `sl-loop.mdx` - Minimal training loop
- `prompt-distillation.mdx` - Distilling prompts
- `sweep-case-study.mdx` - Hyperparameter sweeps

**Reinforcement Learning (`rl/`):**
- `../rl.mdx` - RL overview (RLVR, RLHF)
- `rl-basic.mdx` - First RL run
- `rl-envs.mdx` - Custom Env, EnvGroupBuilder, RLDataset
- `rl-loops.mdx` - Minimal RL loop
- `rl-hyperparams.mdx` - batch_size vs group_size, async training
- `sequence-extension.mdx` - Multi-turn RL, KV-cache

**Preferences (`preferences/`):**
- `../preferences.mdx` - DPO vs RLHF overview
- `dpo-guide.mdx` - DPO training
- `rlhf-example.mdx` - RLHF pipeline

**Other:**
- `rendering.mdx` - Renderers (bridge between chat-style data and token sequences), vision inputs, TrainOnWhat
- `completers.mdx` - TokenCompleter vs MessageCompleter
- `evals.mdx` - Inline evals, Inspect AI, custom evaluators
- `lora-primer.mdx` - LoRA background
- `download-weights.mdx` / `publish-weights.mdx` - Weight export

---

## Composing Types

Agents often struggle with the nested type hierarchy. Key resources:

**Reference:** `docs/api-reference/types.md` documents all API types.

**Core types:**
- `Datum` = `model_input` (ModelInput) + `loss_fn_inputs` (dict of TensorData)
- `ModelInput` = list of chunks (EncodedTextChunk, ImageChunk)
- `TensorData` = wrapper for numpy/torch arrays with shape info

**Helper functions** (use these instead of manual construction):
- `datum_from_model_input_weights(model_input, weights, max_length)` - SL datum creation (`supervised/common.py`)
- `conversation_to_datum(messages, renderer, max_length, train_on_what)` - Full pipeline (`supervised/data.py`)
- `renderer.build_supervised_example(messages)` - Returns (ModelInput, weights)
- `ModelInput.from_ints(tokens)` - Create from token list
- `TensorData.from_numpy(arr)` / `TensorData.from_torch(tensor)` - Wrap arrays

---

## Architecture

**Builder pattern:** Config objects are `chz` dataclasses (SupervisedDatasetBuilder, RLDatasetBuilder, EnvGroupBuilder). They expose `.build()`/`__call__()` returning runtime objects.

**Key code locations:**
- SL: `tinker_cookbook/supervised/train.py`
- RL: `tinker_cookbook/rl/train.py`
- DPO: `tinker_cookbook/preference/train_dpo.py`
- Renderers: `tinker_cookbook/renderers/`
- Completers: `tinker_cookbook/completers.py`
- RL types: `tinker_cookbook/rl/types.py`
- Recipes: `tinker_cookbook/recipes/`

---

## Conventions

**Subscript suffixes** for tensor names: `_P` (problems), `_G` (groups), `_T` (tokens), `_D` (datums). Example: `tokens_P_G_T[p][g][t]`

**Code style:**
- Explicit typing; avoid `Any` / `type: ignore`
- Use `safezip`, `timed`, `scope` helpers
- `@chz.chz` decorator for config serialization
- `ml_log.log_metrics` for metrics; `logtree` for transcripts
- **MINIMIZE USE OF TRY/EXCEPT.** Prefer failing loudly. Only catch exceptions at true system boundaries (e.g., network retries). Never swallow errors silently.

**Env lifecycle:** `Env` objects are single-use (no reset). Create via `EnvGroupBuilder`.

---

## Common Pitfalls

1. **LoRA LR:** Use `hyperparam_utils.get_lr(model_name)` - LoRA needs ~10x higher LR than full fine-tuning.

2. **Renderer mismatch:** Match `renderer_name` to model family (`llama3`, `qwen3`, `role_colon`).

3. **Async gaps:** Submit `forward_backward_async` and `optim_step_async` back-to-back before awaiting.

4. **Sampler desync:** Create a **new** sampling client after saving weights.

5. **Type construction:** Use helper functions, not manual dict construction. See `supervised/data.py` and `supervised/common.py`.

6. **Group semantics:** RL advantages are centered within each group.

7. **DPO:** Start with `dpo_beta=0.1`, LR~1e-5.

8. **Tool injection:** Both RL and SFT must use `renderer.create_conversation_prefix_with_tools(tool_specs, system_prompt)` to inject tool definitions. Do NOT pass `tools=` to `build_generation_prompt` or `build_supervised_example` — those params are accepted but silently ignored.

9. **Debugging evals:** Always save full trajectories including the system prompt (which contains tool definitions). If tools are missing from the system prompt, the model will hallucinate tool names and get "Tool not found" errors.

---

## Wandb Projects

Use these wandb projects (entity: `percepta-ai`):

- **`tau2-rl`** — All RL training runs (with or without injection/exploration)
- **`tau2-sft`** — All SFT training runs

Do not use `tau2-migration-debug` for new runs (legacy, contains early debugging runs).

---

## Testing

```bash
# Unit tests (no API needed)
uv run pytest tinker_cookbook/tests/test_renderers.py
uv run pytest tinker_cookbook/tests/test_utils.py

# Taubench tests
uv run pytest tinker_cookbook/recipes/taubench/tests/ -v

# Smoke tests (requires API key + network)
uv run pytest tinker_cookbook/tests/smoke_tests.py
```

For debugging, shrink workloads via `n_batches`, `batch_size`, `group_size` in dataset builders.

---

## Pre-commit

Pre-commit hooks are configured in `.pre-commit-config.yaml` (ruff lint + format, trailing whitespace, large file check).

```bash
# Install hooks (once per clone/worktree)
uv run pre-commit install

# Run manually on all files
uv run pre-commit run --all-files

# Run on staged files only (this is what the git hook does)
uv run pre-commit run
```

Hooks run automatically on `git commit`. If a hook fails, it will fix the file in-place — re-stage and commit again.

---

## Type Checking

**Types must pass.** We are not strict about full type coverage yet, but all code must pass both type checkers without errors:

```bash
# Pyright (primary type checker)
uvx pyright tinker_cookbook/recipes/taubench/

# ty (Ruff's type checker, secondary)
uvx ty check tinker_cookbook/recipes/taubench/
```

Run both before committing. Fix any errors they report — do not use `type: ignore` to suppress them unless there is a genuine false positive (document why in a comment).

---

## Development Workflow

**TDD is required.** When porting or writing new features:
1. Write tests FIRST. Run them. Watch them FAIL (RED).
2. Write/port the implementation. Run tests. Watch them PASS (GREEN).
3. Never write tests and code simultaneously — that skips the proof that tests catch regressions.
4. For critical code paths, do mutation testing: break the code, verify the test fails.

**GitHub PRs:** This repo is a fork of `thinking-machines-lab/tinker-cookbook`.
- `origin` = `justinchiu/tinker-cookbook` (the fork)
- `upstream` = `thinking-machines-lab/tinker-cookbook`
- PRs for taubench work go to the **fork**, not upstream: `gh pr create --repo justinchiu/tinker-cookbook --base main`
- Without `--repo`, `gh` defaults to upstream due to the fork relationship. Always specify `--repo`.

**PR review cycle:** After pushing, always check for and address PR review comments (from Codex or humans):
1. Commit and push your changes
2. Check PR comments: `gh api repos/justinchiu/tinker-cookbook/pulls/<PR#>/comments`
3. For each review comment: write a regression test (RED), fix the bug (GREEN), commit
4. Push the fixes
5. Reply to each comment on the PR: `gh api repos/justinchiu/tinker-cookbook/pulls/<PR#>/comments/<ID>/replies -f body="Fixed in <commit>"`
6. Repeat until all comments are addressed

---

## Taubench Migration

The taubench recipe is being migrated from the old `taubench` branch (76 commits diverged from upstream) to the `taubench-new` branch (based on current upstream `main`). Porting is done feature-by-feature with TDD.

**Worktree layout:**
- `/home/ubuntu/code/tinker-cookbook` — old `taubench` branch (reference, do not modify)
- `/home/ubuntu/code/tinker-cookbook-merge-upstream` — `taubench-new` branch (active development)

**Source of truth for fork code:** read from `/home/ubuntu/code/tinker-cookbook/tinker_cookbook/recipes/taubench/` when porting.
