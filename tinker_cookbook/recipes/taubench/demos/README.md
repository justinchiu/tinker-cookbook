# Taubench Demo Scripts

Demo scripts for manual inspection and debugging of the taubench environment.

## Scripts

### `demo_sft_injection.py`
Shows how ask_sonnet calls are injected into SFT training data. No API calls needed.

```bash
uv run python -m tinker_cookbook.recipes.taubench.demos.demo_sft_injection
```

### `demo_ask_sonnet_rollout.py`
Runs a rollout with Qwen agent and forced ask_sonnet call. Requires Tinker API.

```bash
# DIRECT mode
uv run python -m tinker_cookbook.recipes.taubench.demos.demo_ask_sonnet_rollout mode=direct

# CONDITIONING mode
uv run python -m tinker_cookbook.recipes.taubench.demos.demo_ask_sonnet_rollout mode=conditioning
```

### `demo_rollout.py`
Basic rollout with Qwen agent (no ask_sonnet). Requires Tinker API.

```bash
uv run python -m tinker_cookbook.recipes.taubench.demos.demo_rollout
```

### `demo_rollout_claude.py`
Rollout using Claude as the agent (via litellm). Requires Anthropic API key.

```bash
uv run python -m tinker_cookbook.recipes.taubench.demos.demo_rollout_claude
```
