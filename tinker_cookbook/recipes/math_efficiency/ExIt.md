# Expert Iteration (ExIt) Plan — Improved

## Goal
Build an expert‑iteration loop with multiple search strategies (experts) and a single off‑policy update that correctly handles likelihood ratios when sampling and training contexts differ.

## Core idea
Every trajectory encodes:
```
(sampling_context + completion)  ->  (training_context + completion)
```
We **reuse the sampled completion tokens verbatim** and **swap only the context** for training. The loss computes the likelihood ratio automatically from:
- `log q_s(y | x_sample)` (stored sampling logprobs)
- `log p_train(y | x_train)` (computed in forward pass)

## Search strategies (each = one EnvGroupBuilder)
1) **IID samples (GRPO/PG)**
   - sampling_context = training_context
   - on‑policy case, ratio = 1

2) **Prompt augmentation**
   - sampling_context includes extra instructions
   - training_context removes them
   - teaches the behavior without scaffolding

3) **2‑turn feedback**
   - sampling_context: problem -> attempt1 -> feedback -> attempt2
   - training_context options:
     - problem + attempt1 (drop feedback)
     - problem only (drop feedback + attempt1)
   - completion = attempt2 tokens

## Stratified mixture of experts
We treat each search strategy as a **stratum** and average their gradient contributions:
```
 g = (1 / N) * Σ_s Σ_{i in s} [ (p_train / q_s) * A_i * ∇ log p_train ]
```
- `s` = strategy ID
- `q_s` = behavior policy under sampling context of s
- `p_train` = model under training context
- `A_i` = advantage (centered within group)
- `N` = total samples across all strategies

Optional: scale each sample by `w_s / n_s` if you want explicit mixture weights.

## Implementation plan (code locations)

### 1) Add strategy metadata and training‑context transform
- **File:** `tinker_cookbook/rl/types.py`
- Add either:
  - `strategy_id: str | None` on `Trajectory` or `TrajectoryGroup`
  - `training_context_transform: Callable[[Observation, int], Observation] | None`

Recommended: store on `TrajectoryGroup` so all trajectories in the group share the same strategy.

### 2) Capture strategy ID during rollouts
- **File:** `tinker_cookbook/rl/rollouts.py`
- When building a `TrajectoryGroup`, attach `strategy_id` (from the `EnvGroupBuilder`).

### 3) Provide training vs sampling contexts in envs
- **File:** `tinker_cookbook/rl/problem_env.py` (or recipe‑specific env in `recipes/`)
- Add fields like:
  - `sampling_convo_prefix`
  - `training_convo_prefix`
- Add method `get_training_observation()` or a transform that rebuilds the training context.

### 4) Swap context when building Datums
- **File:** `tinker_cookbook/rl/data_processing.py`
- Update `trajectory_to_data(...)` to accept a context transform:
  - Extract completion tokens from transitions
  - Replace observation/context with training observation
  - Keep sampled action tokens + `sampling_logprobs` untouched

This is the **critical step** for correct likelihood ratios.

### 5) Stratified averaging / optional weighting
- **Files:**
  - `tinker_cookbook/rl/data_processing.py`
  - `tinker_cookbook/rl/train.py`
- During minibatch assembly, compute per‑strategy counts `n_s` and optionally scale advantages by `w_s / n_s`.
- Otherwise just average across all samples (stratified estimator).

### 6) Metrics and debugging
- **File:** `tinker_cookbook/rl/metric_util.py`
- Include `strategy_id` in `logging_tags()` for per‑strategy metrics.
- Optional: log mean logprob ratio (`log p_train − log q_s`) per strategy.

## Invariants / gotchas
- **Completion tokens are reused verbatim.** Never re‑tokenize from text.
- **Sampling logprobs are never recomputed.** They define `q_s`.
- **Renderer must match model family** in both sampling and training contexts.
- **Multi‑turn:** decide explicitly what to keep in training context (attempt1, feedback). Treat it as part of the strategy.

## Minimal implementation checklist
- [ ] Add `strategy_id` + context transform in `types.py`.
- [ ] Thread strategy ID from `EnvGroupBuilder` through `rollouts.py` to `TrajectoryGroup`.
- [ ] Provide training prompt transform in envs.
- [ ] Update `trajectory_to_data` to swap context.
- [ ] Add stratified weighting in minibatch assembly (optional).
- [ ] Add per‑strategy metrics tags.

## TODO (defer): Multi‑turn sampling support
- [ ] Add a 2‑turn env (attempt1 -> feedback -> attempt2) with explicit feedback generation.
- [ ] Add per‑transition weighting so only attempt2 contributes to loss (attempt1 weight = 0).
- [ ] Add context‑swap logic for the final turn only (drop feedback; optionally drop attempt1).
- [ ] Ensure datum splitting handles non‑prefix contexts cleanly when feedback is removed.

## Detailed walkthrough (implementation steps)

### Step 1: Strategy metadata + context transform plumbing
**Files:**  
- `tinker_cookbook/rl/types.py`  
- `tinker_cookbook/rl/rollouts.py`

**What to do:**
1) Add metadata on `TrajectoryGroup` (preferred) or `Trajectory`:
   - `strategy_id: str | None`
   - `context_transform: Callable[[Observation, int], Observation] | None`  
   The `int` argument is the transition index (turn).
2) Add the same fields to the concrete `EnvGroupBuilder` you use in recipes (e.g., math_efficiency builders), so each group knows which strategy it represents.
3) In `do_group_rollout(...)`, set these fields on the returned `TrajectoryGroup` so they flow into training:
   - `TrajectoryGroup(..., strategy_id=builder.strategy_id, context_transform=builder.context_transform)`

**Outcome:** every group carries the strategy ID and the correct training‑context transform.

---

### Step 2: Provide training vs sampling context at the env level
**File:**  
- `tinker_cookbook/rl/problem_env.py` or a recipe‑specific env

**What to do:**
1) Add fields:
   - `sampling_convo_prefix: list[Message]`
   - `training_convo_prefix: list[Message]`
2) Add a method to generate training context (e.g. `get_training_observation()` or `build_training_observation(question)`).
3) The env’s `initial_observation()` should continue to return the **sampling** context (this is what the sampler sees).
4) For prompt augmentation, the training method should **omit** the augmentation while preserving the same renderer.

**Outcome:** we can rebuild a training prompt later without re‑tokenizing completions.

---

### Step 3: Swap context during data processing
**File:**  
- `tinker_cookbook/rl/data_processing.py`

**What to do:**
1) Add a `context_transform` parameter to `trajectory_to_data(...)` and thread it from `assemble_training_data(...)`.
2) Apply it per transition to swap `transition.ob` to a training observation:
   - `ob = context_transform(transition.ob, i_turn)` when present.
3) Keep `transition.ac.tokens` and `transition.ac.logprobs` **unchanged**.
4) Ensure `sampling_logprobs` are aligned to the same action tokens (no re‑tokenization).

**Outcome:** `model_input` uses `x_train`, while `logprobs` remain `log q_s(y|x_sample)`.

---

### Step 4: Stratified averaging / optional weighting
**Files:**  
- `tinker_cookbook/rl/data_processing.py`  
- `tinker_cookbook/rl/train.py`

**What to do:**
1) During minibatch assembly, compute counts `n_s` by `strategy_id` (use `metadata_D` or per‑group stats).
2) If using explicit mixture weights `w_s`, scale each trajectory advantage by `w_s / n_s` before expanding to tokens.
3) Otherwise, just average across all samples (stratified estimator).
4) Log `n_s` per batch to confirm you’re sampling as expected.

**Outcome:** unbiased gradient estimate across multiple strategies.

---

### Step 5: Metrics + debugging
**File:**  
- `tinker_cookbook/rl/metric_util.py`

**What to do:**
1) Add `strategy_id` to `logging_tags()` so metrics are grouped per strategy.
2) (Optional) log mean `log p_train − log q_s` by strategy for sanity checks.
3) Add a small sanity check metric: fraction of masked tokens per strategy (should be consistent).

**Outcome:** visibility into which strategies help and whether ratios look sane.

---

### Step 6: Recipe wiring (math_efficiency)
**Files:**  
- `tinker_cookbook/recipes/math_efficiency/efficient_env.py`  
- `tinker_cookbook/recipes/math_efficiency/train_rl.py`

**What to do:**
1) Create one `EnvGroupBuilder` per strategy (IID, prompt‑aug, etc.).  
2) Assign a `strategy_id` and appropriate `context_transform` for each builder.  
3) Keep group sizes consistent so advantage centering remains stable.
4) Interleave strategies in `get_batch(...)` so each batch contains a mix (or make it a config toggle).

**Outcome:** ExIt runs multiple search modes per batch and trains on the unified stratified estimator.

---

## Testing

### Unit tests (no API)
Add a new test file: `tinker_cookbook/tests/test_rl_data_processing.py`

**Test 1: context swap correctness**
- Build a tiny `Trajectory` with:
  - `ob` = `ModelInput.from_ints([1,2,3])`
  - `ac.tokens` = `[4,5]`, `ac.logprobs` = `[-0.1,-0.2]`
- Provide `context_transform` that replaces observation with `ModelInput.from_ints([9,9])`.
- Assert:
  - `datum.model_input` starts with `[9,9]` (not `[1,2,3]`)
  - `loss_fn_inputs["logprobs"]` still matches the sampled logprobs for action tokens
  - mask covers only action tokens

**Test 2: stratified weighting**
- Create two fake `TrajectoryGroup`s with different `strategy_id`s.
- If weighting is enabled, assert advantages are scaled by `w_s / n_s`.

### Integration-ish (no API)
- Build a dummy `EnvGroupBuilder` + stub `TokenCompleter`.
- Run `assemble_training_data(...)` and validate datum count + token alignment.

### End-to-end (requires API)
- Run smoke tests: `pytest tinker_cookbook/tests/smoke_tests.py`
- Or run the recipe with tiny settings (1–2 problems, small group size) and inspect logs.

---

## Concrete snippets to keep (from original)

### PromptAugmentedEnv (copy‑pasteable)
```python
class PromptAugmentedEnv(ProblemEnv):
    def __init__(
        self,
        renderer,
        sampling_convo_prefix: list[Message],
        training_convo_prefix: list[Message],
        ...
    ):
        super().__init__(renderer, convo_prefix=None)
        self.sampling_convo_prefix = sampling_convo_prefix
        self.training_convo_prefix = training_convo_prefix

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        convo = self.sampling_convo_prefix + [{"role": "user", "content": self.get_question()}]
        return self.renderer.build_generation_prompt(convo), self.stop_condition

    def get_training_observation(self) -> Observation:
        convo = self.training_convo_prefix + [{"role": "user", "content": self.get_question()}]
        return self.renderer.build_generation_prompt(convo)
```

### feedback_transform example (multi‑turn)
```python
def feedback_transform(ob: Observation, turn_idx: int) -> Observation:
    if turn_idx == 1:  # second turn
        # Rebuild without feedback; keep problem + attempt1
        return rebuild_without_feedback(ob)
    return ob
```
