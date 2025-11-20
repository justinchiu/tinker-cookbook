# Training on tau2 Benchmark Customer Service Tasks

```bash
uv run python -m tinker_cookbook.recipes.taubench.train \
    model_name=Qwen/Qwen3-30B-A3B-Instruct-2507 \
    batch_size=32 \
    group_size=8 \
    domain=telecom \
    task_set=small
```

### Background: tau2 Benchmark

tau2 is a benchmark for training and evaluating customer service agents. It provides realistic multi-turn conversations where agents must help users resolve technical issues. The benchmark includes domains like telecom, airline, and retail, with user simulators powered by GPT-4 that generate contextually appropriate customer requests and responses.

### Implementation

The `Tau2Env` class in [env.py](./env.py) wraps tau2's `AgentGymEnv` to provide the RL interface:

* **Initial observation**: The customer's problem description
* **Actions**: Agent responses attempting to diagnose and fix the issue
* **Rewards**: Based on successful task completion (e.g., resolving the mobile data issue)
* **Episode termination**: When the task is completed or the conversation reaches a natural end

The tau2 orchestrator manages the conversation flow between the agent (policy being trained) and the user simulator, creating realistic customer service interactions for RL training.

### Domains and Task Sets

* **Domains**: `telecom`, `airline`, `retail`, `mock`
* **Task sets**: `small` (20 tasks), `default` (50 tasks), `full` (all tasks)

Each task represents a specific customer issue with evaluation criteria for successful resolution.