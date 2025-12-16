"""
Dataset builder for supervised fine-tuning on tau2 simulation data.
"""

import chz
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import cast

import tau2.registry as reg

import tinker
from tinker_cookbook.recipes.taubench.components import AskSonnetMode, get_ask_sonnet_renderer
from tinker_cookbook.recipes.taubench.env import construct_tau2_env, ASK_SONNET_TOOL
from tinker_cookbook.renderers import TrainOnWhat, ToolCall
from tinker_cookbook.supervised.common import datum_from_tokens_weights
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset

logger = logging.getLogger(__name__)


def _inject_ask_sonnet_calls(
    messages: list[dict],
    injection_rate: float,
    rng: random.Random,
    mode: AskSonnetMode = AskSonnetMode.DIRECT_INJECTION,
) -> list[dict]:
    """Randomly inject ask_sonnet tool calls before assistant messages.

    For each assistant message (except the first), with probability `injection_rate`:

    In DIRECT_INJECTION mode:
    - Insert an ask_sonnet tool call before it
    - Convert the original assistant message to a tool result (Sonnet's response)
    - Teaches: ask_sonnet -> Sonnet responds with action -> done

    In CONDITIONING mode:
    - Insert an ask_sonnet tool call
    - Insert synthesized advice as tool result (using ConditioningRenderer format)
    - Keep original assistant message as policy's followup
    - Teaches: ask_sonnet -> Sonnet gives advice -> policy acts

    The first assistant message is never replaced because the agent should
    greet the customer directly, not delegate the greeting.
    """
    if injection_rate <= 0:
        return messages

    renderer = get_ask_sonnet_renderer(mode)

    result = []
    is_first_assistant = True
    for msg in messages:
        # Skip first assistant message - agent should greet directly
        if msg["role"] == "assistant" and is_first_assistant:
            is_first_assistant = False
            result.append(msg)
            continue

        if msg["role"] == "assistant" and rng.random() < injection_rate:
            # Insert ask_sonnet call
            ask_sonnet_msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    ToolCall(
                        function=ToolCall.FunctionBody(
                            name="ask_sonnet",
                            arguments="{}"
                        )
                    )
                ]
            }
            result.append(ask_sonnet_msg)

            if mode == AskSonnetMode.CONDITIONING:
                # CONDITIONING: Sonnet gives advice, policy takes action
                # Generate advice and format using ConditioningRenderer
                advice = _generate_advice_for_action(msg)
                tool_result_msg = renderer.format_sonnet_response_for_messages(advice)
                result.append(tool_result_msg)
                # Keep original assistant message as policy's followup
                result.append(msg)
            else:
                # DIRECT_INJECTION: Sonnet's response IS the action
                # Serialize the original content (including any tool calls)
                if msg.get("tool_calls"):
                    sonnet_response = json.dumps({
                        "content": msg.get("content", ""),
                        "tool_calls": [
                            {"name": tc.function.name, "arguments": tc.function.arguments}
                            if hasattr(tc, 'function') else tc
                            for tc in msg["tool_calls"]
                        ]
                    })
                else:
                    sonnet_response = msg.get("content", "")

                tool_result_msg = renderer.format_sonnet_response_for_messages(sonnet_response)
                result.append(tool_result_msg)
        else:
            result.append(msg)

    return result


def _generate_advice_for_action(msg: dict) -> str:
    """Generate advice by copying the full next assistant message in Qwen format.

    For conditioning mode, Sonnet's advice is the complete action the policy
    should take. This teaches the model to follow Sonnet's recommendations.
    Tool calls are wrapped in <tool_call> tags per Qwen format.
    """
    tool_calls = msg.get("tool_calls", [])
    content = msg.get("content", "")

    parts = []

    # Include text content if present
    if content:
        parts.append(content)

    # Include tool calls in Qwen <tool_call> format
    if tool_calls:
        for tc in tool_calls:
            if hasattr(tc, 'function'):
                # ToolCall object
                tool_json = json.dumps({
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                })
            elif isinstance(tc, dict):
                # Dict format
                tool_json = json.dumps({
                    "name": tc.get("name", tc.get("function", {}).get("name", "unknown")),
                    "arguments": tc.get("arguments", tc.get("function", {}).get("arguments", {}))
                })
            else:
                continue
            parts.append(f"<tool_call>\n{tool_json}\n</tool_call>")

    return "\n".join(parts) if parts else "Proceed with the customer's request."

_TASK_SPLIT_CACHE: dict[tuple[str, str], set[str]] = {}


def _get_task_ids_for_split(domain: str, split_name: str) -> set[str]:
    key = (domain, split_name)
    if key in _TASK_SPLIT_CACHE:
        return _TASK_SPLIT_CACHE[key]

    try:
        split_loader = reg.registry.get_task_splits_loader(domain)
    except KeyError:
        logger.warning("No task split loader registered for domain '%s'", domain)
        task_ids: set[str] = set()
    else:
        task_ids = set()
        if split_loader is not None:
            splits = split_loader()
            task_ids = set(splits.get(split_name, []))
        else:
            logger.warning("Domain '%s' does not expose task splits", domain)

    _TASK_SPLIT_CACHE[key] = task_ids
    return task_ids


def _split_datums_by_tasks(
    datums_with_tasks: list[tuple[tinker.Datum, str]],
    domain: str,
    use_official_test_split: bool,
    fallback_test_fraction: float | None,
) -> tuple[list[tinker.Datum], list[tinker.Datum]]:
    """Split datums into train/test, preferring official TauBench task splits."""

    if use_official_test_split:
        test_ids = _get_task_ids_for_split(domain, "test")
        if test_ids:
            test_data = [datum for datum, task_id in datums_with_tasks if task_id in test_ids]
            train_data = [datum for datum, task_id in datums_with_tasks if task_id not in test_ids]
            logger.info(
                "Domain '%s': %d datums mapped to official test split", domain, len(test_data)
            )
            return train_data, test_data
        else:
            logger.warning(
                "Domain '%s' lacks an official test split; falling back to %.0f%% random split",
                domain,
                (fallback_test_fraction or 0.0) * 100,
            )

    if fallback_test_fraction and fallback_test_fraction > 0:
        rng = random.Random(0)
        shuffled = datums_with_tasks.copy()
        rng.shuffle(shuffled)
        num_test = max(1, int(len(shuffled) * fallback_test_fraction)) if shuffled else 0
        test_data = [datum for datum, _ in shuffled[:num_test]]
        train_data = [datum for datum, _ in shuffled[num_test:]]
        logger.info(
            "Domain '%s': using fallback %.0f%% test split → %d test / %d train",
            domain,
            fallback_test_fraction * 100,
            len(test_data),
            len(train_data),
        )
        return train_data, test_data

    logger.info(
        "Domain '%s': assigning all %d datums to train (no test split configured)",
        domain,
        len(datums_with_tasks),
    )
    return [datum for datum, _ in datums_with_tasks], []


class SimpleSupervisedDataset(SupervisedDataset):
    """Simple list-based supervised dataset that avoids HF dataset overhead."""

    def __init__(self, data: list[tinker.Datum], batch_size: int):
        self.data = data
        self.batch_size = batch_size
        self.shuffled_data = data.copy()

    def get_batch(self, index: int) -> list[tinker.Datum]:
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.shuffled_data))
        return self.shuffled_data[start:end]

    def set_epoch(self, seed: int = 0):
        # Shuffle data with the given seed
        rng = random.Random(seed)
        self.shuffled_data = self.data.copy()
        rng.shuffle(self.shuffled_data)

    def __len__(self) -> int:
        return len(self.data) // self.batch_size


def _get_tau2_tools(domain: str, task_id: str | None = None) -> list[dict] | None:
    """Get tool definitions from tau2 gym environment.

    Returns None if tau2 is not available (optional dependency).

    Args:
        domain: Domain name (e.g., "telecom")
        task_id: Optional specific task ID. If None, uses first task from domain.
    """
    # Use a default task if none provided
    if task_id is None:
        tasks_loader = reg.registry.get_tasks_loader(domain)
        tasks = tasks_loader()
        if not tasks:
            logger.warning(f"No tasks found for domain '{domain}'")
            return None
        task_id = tasks[0].id

    # Create env using the same method as RL training
    # Use Qwen3 as default for consistency
    env = construct_tau2_env(domain=domain, task_id=task_id, model_name="Qwen/Qwen3-30B-A3B-Instruct-2507")

    # Tools are already extracted and converted in construct_tau2_env
    return env.tools


def _normalize_tau2_messages(messages: list[dict]) -> list[dict]:
    """Normalize tau2 messages to training format.

    Training format (what the model should learn):
        - Assistant with tool call: {"role": "assistant", "tool_calls": [{"name": "...", "arguments": {...}}], "content": ""}
        - Tool response: {"role": "tool", "content": "...", "tool_call_id": "..."}
        - User/assistant text: {"role": "...", "content": "..."}

    This strips out the "id" field from tool_calls since the model shouldn't learn to generate IDs.
    """
    normalized = []

    for msg in messages:
        # Copy the message
        new_msg = {"role": msg["role"]}

        # Handle content - always set it (empty string if None)
        if msg.get("content") is not None:
            new_msg["content"] = msg["content"]
        else:
            # Messages with tool calls or no content should have empty string
            new_msg["content"] = ""

        # Strip "id" field from tool_calls (model should only learn name + arguments)
        # Convert to pydantic ToolCall format expected by renderer
        if msg.get("tool_calls"):
            new_msg["tool_calls"] = [
                ToolCall(
                    function=ToolCall.FunctionBody(
                        name=tc["name"],
                        arguments=json.dumps(tc["arguments"]) if isinstance(tc["arguments"], dict) else tc["arguments"]
                    )
                )
                for tc in msg["tool_calls"]
            ]

        # For tool messages, ensure tool_call_id is set (rename from "id" if needed)
        if msg["role"] == "tool":
            if "tool_call_id" in msg:
                new_msg["tool_call_id"] = msg["tool_call_id"]
            elif "id" in msg:
                new_msg["tool_call_id"] = msg["id"]

        normalized.append(new_msg)

    return normalized


@chz.chz
class Tau2SimulationBuilder(ChatDatasetBuilder):
    """
    Load tau2 simulation data for supervised fine-tuning.

    File format: JSON with structure:
    {
        "simulations": [
            {
                "messages": [...],  # List of message dicts with role and content
                ...
            },
            ...
        ]
    }
    """

    simulation_file: str  # Path to simulation JSON file

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset]:
        # Load simulation file
        with open(self.simulation_file, 'r') as f:
            data = json.load(f)

        # Extract domain and get tools
        domain = data['info']['environment_info']['domain_name']
        tools = _get_tau2_tools(domain)
        if tools:
            logger.info(f"Loaded {len(tools)} tool definitions for domain '{domain}'")
        else:
            logger.warning(f"No tools available for domain '{domain}' - training without tool definitions")

        # Extract, normalize, and convert to Datum objects immediately
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        datums_with_tasks: list[tuple[tinker.Datum, str]] = []
        for sim in data['simulations']:
            messages = sim['messages']
            if messages:  # Skip empty conversations
                # Normalize messages
                normalized = _normalize_tau2_messages(messages)
                # Convert to Datum immediately
                tokens, weights = self.renderer.build_supervised_example(
                    normalized,
                    train_on_what=train_on_what,
                    tools=tools
                )
                datum = datum_from_tokens_weights(tokens, weights, self.common_config.max_length)
                datums_with_tasks.append((datum, sim['task_id']))

        logger.info(f"Loaded {len(datums_with_tasks)} conversations from {self.simulation_file}")

        train_data, test_data = _split_datums_by_tasks(
            datums_with_tasks,
            domain,
            True,
            None,
        )

        logger.info(
            "Domain '%s': Train=%d examples, Test=%d examples",
            domain,
            len(train_data),
            len(test_data),
        )

        return SimpleSupervisedDataset(
            train_data, batch_size=self.common_config.batch_size
        ), SimpleSupervisedDataset(
            test_data, batch_size=self.common_config.batch_size
        )


@chz.chz
class Tau2SimulationDirectoryBuilder(ChatDatasetBuilder):
    """
    Load tau2 simulation data from a directory of JSON files.

    All .json files in the directory will be loaded and combined.
    """

    simulation_dir: str  # Path to directory containing simulation JSON files
    pattern: str = "*.json"  # Glob pattern for files to load

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset]:
        # Find all simulation files
        sim_dir = Path(self.simulation_dir)
        sim_files = list(sim_dir.glob(self.pattern))

        logger.info(f"Found {len(sim_files)} simulation files in {sim_dir}")

        # Extract domain from first file and get tools
        # (assuming all files are from the same domain)
        domain = None
        tools = None
        if sim_files:
            with open(sim_files[0], 'r') as f:
                data = json.load(f)
            domain = data['info']['environment_info']['domain_name']
            tools = _get_tau2_tools(domain)
            if tools:
                logger.info(f"Loaded {len(tools)} tool definitions for domain '{domain}'")
            else:
                logger.warning(f"No tools available for domain '{domain}' - training without tool definitions")

        # Extract, normalize, and convert to Datum objects immediately
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        datums_with_tasks: list[tuple[tinker.Datum, str]] = []
        for sim_file in sim_files:
            with open(sim_file, 'r') as f:
                data = json.load(f)

            # Verify domain consistency
            file_domain = data['info']['environment_info']['domain_name']
            if file_domain != domain:
                logger.warning(f"File {sim_file.name} has domain '{file_domain}' but expected '{domain}'")

            for sim in data['simulations']:
                messages = sim['messages']
                if messages:  # Skip empty conversations
                    # Normalize messages
                    normalized = _normalize_tau2_messages(messages)
                    # Convert to Datum immediately
                    tokens, weights = self.renderer.build_supervised_example(
                        normalized,
                        train_on_what=train_on_what,
                        tools=tools
                    )
                    datum = datum_from_tokens_weights(tokens, weights, self.common_config.max_length)
                    datums_with_tasks.append((datum, sim['task_id']))

        logger.info(f"Loaded {len(datums_with_tasks)} conversations from {len(sim_files)} files")

        train_data, test_data = _split_datums_by_tasks(
            datums_with_tasks,
            domain or "unknown",
            True,
            None,
        )

        logger.info(
            "Domain '%s': Train=%d examples, Test=%d examples",
            domain,
            len(train_data),
            len(test_data),
        )

        return SimpleSupervisedDataset(
            train_data, batch_size=self.common_config.batch_size
        ), SimpleSupervisedDataset(
            test_data, batch_size=self.common_config.batch_size
        )

@chz.chz
class Tau2SimulationFilesBuilder(ChatDatasetBuilder):
    """
    Load tau2 simulation data from a list of specific JSON files.

    This is useful when you want to train on specific simulation files
    from potentially different domains.
    """

    simulation_files: list[str]  # List of paths to simulation JSON files
    ask_sonnet_injection_rate: float = 0.0  # Rate at which to inject ask_sonnet calls (0.0 = disabled)
    ask_sonnet_injection_seed: int = 42  # Seed for reproducible injection
    ask_sonnet_injection_mode: AskSonnetMode = AskSonnetMode.DIRECT_INJECTION

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset]:
        # Extract, normalize, and convert to Datum objects immediately
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        # RNG for ask_sonnet injection
        injection_rng = random.Random(self.ask_sonnet_injection_seed)

        datums_by_domain: dict[str, list[tuple[tinker.Datum, str]]] = defaultdict(list)
        domain_counts: dict[str, int] = defaultdict(int)
        domain_tools: dict[str, list[dict] | None] = {}
        injection_count = 0
        total_assistant_msgs = 0

        for sim_file in self.simulation_files:
            with open(sim_file, 'r') as f:
                data = json.load(f)

            # Track domain for logging
            file_domain = data['info']['environment_info']['domain_name']
            if file_domain not in domain_tools:
                domain_tools[file_domain] = _get_tau2_tools(file_domain)
                if domain_tools[file_domain]:
                    logger.info(
                        "Loaded %d tool definitions for domain '%s'",
                        len(domain_tools[file_domain]),
                        file_domain,
                    )
                else:
                    logger.warning(
                        "No tools available for domain '%s' - training without tool definitions",
                        file_domain,
                    )

                # Add ask_sonnet tool if injection is enabled
                if self.ask_sonnet_injection_rate > 0 and domain_tools[file_domain] is not None:
                    domain_tools[file_domain] = domain_tools[file_domain] + [ASK_SONNET_TOOL]
                    logger.info("Added ask_sonnet tool for domain '%s'", file_domain)

            for sim in data['simulations']:
                messages = sim['messages']
                if messages:  # Skip empty conversations
                    # Normalize messages
                    normalized = _normalize_tau2_messages(messages)

                    # Count assistant messages before injection
                    assistant_count = sum(1 for m in normalized if m["role"] == "assistant")
                    total_assistant_msgs += assistant_count

                    # Inject ask_sonnet calls if enabled
                    if self.ask_sonnet_injection_rate > 0:
                        pre_len = len(normalized)
                        normalized = _inject_ask_sonnet_calls(
                            normalized,
                            self.ask_sonnet_injection_rate,
                            injection_rng,
                            mode=self.ask_sonnet_injection_mode,
                        )
                        # Count injections
                        # direct_injection: adds 2 messages (ask_sonnet + tool result), removes 1 = net +1
                        # conditioning: adds 3 messages (ask_sonnet + tool result + followup), removes 1 = net +2
                        if self.ask_sonnet_injection_mode == AskSonnetMode.CONDITIONING:
                            injection_count += (len(normalized) - pre_len) // 2
                        else:
                            injection_count += len(normalized) - pre_len

                    # Convert to Datum immediately
                    tokens, weights = self.renderer.build_supervised_example(
                        normalized,
                        train_on_what=train_on_what,
                        tools=domain_tools[file_domain]
                    )
                    datum = datum_from_tokens_weights(tokens, weights, self.common_config.max_length)
                    datums_by_domain[file_domain].append((datum, sim['task_id']))
                    domain_counts[file_domain] += 1

        total_datums = sum(domain_counts.values())
        logger.info(f"Loaded {total_datums} conversations from {len(self.simulation_files)} files")
        logger.info(f"Domain distribution: {dict(domain_counts)}")

        if self.ask_sonnet_injection_rate > 0:
            logger.info(
                "ask_sonnet injection: %d/%d assistant turns replaced with ask_sonnet->tool_result (%.1f%% target rate)",
                injection_count, total_assistant_msgs,
                self.ask_sonnet_injection_rate * 100
            )

        train_data: list[tinker.Datum] = []
        test_data: list[tinker.Datum] = []
        for domain_name, datums_with_tasks in datums_by_domain.items():
            domain_train, domain_test = _split_datums_by_tasks(
                datums_with_tasks,
                domain_name,
                True,
                None,
            )
            train_data.extend(domain_train)
            test_data.extend(domain_test)

        logger.info(
            "Combined domains: Train=%d examples, Test=%d examples",
            len(train_data),
            len(test_data),
        )

        return SimpleSupervisedDataset(
            train_data, batch_size=self.common_config.batch_size
        ), SimpleSupervisedDataset(
            test_data, batch_size=self.common_config.batch_size
        )
