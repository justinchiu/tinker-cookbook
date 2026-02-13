"""
Dataset builder for supervised fine-tuning on tau2 simulation data.
"""

import chz
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import tau2.registry as reg

import tinker
from tinker_cookbook.recipes.taubench.components import (
    ASK_SONNET_INSTRUCTION,
    ASK_SONNET_TOOL,
    AskSonnetMode,
    get_ask_sonnet_renderer,
)
from tinker_cookbook.recipes.taubench.env import construct_tau2_env
from tinker_cookbook import renderers
from tinker_cookbook.renderers import TrainOnWhat, ToolCall
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset

logger = logging.getLogger(__name__)


def _inject_ask_sonnet_calls(
    messages: list[dict],
    injection_rate: float,
    rng: random.Random,
    mode: AskSonnetMode = AskSonnetMode.DIRECT_INJECTION,
) -> list[dict]:
    """Randomly inject ask_sonnet tool calls before assistant messages.

    First samples a binary mask for all eligible assistant turns (excluding the first),
    then applies the injection. This ensures clean sampling without weird consecutive patterns.

    In DIRECT_INJECTION mode:
    - Insert an ask_sonnet tool call before the assistant message
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

    # Find indices of eligible assistant messages (all except first)
    eligible_indices = []
    is_first_assistant = True
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant":
            if is_first_assistant:
                is_first_assistant = False
            else:
                eligible_indices.append(i)

    if not eligible_indices:
        return messages

    # Sample binary mask for eligible turns
    inject_mask = [rng.random() < injection_rate for _ in eligible_indices]
    inject_set = {idx for idx, do_inject in zip(eligible_indices, inject_mask) if do_inject}

    # Build result with injections
    renderer = get_ask_sonnet_renderer(mode)
    result = []

    for i, msg in enumerate(messages):
        if i in inject_set:
            # Insert ask_sonnet call
            ask_sonnet_msg = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    ToolCall(function=ToolCall.FunctionBody(name="ask_sonnet", arguments="{}"))
                ],
            }
            result.append(ask_sonnet_msg)

            # Both CONDITIONING and DIRECT_INJECTION use the same SFT format:
            # ask_sonnet call + advice (tool result) + original assistant message
            # The difference between modes is only at inference time.
            advice = _generate_advice_for_action(msg)
            tool_result_msg = renderer.format_sonnet_response_for_messages(advice)
            result.append(tool_result_msg)
            # Keep original assistant message as policy's followup
            result.append(msg)
        else:
            result.append(msg)

    return result


def _mark_trainable_fields(messages: list[dict]) -> list[dict]:
    """Mark trainable field on all messages for CUSTOMIZED train_on_what.

    Only ask_sonnet tool calls are marked as trainable=True.
    Everything else is marked as trainable=False.
    """
    result = []
    for msg in messages:
        # Check if this is an ask_sonnet tool call
        is_ask_sonnet_call = False
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if hasattr(tc, "function") and tc.function.name == "ask_sonnet":
                    is_ask_sonnet_call = True
                    break
                elif isinstance(tc, dict):
                    name = tc.get("name") or tc.get("function", {}).get("name")
                    if name == "ask_sonnet":
                        is_ask_sonnet_call = True
                        break

        result.append({**msg, "trainable": is_ask_sonnet_call})
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
            if hasattr(tc, "function"):
                # ToolCall object
                tool_json = json.dumps(
                    {
                        "name": tc.function.name,
                        "arguments": (
                            json.loads(tc.function.arguments)
                            if isinstance(tc.function.arguments, str)
                            else tc.function.arguments
                        ),
                    }
                )
            elif isinstance(tc, dict):
                # Dict format
                tool_json = json.dumps(
                    {
                        "name": tc.get("name", tc.get("function", {}).get("name", "unknown")),
                        "arguments": tc.get(
                            "arguments",
                            tc.get("function", {}).get("arguments", {}),
                        ),
                    }
                )
            else:
                continue
            parts.append(f"<tool_call>\n{tool_json}\n</tool_call>")

    return "\n".join(parts) if parts else "Proceed with the customer's request."


def _get_tau2_system_prompt(domain: str) -> str:
    """Get the tau2 system prompt for a domain.

    This builds the same system prompt used at inference time in env.py.
    """
    from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT

    # Get domain policy
    try:
        from tau2.gym.gym_agent import AgentGymEnv

        # Create a temporary env to get the policy
        tasks_loader = reg.registry.get_tasks_loader(domain)
        tasks = tasks_loader()
        if tasks:
            task_id = tasks[0].id
            env = AgentGymEnv(domain=domain, task_id=task_id)
            env.reset()  # Must reset to initialize orchestrator
            domain_policy = env._get_policy()
        else:
            domain_policy = ""
    except Exception as e:
        logger.warning(f"Could not get domain policy for {domain}: {e}")
        domain_policy = ""

    return SYSTEM_PROMPT.format(
        domain_policy=domain_policy,
        agent_instruction=AGENT_INSTRUCTION,
    )


# Cache for tau2 system prompts (one per domain)
_TAU2_SYSTEM_PROMPT_CACHE: dict[str, str] = {}


def _get_cached_tau2_system_prompt(domain: str) -> str:
    """Get cached tau2 system prompt for a domain."""
    if domain not in _TAU2_SYSTEM_PROMPT_CACHE:
        _TAU2_SYSTEM_PROMPT_CACHE[domain] = _get_tau2_system_prompt(domain)
    return _TAU2_SYSTEM_PROMPT_CACHE[domain]


def _add_system_prompt_with_ask_sonnet(messages: list[dict], domain: str) -> list[dict]:
    """Add tau2 system prompt with ASK_SONNET_INSTRUCTION.

    Prepends a system message with:
    1. The tau2 domain-specific system prompt
    2. The ASK_SONNET_INSTRUCTION

    If there's already a system message, replaces it.
    """
    if not messages:
        return messages

    # Build full system prompt
    tau2_prompt = _get_cached_tau2_system_prompt(domain)
    full_system_content = tau2_prompt + ASK_SONNET_INSTRUCTION

    system_msg: dict = {
        "role": "system",
        "content": full_system_content,
    }

    # Check if first message is system - replace it
    if messages[0].get("role") == "system":
        return [system_msg] + messages[1:]
    else:
        return [system_msg] + messages


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
                "Domain '%s': %d datums mapped to official test split",
                domain,
                len(test_data),
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


@dataclass
class ConversationRecord:
    """Raw conversation data before injection/rendering."""

    messages: list[dict]
    task_id: str
    domain: str


class DynamicInjectionDataset(SupervisedDataset):
    """Dataset that re-injects ask_sonnet calls with different randomization per epoch.

    Unlike SimpleSupervisedDataset which stores pre-computed datums, this dataset
    stores raw messages and re-injects + re-renders on each set_epoch() call.
    This ensures different ask_sonnet injection patterns across epochs.
    """

    def __init__(
        self,
        conversations: list[ConversationRecord],
        batch_size: int,
        renderer: "renderers.Renderer",
        train_on_what: TrainOnWhat,
        domain_tools: dict[str, list[dict] | None],
        injection_rate: float,
        injection_mode: AskSonnetMode,
        max_length: int | None,
        train_on_ask_sonnet_only: bool = False,
    ):
        self.conversations = conversations
        self.batch_size = batch_size
        self.renderer = renderer
        self.train_on_what = train_on_what
        self.domain_tools = domain_tools
        self.injection_rate = injection_rate
        self.injection_mode = injection_mode
        self.max_length = max_length
        self.train_on_ask_sonnet_only = train_on_ask_sonnet_only

        # Will be populated by set_epoch
        self.datums: list[tinker.Datum] = []
        # Initialize with epoch 0
        self.set_epoch(0)

    def _render_conversations(self, rng: random.Random) -> list[tinker.Datum]:
        """Inject ask_sonnet and render all conversations with given RNG."""
        datums = []
        for conv in self.conversations:
            # Copy messages to avoid mutating original
            messages = [m.copy() for m in conv.messages]

            # Inject ask_sonnet calls and add system prompt with instruction
            if self.injection_rate > 0:
                messages = _inject_ask_sonnet_calls(
                    messages,
                    self.injection_rate,
                    rng,
                    mode=self.injection_mode,
                )
                # Add tau2 system prompt + ask_sonnet instruction
                messages = _add_system_prompt_with_ask_sonnet(messages, conv.domain)

            # Mark trainable fields for CUSTOMIZED train_on_what
            # This happens AFTER all message transformations so it covers everything
            if self.train_on_what == TrainOnWhat.CUSTOMIZED:
                messages = _mark_trainable_fields(messages)

            # Render to tokens
            tools = self.domain_tools.get(conv.domain)
            model_input, weights = self.renderer.build_supervised_example(
                messages,
                train_on_what=self.train_on_what,
                tools=tools,
            )
            datum = datum_from_model_input_weights(model_input, weights, self.max_length)
            datums.append(datum)

        return datums

    def get_batch(self, index: int) -> list[tinker.Datum]:
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.datums))
        return self.datums[start:end]

    def set_epoch(self, seed: int = 0):
        """Re-inject and re-render with new randomization, then shuffle."""
        # Use seed for both injection randomization and shuffle
        injection_rng = random.Random(seed)
        shuffle_rng = random.Random(seed + 1000000)  # Different seed for shuffle

        # Re-render all conversations with new injection pattern
        self.datums = self._render_conversations(injection_rng)

        # Shuffle
        shuffle_rng.shuffle(self.datums)

        logger.info(
            "Epoch %d: re-injected ask_sonnet calls and shuffled %d datums",
            seed,
            len(self.datums),
        )

    def __len__(self) -> int:
        return len(self.datums) // self.batch_size


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
    env = construct_tau2_env(
        domain=domain,
        task_id=task_id,
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
    )

    # Tools are already extracted and converted in construct_tau2_env
    return env.tools


def _normalize_tau2_messages(messages: list[dict]) -> list[dict]:
    """Normalize tau2 messages to training format.

    Training format (what the model should learn):
        - Assistant with tool call: {"role": "assistant", "tool_calls": [ToolCall(...)], "content": ""}
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
                        arguments=(
                            json.dumps(tc["arguments"])
                            if isinstance(tc["arguments"], dict)
                            else tc["arguments"]
                        ),
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
        with open(self.simulation_file) as f:
            data = json.load(f)

        # Extract domain and get tools
        domain = data["info"]["environment_info"]["domain_name"]
        tools = _get_tau2_tools(domain)
        if tools:
            logger.info(f"Loaded {len(tools)} tool definitions for domain '{domain}'")
        else:
            logger.warning(
                f"No tools available for domain '{domain}' - training without tool definitions"
            )

        # Extract, normalize, and convert to Datum objects immediately
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        datums_with_tasks: list[tuple[tinker.Datum, str]] = []
        for sim in data["simulations"]:
            messages = sim["messages"]
            if messages:  # Skip empty conversations
                # Normalize messages
                normalized = _normalize_tau2_messages(messages)
                # Convert to Datum immediately
                model_input, weights = self.renderer.build_supervised_example(
                    normalized, train_on_what=train_on_what, tools=tools
                )
                datum = datum_from_model_input_weights(
                    model_input, weights, self.common_config.max_length
                )
                datums_with_tasks.append((datum, sim["task_id"]))

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
        ), SimpleSupervisedDataset(test_data, batch_size=self.common_config.batch_size)


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
            with open(sim_files[0]) as f:
                data = json.load(f)
            domain = data["info"]["environment_info"]["domain_name"]
            tools = _get_tau2_tools(domain)
            if tools:
                logger.info(f"Loaded {len(tools)} tool definitions for domain '{domain}'")
            else:
                logger.warning(
                    f"No tools available for domain '{domain}' - training without tool definitions"
                )

        # Extract, normalize, and convert to Datum objects immediately
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        datums_with_tasks: list[tuple[tinker.Datum, str]] = []
        for sim_file in sim_files:
            with open(sim_file) as f:
                data = json.load(f)

            # Verify domain consistency
            file_domain = data["info"]["environment_info"]["domain_name"]
            if file_domain != domain:
                logger.warning(
                    f"File {sim_file.name} has domain '{file_domain}' but expected '{domain}'"
                )

            for sim in data["simulations"]:
                messages = sim["messages"]
                if messages:  # Skip empty conversations
                    # Normalize messages
                    normalized = _normalize_tau2_messages(messages)
                    # Convert to Datum immediately
                    model_input, weights = self.renderer.build_supervised_example(
                        normalized, train_on_what=train_on_what, tools=tools
                    )
                    datum = datum_from_model_input_weights(
                        model_input, weights, self.common_config.max_length
                    )
                    datums_with_tasks.append((datum, sim["task_id"]))

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
        ), SimpleSupervisedDataset(test_data, batch_size=self.common_config.batch_size)


@chz.chz
class Tau2SimulationFilesBuilder(ChatDatasetBuilder):
    """
    Load tau2 simulation data from a list of specific JSON files.

    This is useful when you want to train on specific simulation files
    from potentially different domains.

    When ask_sonnet_injection_rate > 0, uses DynamicInjectionDataset which
    re-randomizes injection patterns on each epoch.
    """

    simulation_files: list[str]  # List of paths to simulation JSON files
    ask_sonnet_injection_rate: float = (
        0.0  # Rate at which to inject ask_sonnet calls (0.0 = disabled)
    )
    ask_sonnet_injection_seed: int = (
        42  # Seed for reproducible injection (unused with dynamic injection)
    )
    ask_sonnet_injection_mode: AskSonnetMode = AskSonnetMode.DIRECT_INJECTION
    train_on_ask_sonnet_only: bool = False  # Only train on ask_sonnet tool call turns

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset]:
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.ALL_ASSISTANT_MESSAGES
        )

        # Load and normalize all conversations, split by domain
        conversations_by_domain: dict[str, list[tuple[ConversationRecord, str]]] = defaultdict(list)
        domain_tools: dict[str, list[dict] | None] = {}
        domain_counts: dict[str, int] = defaultdict(int)

        for sim_file in self.simulation_files:
            with open(sim_file) as f:
                data = json.load(f)

            file_domain = data["info"]["environment_info"]["domain_name"]

            # Load tools for this domain (once per domain)
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

            for sim in data["simulations"]:
                messages = sim["messages"]
                if messages:
                    normalized = _normalize_tau2_messages(messages)
                    conv = ConversationRecord(
                        messages=normalized,
                        task_id=sim["task_id"],
                        domain=file_domain,
                    )
                    conversations_by_domain[file_domain].append((conv, sim["task_id"]))
                    domain_counts[file_domain] += 1

        total_convs = sum(domain_counts.values())
        logger.info(f"Loaded {total_convs} conversations from {len(self.simulation_files)} files")
        logger.info(f"Domain distribution: {dict(domain_counts)}")

        # Split into train/test by task IDs
        train_convs: list[ConversationRecord] = []
        test_convs: list[ConversationRecord] = []

        for domain_name, convs_with_tasks in conversations_by_domain.items():
            test_ids = _get_task_ids_for_split(domain_name, "test")
            if test_ids:
                domain_train = [
                    conv for conv, task_id in convs_with_tasks if task_id not in test_ids
                ]
                domain_test = [conv for conv, task_id in convs_with_tasks if task_id in test_ids]
                logger.info(
                    "Domain '%s': %d train, %d test (official split)",
                    domain_name,
                    len(domain_train),
                    len(domain_test),
                )
            else:
                domain_train = [conv for conv, _ in convs_with_tasks]
                domain_test = []
                logger.info(
                    "Domain '%s': %d train, no test split",
                    domain_name,
                    len(domain_train),
                )
            train_convs.extend(domain_train)
            test_convs.extend(domain_test)

        logger.info(
            "Combined domains: Train=%d conversations, Test=%d conversations",
            len(train_convs),
            len(test_convs),
        )

        # Build datasets
        if self.ask_sonnet_injection_rate > 0:
            # Use dynamic injection dataset for training (re-randomizes each epoch)
            logger.info(
                "Using DynamicInjectionDataset with %.1f%% injection rate (randomized per epoch)",
                self.ask_sonnet_injection_rate * 100,
            )
            train_dataset = DynamicInjectionDataset(
                conversations=train_convs,
                batch_size=self.common_config.batch_size,
                renderer=self.renderer,
                train_on_what=train_on_what,
                domain_tools=domain_tools,
                injection_rate=self.ask_sonnet_injection_rate,
                injection_mode=self.ask_sonnet_injection_mode,
                max_length=self.common_config.max_length,
                train_on_ask_sonnet_only=self.train_on_ask_sonnet_only,
            )
        else:
            # No injection - use simple pre-rendered dataset
            train_datums = []
            for conv in train_convs:
                messages = conv.messages
                # Mark trainable fields if needed for CUSTOMIZED train_on_what
                if train_on_what == TrainOnWhat.CUSTOMIZED:
                    messages = _mark_trainable_fields(messages)
                tools = domain_tools.get(conv.domain)
                model_input, weights = self.renderer.build_supervised_example(
                    messages,
                    train_on_what=train_on_what,
                    tools=tools,
                )
                datum = datum_from_model_input_weights(
                    model_input, weights, self.common_config.max_length
                )
                train_datums.append(datum)
            train_dataset = SimpleSupervisedDataset(
                train_datums, batch_size=self.common_config.batch_size
            )

        # Test dataset uses same loss mask as training
        # If ask_sonnet injection is enabled, inject with a fixed seed for reproducibility
        test_rng = random.Random(999)  # Fixed seed for test set
        test_datums = []
        for conv in test_convs:
            messages = [m.copy() for m in conv.messages]

            # Apply same injection as training (with fixed seed)
            if self.ask_sonnet_injection_rate > 0:
                messages = _inject_ask_sonnet_calls(
                    messages,
                    self.ask_sonnet_injection_rate,
                    test_rng,
                    mode=self.ask_sonnet_injection_mode,
                )
                messages = _add_system_prompt_with_ask_sonnet(messages, conv.domain)

            # Mark trainable fields if using CUSTOMIZED
            if train_on_what == TrainOnWhat.CUSTOMIZED:
                messages = _mark_trainable_fields(messages)

            tools = domain_tools.get(conv.domain)
            model_input, weights = self.renderer.build_supervised_example(
                messages,
                train_on_what=train_on_what,
                tools=tools,
            )
            datum = datum_from_model_input_weights(
                model_input, weights, self.common_config.max_length
            )
            test_datums.append(datum)
        test_dataset = SimpleSupervisedDataset(
            test_datums, batch_size=self.common_config.batch_size
        )

        return train_dataset, test_dataset
