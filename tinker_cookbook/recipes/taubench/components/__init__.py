"""Components for Tau2Env - modular pieces for cleaner architecture."""

from tinker_cookbook.recipes.taubench.components.types import (
    AskSonnetMode,
    ActionType,
    ObservationType,
    ParsedAction,
    Tau2StepResult,
    ExternalLLMConfig,
)
from tinker_cookbook.recipes.taubench.components.action_parser import ActionParser
from tinker_cookbook.recipes.taubench.components.message_manager import MessageManager
from tinker_cookbook.recipes.taubench.components.external_llm import ExternalLLMClient, LLMCallResult
from tinker_cookbook.recipes.taubench.components.tau2_gym_wrapper import Tau2GymWrapper
from tinker_cookbook.recipes.taubench.components.ask_sonnet_renderers import (
    AskSonnetRenderer,
    DirectInjectionRenderer,
    ConditioningRenderer,
    get_ask_sonnet_renderer,
)
from tinker_cookbook.recipes.taubench.components.rollout_logger import RolloutLogger
from tinker_cookbook.recipes.taubench.components.epsilon_policy import (
    EpsilonAskSonnetPolicy,
    EpsilonAskSonnetMetrics,
    linear_decay,
    exponential_decay,
)


# ask_sonnet tool definition (OpenAI function format)
ASK_SONNET_TOOL = {
    "type": "function",
    "function": {
        "name": "ask_sonnet",
        "description": "Delegate this turn to Claude Sonnet. Sonnet will see the full conversation and respond on your behalf.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

# Instruction to append to system prompt when ask_sonnet is available
ASK_SONNET_INSTRUCTION = """

IMPORTANT: You have access to a special tool called `ask_sonnet` that delegates the current turn to a more capable AI assistant (Claude Sonnet). Use this tool when:
- You are unsure how to proceed with a complex request
- You need help understanding the customer's needs
- You want to verify your approach before taking an action
- The task requires careful reasoning or nuanced judgment

When you call `ask_sonnet`, Claude Sonnet will see the full conversation and respond on your behalf. Use this tool liberally when uncertain - it's better to ask for help than to make mistakes.

NOTE: Always greet the customer yourself on the first turn. Do not use `ask_sonnet` for the initial greeting - handle it directly, then use `ask_sonnet` for subsequent turns if needed."""


__all__ = [
    # Types
    "AskSonnetMode",
    "ActionType",
    "ObservationType",
    "ParsedAction",
    "Tau2StepResult",
    "ExternalLLMConfig",
    "LLMCallResult",
    # Components
    "ActionParser",
    "MessageManager",
    "ExternalLLMClient",
    "Tau2GymWrapper",
    # Ask Sonnet Renderers
    "AskSonnetRenderer",
    "DirectInjectionRenderer",
    "ConditioningRenderer",
    "get_ask_sonnet_renderer",
    # Epsilon Policy
    "EpsilonAskSonnetPolicy",
    "EpsilonAskSonnetMetrics",
    "linear_decay",
    "exponential_decay",
    # Logging
    "RolloutLogger",
    # Ask Sonnet Constants
    "ASK_SONNET_TOOL",
    "ASK_SONNET_INSTRUCTION",
]
