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
    # Logging
    "RolloutLogger",
]
