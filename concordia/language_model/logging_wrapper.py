"""Language model wrapper for logging all interactions."""

import threading
import warnings
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from concordia.language_model import language_model
from concordia.utils import information_flow_history
from typing_extensions import override

# Thread-local storage for component context
_context = threading.local()


def set_component_context(component_name: str | None, phase: str | None) -> None:
    """Set current component context for logging.

    Args:
        component_name: Name of the component making the call.
        phase: Simulation phase ('pre_act', 'act', 'post_act', 'observe').
    """
    _context.component_name = component_name
    _context.phase = phase


def clear_component_context() -> None:
    """Clear component context."""
    _context.component_name = None
    _context.phase = None


def get_component_context() -> tuple[str | None, str | None]:
    """Get current component context.

    Returns:
        Tuple of (component_name, phase) or (None, None) if not set.
    """
    return (
        getattr(_context, 'component_name', None),
        getattr(_context, 'phase', None),
    )


class LoggingLanguageModel(language_model.LanguageModel):
    """Wraps a language model to log all inputs and outputs."""

    def __init__(
        self,
        model: language_model.LanguageModel,
        history_bank: information_flow_history.InformationFlowHistoryBank,
        agent_name: str,
    ):
        """Initialize logging wrapper.

        Args:
            model: The underlying language model to wrap.
            history_bank: History bank to store interactions.
            agent_name: Name of the agent using this model.
        """
        self._model = model
        self._history_bank = history_bank
        self._agent_name = agent_name

    @override
    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
        terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
        temperature: float = language_model.DEFAULT_TEMPERATURE,
        top_p: float = language_model.DEFAULT_TOP_P,
        top_k: int = language_model.DEFAULT_TOP_K,
        timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
        seed: int | None = None,
    ) -> str:
        """Sample text from the model and log the interaction."""
        # Get context from thread-local storage
        component_name, phase = get_component_context()

        # Prepare kwargs for logging
        kwargs = {
            'max_tokens': max_tokens,
            'terminators': list(terminators),
            'temperature': temperature,
            'top_p': top_p,
            'top_k': top_k,
            'timeout': timeout,
            'seed': seed,
        }

        # Call model
        try:
            response = self._model.sample_text(
                prompt=prompt,
                max_tokens=max_tokens,
                terminators=terminators,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                timeout=timeout,
                seed=seed,
            )
        except Exception as e:
            # Log error interaction
            try:
                self._history_bank.add_interaction(
                    agent_name=self._agent_name,
                    prompt=prompt,
                    response=f"<ERROR: {str(e)}>",
                    method='sample_text',
                    kwargs=kwargs,
                    component_name=component_name,
                    phase=phase,
                    metadata={
                        'error': str(e),
                        'error_type': type(e).__name__,
                    },
                )
            except Exception as log_error:
                warnings.warn(f"Failed to log error interaction: {log_error}")
            raise

        # Log successful interaction
        try:
            self._history_bank.add_interaction(
                agent_name=self._agent_name,
                prompt=prompt,
                response=response,
                method='sample_text',
                kwargs=kwargs,
                component_name=component_name,
                phase=phase,
            )
        except Exception as e:
            # Don't fail the simulation if logging fails
            warnings.warn(f"Failed to log interaction: {e}")

        return response

    @override
    def sample_choice(
        self,
        prompt: str,
        responses: Sequence[str],
        *,
        seed: int | None = None,
    ) -> tuple[int, str, Mapping[str, Any]]:
        """Sample a choice from the model and log the interaction."""
        # Get context from thread-local storage
        component_name, phase = get_component_context()

        # Prepare kwargs for logging
        kwargs = {
            'seed': seed,
            'responses': list(responses),
        }

        # Call model
        try:
            index, response, info = self._model.sample_choice(
                prompt=prompt,
                responses=responses,
                seed=seed,
            )
        except Exception as e:
            # Log error interaction
            try:
                self._history_bank.add_interaction(
                    agent_name=self._agent_name,
                    prompt=prompt,
                    response=f"<ERROR: {str(e)}>",
                    method='sample_choice',
                    kwargs=kwargs,
                    component_name=component_name,
                    phase=phase,
                    metadata={
                        'error': str(e),
                        'error_type': type(e).__name__,
                    },
                )
            except Exception as log_error:
                warnings.warn(f"Failed to log error interaction: {log_error}")
            raise

        # Log successful interaction
        try:
            self._history_bank.add_interaction(
                agent_name=self._agent_name,
                prompt=prompt,
                response=response,
                method='sample_choice',
                kwargs=kwargs,
                component_name=component_name,
                phase=phase,
                metadata={
                    'index': index,
                    'info': dict(info) if info else {},
                },
            )
        except Exception as e:
            # Don't fail the simulation if logging fails
            warnings.warn(f"Failed to log interaction: {e}")

        return index, response, info
