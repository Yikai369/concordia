"""Simple act component for audience that returns stored evaluation response."""

from concordia.components.agent import impression_management_pe as impe
from concordia.components.agent import action_spec_ignored
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


class SimpleAudienceActComponent(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging
):
  """Simple act component that returns the audience's stored evaluation response."""

  def __init__(
      self,
      memory_component_key: str = impe.DEFAULT_IMPE_MEMORY_COMPONENT_KEY,
      pre_act_label: str = 'Audience Response',
  ):
    """Initialize simple audience act component."""
    super().__init__(pre_act_label)
    self._memory_component_key = memory_component_key

  def _make_pre_act_value(self) -> str:
    """Make pre-act value (returns empty for acting component)."""
    return ''

  def get_action_attempt(
      self,
      context: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    """Return the most recent evaluation response."""
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=impe.IMPEMemoryComponent
    )

    # Get most recent evaluation
    evaluations = memory.get_recent_evaluations()
    if not evaluations:
      # No evaluation yet, return empty (shouldn't happen in normal flow)
      return ''

    # Get the most recent evaluation's utterance
    latest_eval = evaluations[-1]
    utt = latest_eval.utterance

    # Return formatted response
    return f'DIALOGUE: {utt.text}\nBODY: {utt.body}'

  def get_state(self) -> entity_component.ComponentState:
    """Get component state."""
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Set component state."""
    pass
