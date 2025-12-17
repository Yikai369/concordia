# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Components for PE (Prediction Error) driven conversation."""

from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
import re
from typing import Any

from concordia.components.agent import action_spec_ignored
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


@dataclass
class Goal:
  """Goal definition for PE conversation."""
  name: str
  description: str
  ideal: float = 1.0
  role: str | None = None  # Interview context role (e.g., "Product Manager")


@dataclass
class Utterance:
  """A conversation utterance."""
  turn: int
  speaker: str
  text: str
  body: str = ""  # Body language description


@dataclass
class PERecord:
  """A prediction error record."""
  turn: int
  partner_text: str
  estimate: float
  pe: float


@dataclass
class ReflectionRecord:
  """A reflection record."""
  turn: int
  text: str


DEFAULT_PE_MEMORY_COMPONENT_KEY = 'PE_Memory'
DEFAULT_PE_ESTIMATION_COMPONENT_KEY = 'PE_Estimation'
DEFAULT_PE_REFLECTION_COMPONENT_KEY = 'PE_Reflection'


class PEMemoryComponent(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging
):
  """Component to store PE conversation memory."""

  def __init__(
      self,
      goal: Goal,
      recent_k: int = 3,
      pre_act_label: str = 'PE Memory',
  ):
    """Initialize PE memory component.

    Args:
      goal: The goal for this agent.
      recent_k: Number of recent items to retrieve.
      pre_act_label: Label for pre_act output.
    """
    super().__init__(pre_act_label)
    self._goal = goal
    self._recent_k = recent_k
    self._conversation: list[Utterance] = []
    self._pe_history: list[PERecord] = []
    self._reflections: list[ReflectionRecord] = []

  def add_utterance(self, turn: int, speaker: str, text: str) -> None:
    """Add a conversation utterance."""
    self._conversation.append(Utterance(turn=turn, speaker=speaker, text=text))

  def add_pe_record(
      self, turn: int, partner_text: str, estimate: float, pe: float
  ) -> None:
    """Add a PE record."""
    self._pe_history.append(
        PERecord(
            turn=turn, partner_text=partner_text, estimate=estimate, pe=pe
        )
    )

  def add_reflection(self, turn: int, text: str) -> None:
    """Add a reflection."""
    self._reflections.append(ReflectionRecord(turn=turn, text=text))

  def get_recent_conversation(self, k: int | None = None) -> list[Utterance]:
    """Get recent conversation entries."""
    if k is None:
      k = self._recent_k
    return self._conversation[-k:]

  def get_recent_pe_history(self, k: int | None = None) -> list[PERecord]:
    """Get recent PE history."""
    if k is None:
      k = self._recent_k
    return self._pe_history[-k:]

  def get_recent_reflections(
      self, k: int | None = None
  ) -> list[ReflectionRecord]:
    """Get recent reflections."""
    if k is None:
      k = self._recent_k
    return self._reflections[-k:]

  def get_goal(self) -> Goal:
    """Get the goal."""
    return self._goal

  def get_last_pe(self) -> float:
    """Get the last PE value."""
    if self._pe_history:
      return self._pe_history[-1].pe
    return 0.0

  def _make_pre_act_value(self) -> str:
    """Format memory for pre_act context."""
    conv_k = self.get_recent_conversation()
    pe_k = self.get_recent_pe_history()
    refl_k = self.get_recent_reflections()

    lines = [
        f'Goal: {self._goal.name}',
        f'Goal description: {self._goal.description}',
        f'Ideal value: {self._goal.ideal:.2f}',
        '',
        f'Recent conversation (last {self._recent_k}):',
    ]
    if conv_k:
      for u in conv_k:
        lines.append(f'  [t={u.turn} {u.speaker}] {u.text}')
    else:
      lines.append('  (none)')

    lines.append('')
    lines.append('Recent PE history:')
    if pe_k:
      for p in pe_k:
        lines.append(
            f'  (turn {p.turn}) estimate={p.estimate:.2f}, PE={p.pe:+.2f} '
            f'← partner: "{p.partner_text}"'
        )
    else:
      lines.append('  (none)')

    lines.append('')
    lines.append('Recent reflections:')
    if refl_k:
      for r in refl_k:
        lines.append(f'  (turn {r.turn}) {r.text}')
    else:
      lines.append('  (none)')

    result = '\n'.join(lines)
    self._logging_channel({
        'Key': self.get_pre_act_label(),
        'Value': result
    })
    return result

  def get_state(self) -> entity_component.ComponentState:
    """Get component state for checkpointing."""
    return {
        'conversation': [asdict(u) for u in self._conversation],
        'pe_history': [asdict(p) for p in self._pe_history],
        'reflections': [asdict(r) for r in self._reflections],
        'goal': asdict(self._goal),
        'recent_k': self._recent_k,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Set component state from checkpoint."""
    self._conversation = [
        Utterance(**u) for u in state.get('conversation', [])
    ]
    self._pe_history = [PERecord(**p) for p in state.get('pe_history', [])]
    self._reflections = [
        ReflectionRecord(**r) for r in state.get('reflections', [])
    ]
    goal_dict = state.get('goal', asdict(self._goal))
    self._goal = Goal(**goal_dict)
    self._recent_k = state.get('recent_k', self._recent_k)


class PEEstimationComponent(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging
):
  """Component to estimate state and compute PE during observe phase."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory_component_key: str = DEFAULT_PE_MEMORY_COMPONENT_KEY,
      pre_act_label: str = 'PE Estimation',
  ):
    """Initialize PE estimation component.

    Args:
      model: Language model for estimation.
      memory_component_key: Key of PEMemoryComponent.
      pre_act_label: Label for pre_act output.
    """
    super().__init__(pre_act_label)
    self._model = model
    self._memory_component_key = memory_component_key
    self._last_estimate: float | None = None
    self._last_pe: float | None = None
    self._last_partner_text: str = ''

  def pre_observe(self, observation: str) -> str:
    """Extract partner's text from observation."""
    # Try to extract partner's utterance from observation
    # The observation may contain the partner's action in various formats
    self._last_partner_text = observation.strip()

    # Try to extract quoted text (most reliable)
    import re

    # Look for patterns like: "text" or 'text' or [Agent Name] text
    quoted_match = re.search(r'["\']([^"\']+)["\']', observation)
    if quoted_match:
      self._last_partner_text = quoted_match.group(1)
    else:
      # Look for agent name pattern: [Agent Name] or Agent Name:
      agent_pattern = r'(?:Agent [AB]|Partner)[:\s]+(.+)'
      agent_match = re.search(agent_pattern, observation, re.I)
      if agent_match:
        self._last_partner_text = agent_match.group(1).strip()
      else:
        # Remove common prefixes
        for prefix in ['Partner said:', '[observation]', 'Observation:', 'said:']:
          if self._last_partner_text.startswith(prefix):
            self._last_partner_text = self._last_partner_text[len(prefix):].strip()
        # If still contains the entity's own name, try to extract after it
        entity_name = self.get_entity().name
        if entity_name in self._last_partner_text:
          # Extract text after the other agent's name
          other_agents = ['Agent A', 'Agent B']
          other_agents.remove(entity_name)
          if other_agents:
            other_name = other_agents[0]
            idx = self._last_partner_text.find(other_name)
            if idx >= 0:
              self._last_partner_text = self._last_partner_text[idx + len(other_name):].strip()
              # Remove common separators
              for sep in [':', 'said', '-', '—']:
                if self._last_partner_text.startswith(sep):
                  self._last_partner_text = self._last_partner_text[len(sep):].strip()

    return ''

  def _make_pre_act_value(self) -> str:
    """Make pre-act value (returns empty for estimation component)."""
    return ''

  def post_observe(self) -> str:
    """Estimate state and compute PE."""
    if not self._last_partner_text:
      return ''

    memory = self.get_entity().get_component(
        self._memory_component_key, type_=PEMemoryComponent
    )
    goal = memory.get_goal()

    # Get current turn (approximate from conversation length)
    current_turn = len(memory.get_recent_conversation()) + 1

    # Prompt LLM for estimation
    prompt = f"""You are {self.get_entity().name}. Goal: {goal.name}.
Goal description: {goal.description}
Ideal value on goal dimension: {goal.ideal:.2f}

Task: From only the partner's last response, estimate the CURRENT STATE on the goal dimension
as a single number in [0,1], where 1 means perfectly achieving the goal.
Partner said: "{self._last_partner_text}"

Respond with a single number in [0,1]. You may include a brief comment after the number.
"""

    raw = self._model.sample_text(prompt)
    # Parse first float in [0,1]
    m = re.search(r'([01](?:\.\d+)?|\d\.\d+)', raw)
    estimate = float(m.group(1)) if m else 0.5
    estimate = max(0.0, min(1.0, estimate))
    pe = goal.ideal - estimate

    # Store in memory
    memory.add_pe_record(
        turn=current_turn,
        partner_text=self._last_partner_text,
        estimate=estimate,
        pe=pe,
    )

    self._last_estimate = estimate
    self._last_pe = pe

    result = f'Estimated state: {estimate:.2f}, PE: {pe:+.2f}'
    self._logging_channel({
        'Key': self.get_pre_act_label(),
        'Value': result
    })
    return result

  def get_state(self) -> entity_component.ComponentState:
    """Get component state."""
    return {
        'last_estimate': self._last_estimate,
        'last_pe': self._last_pe,
        'last_partner_text': self._last_partner_text,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Set component state."""
    self._last_estimate = state.get('last_estimate')
    self._last_pe = state.get('last_pe')
    self._last_partner_text = state.get('last_partner_text', '')


class PEReflectionComponent(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging
):
  """Component to generate reflection to reduce PE."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory_component_key: str = DEFAULT_PE_MEMORY_COMPONENT_KEY,
      pre_act_label: str = 'PE Reflection',
  ):
    """Initialize PE reflection component.

    Args:
      model: Language model for reflection generation.
      memory_component_key: Key of PEMemoryComponent.
      pre_act_label: Label for pre_act output.
    """
    super().__init__(pre_act_label)
    self._model = model
    self._memory_component_key = memory_component_key
    self._last_reflection: str = ''

  def _make_pre_act_value(self) -> str:
    """Make pre-act value (returns empty for reflection component)."""
    return ''

  def post_observe(self) -> str:
    """Generate reflection after PE estimation."""
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=PEMemoryComponent
    )
    goal = memory.get_goal()
    pe_last = memory.get_last_pe()

    # Get current turn
    current_turn = len(memory.get_recent_conversation()) + 1

    prompt = f"""You are {self.get_entity().name}. Goal: {goal.name}.
Given the PE of last turn (PE_last = {pe_last:+.3f}), write a short reflection:
What will you change next turn to REDUCE PE? Keep it concrete and brief.
"""

    text = self._model.sample_text(prompt).strip()
    memory.add_reflection(turn=current_turn, text=text)
    self._last_reflection = text

    self._logging_channel({
        'Key': self.get_pre_act_label(),
        'Value': text
    })
    return text

  def get_state(self) -> entity_component.ComponentState:
    """Get component state."""
    return {'last_reflection': self._last_reflection}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Set component state."""
    self._last_reflection = state.get('last_reflection', '')


class PEActComponent(entity_component.ActingComponent):
  """Component to generate utterance based on PE history."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory_component_key: str = DEFAULT_PE_MEMORY_COMPONENT_KEY,
  ):
    """Initialize PE act component.

    Args:
      model: Language model for action generation.
      memory_component_key: Key of PEMemoryComponent.
    """
    super().__init__()
    self._model = model
    self._memory_component_key = memory_component_key

  def get_action_attempt(
      self,
      context: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    """Generate utterance based on PE context."""
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=PEMemoryComponent
    )
    goal = memory.get_goal()
    recent_k = memory._recent_k

    conv_k = memory.get_recent_conversation()
    pe_k = memory.get_recent_pe_history()
    refl_k = memory.get_recent_reflections()

    def fmt_conv(u: Utterance) -> str:
      return f'[t={u.turn} {u.speaker}] {u.text}'

    def fmt_pe(p: PERecord) -> str:
      return (
          f'(turn {p.turn}) estimate={p.estimate:.2f}, PE={p.pe:+.2f} '
          f'← partner: "{p.partner_text}"'
      )

    def fmt_refl(r: ReflectionRecord) -> str:
      return f'(turn {r.turn}) {r.text}'

    prompt = f"""You are {self.get_entity().name}. Your goal is "{goal.name}".
Definition: {goal.description}
Ideal value: {goal.ideal:.2f}

You must talk in a way that MINIMIZES PREDICTION ERROR (PE = ideal - estimated current state).
Consider recent conversation, PE history, and your reflections.

Recent conversation (last {recent_k}):
{chr(10).join("- " + fmt_conv(u) for u in conv_k) or "- (none)"}

Recent PE history:
{chr(10).join("- " + fmt_pe(p) for p in pe_k) or "- (none)"}

Recent reflections:
{chr(10).join("- " + fmt_refl(r) for r in refl_k) or "- (none)"}

Now produce ONE concise utterance to your partner that is likely to REDUCE PE next turn.
Avoid meta-talk; speak naturally.
"""

    text = self._model.sample_text(prompt).strip()

    # Store own utterance
    # Get current turn from conversation length
    current_turn = len(conv_k) + 1
    memory.add_utterance(
        turn=current_turn,
        speaker=self.get_entity().name,
        text=text,
    )

    return text

  def get_state(self) -> entity_component.ComponentState:
    """Get component state."""
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Set component state."""
    pass
  def set_state(self, state: entity_component.ComponentState) -> None:
    """Set component state."""
    pass
