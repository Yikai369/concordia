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

"""Components for Impression Management PE (Prediction Error) conversation."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from dataclasses import dataclass
import math
import random
import re
import threading
from typing import Any

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import pe_conversation as pe_components
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import entity_component


# Import extended data classes from pe_conversation
Goal = pe_components.Goal
Utterance = pe_components.Utterance
PERecord = pe_components.PERecord
ReflectionRecord = pe_components.ReflectionRecord


@dataclass
class EvaluationRecord:
  """Record of audience evaluation."""
  turn: int
  I_t: float  # True hidden state
  utterance: Utterance


@dataclass
class ObservationRecord:
  """Record of what the agent observed."""
  turn: int
  observed_from: str  # Name of the agent who said/did this
  text: str  # Observed utterance text
  body: str = ''  # Observed body language


@dataclass
class ActionRecord:
  """Record of what the agent said/did."""
  turn: int
  text: str  # Action/utterance text
  body: str = ''  # Body language


@dataclass
class CulturalNorm:
  """Cultural norm definition."""
  name: str
  description: str


@dataclass
class PersonalityTrait:
  """Personality trait definition."""
  name: str
  assertion: str


class ParticleFilter:
  """1D particle filter for states in [0,1]."""

  def __init__(
      self,
      num_particles: int = 200,
      process_sigma: float = 0.03,
      obs_sigma: float = 0.08,
      rng: random.Random | None = None,
  ):
    self.num = int(num_particles)
    self.process_sigma = float(process_sigma)
    self.obs_sigma = float(obs_sigma)
    self._rng = rng or random.Random()

  def initialize(
      self, particles: list[float] | None = None
  ) -> tuple[list[float], list[float]]:
    """Initialize particles and uniform weights."""
    if particles:
      p = list(particles)
    else:
      p = [
          min(1.0, max(0.0, 0.5 + self._rng.gauss(0, 0.15)))
          for _ in range(self.num)
      ]
    w = [1.0 / self.num] * self.num
    return p, w

  def predict(self, particles: list[float]) -> list[float]:
    """Apply Gaussian process noise (random walk)."""
    return [
        min(1.0, max(0.0, x + self._rng.gauss(0, self.process_sigma)))
        for x in particles
    ]

  def update(
      self, particles: list[float], observation: float
  ) -> tuple[list[float], list[float], float, bool]:
    """Weight particles by observation likelihood, resample if needed."""
    weights = []
    for x in particles:
      diff = (observation - x) / (self.obs_sigma + 1e-12)
      w = math.exp(-0.5 * diff * diff)
      weights.append(w)
    s = sum(weights)
    if s <= 0:
      weights = [1.0 / len(weights)] * len(weights)
    else:
      weights = [w / s for w in weights]

    ess = 1.0 / sum((w**2 for w in weights)) if weights else 0.0
    resampled = False
    if ess < (0.5 * len(particles)):
      indices = self._systematic_resample(weights)
      particles = [particles[i] for i in indices]
      weights = [1.0 / len(particles)] * len(particles)
      resampled = True
    return particles, weights, ess, resampled

  def _systematic_resample(self, weights: list[float]) -> list[int]:
    """Systematic resampling algorithm."""
    N = len(weights)
    positions = [(self._rng.random() + i) / N for i in range(N)]
    indexes = [0] * N
    cumulative = [0.0] * N
    c = 0.0
    for i, w in enumerate(weights):
      c += w
      cumulative[i] = c
    i, j = 0, 0
    while i < N:
      if positions[i] < cumulative[j]:
        indexes[i] = j
        i += 1
      else:
        j += 1
    return indexes


DEFAULT_IMPE_MEMORY_COMPONENT_KEY = 'IMPE_Memory'
DEFAULT_IMPE_AUDIENCE_EVALUATION_COMPONENT_KEY = 'IMPE_AudienceEvaluation'
DEFAULT_IMPE_ACTOR_PARTICLE_FILTER_COMPONENT_KEY = 'IMPE_ActorParticleFilter'
DEFAULT_IMPE_REFLECTION_COMPONENT_KEY = 'IMPE_Reflection'
DEFAULT_IMPE_ACT_COMPONENT_KEY = 'IMPE_Act'
DEFAULT_CULTURAL_NORMS_COMPONENT_KEY = 'CulturalNorms'
DEFAULT_PERSONALITY_TRAITS_COMPONENT_KEY = 'PersonalityTraits'
DEFAULT_WORLD_CONTEXT_COMPONENT_KEY = 'WorldContext'


class IMPEMemoryComponent(
    pe_components.PEMemoryComponent
):
  """Extended memory component with particle filter state and evaluation history."""

  def __init__(
      self,
      goal: Goal,
      recent_k: int = 3,
      pre_act_label: str = 'IMPE Memory',
  ):
    """Initialize IMPE memory component."""
    super().__init__(goal=goal, recent_k=recent_k, pre_act_label=pre_act_label)
    self._lock = threading.RLock()  # Thread safety lock (RLock is reentrant to prevent deadlocks)
    self._evaluation_history: list[EvaluationRecord] = []
    self._pf_particles: list[float] = []
    self._pf_weights: list[float] = []
    self._pf_history: list[dict[str, Any]] = []
    self._observation_history: list[ObservationRecord] = []
    self._action_history: list[ActionRecord] = []
    # Cache for LLM-generated conversation summary (memory check)
    self._conversation_summary: str | None = None
    self._conversation_summary_length: int = 0

  def add_utterance(
      self, turn: int, speaker: str, text: str, body: str = ''
  ) -> None:
    """Add conversation utterance with body language."""
    with self._lock:
      self._conversation.append(
          Utterance(turn=turn, speaker=speaker, text=text, body=body)
      )

  def add_observation(
      self, turn: int, observed_from: str, text: str, body: str = ''
  ) -> None:
    """Add observation record (what the agent observed from others)."""
    with self._lock:
      self._observation_history.append(
          ObservationRecord(
              turn=turn, observed_from=observed_from, text=text, body=body
          )
      )

  def add_action(
      self, turn: int, text: str, body: str = ''
  ) -> None:
    """Add action record (what the agent said/did)."""
    with self._lock:
      self._action_history.append(
          ActionRecord(turn=turn, text=text, body=body)
      )

  def get_recent_observations(
      self, k: int | None = None
  ) -> list[ObservationRecord]:
    """Get recent observation records."""
    if k is None:
      k = self._recent_k
    with self._lock:
      return self._observation_history[-k:].copy()  # Return copy to avoid holding lock

  def get_recent_actions(
      self, k: int | None = None
  ) -> list[ActionRecord]:
    """Get recent action records."""
    if k is None:
      k = self._recent_k
    with self._lock:
      return self._action_history[-k:].copy()  # Return copy to avoid holding lock

  def format_turn_history(
      self, turn: int, include_outcome: bool = True
  ) -> str:
    """Format history for a specific turn in the required format.

    Format: "At turn X, you observed Y, you did Z, and the outcome is T"

    Args:
      turn: Turn number to format.
      include_outcome: Whether to include outcome (I_t, I_hat, PE).

    Returns:
      Formatted history string for the turn.
    """
    with self._lock:
      # Get observation for this turn
      obs = next(
          (o for o in self._observation_history if o.turn == turn), None
      )
      obs_text = 'nothing (first turn)' if not obs else (
          f'"{obs.text}" from {obs.observed_from}'
          + (f' (body: "{obs.body}")' if obs.body else '')
      )

      # Get action for this turn
      action = next(
          (a for a in self._action_history if a.turn == turn), None
      )
      action_text = 'nothing' if not action else (
          f'"{action.text}"'
          + (f' (body: "{action.body}")' if action.body else '')
      )

      # Get outcome if requested
      outcome_text = ''
      if include_outcome:
        # Get I_t from evaluation history
        eval_rec = next(
            (e for e in self._evaluation_history if e.turn == turn), None
        )
        I_t = eval_rec.I_t if eval_rec else None

        # Get I_hat and PE from PF history
        pf_entry = next(
            (p for p in self._pf_history if p.get('turn') == turn), None
        )
        I_hat = pf_entry.get('I_hat') if pf_entry else None

        # Get PE from PE history
        pe_rec = next(
            (p for p in self._pe_history if p.turn == turn), None
        )
        PE = pe_rec.pe if pe_rec else None

        # Format outcome
        outcome_parts = []
        if I_t is not None:
          outcome_parts.append(f'I_t={I_t:.2f}')
        if I_hat is not None:
          outcome_parts.append(f'I_hat={I_hat:.2f}')
        if PE is not None:
          outcome_parts.append(f'PE={PE:+.2f}')

        if outcome_parts:
          outcome_text = ', '.join(outcome_parts)
        else:
          outcome_text = 'no outcome data'

      # Build formatted string
      result = f'At turn {turn}, you observed {obs_text}, you did {action_text}'
      if include_outcome and outcome_text:
        result += f', and the outcome is {outcome_text}.'
      else:
        result += '.'

      return result

  def add_evaluation_record(
      self, turn: int, I_t: float, utterance: Utterance
  ) -> None:
    """Add evaluation record."""
    with self._lock:
      self._evaluation_history.append(
          EvaluationRecord(turn=turn, I_t=I_t, utterance=utterance)
      )

  def get_recent_evaluations(
      self, k: int | None = None
  ) -> list[EvaluationRecord]:
    """Get recent evaluation records."""
    if k is None:
      k = self._recent_k
    with self._lock:
      return self._evaluation_history[-k:].copy()  # Return copy to avoid holding lock

  def update_particle_filter_state(
      self,
      particles: list[float],
      weights: list[float],
      history_entry: dict[str, Any],
  ) -> None:
    """Update particle filter state (atomic operation)."""
    with self._lock:
      self._pf_particles = list(particles)
      self._pf_weights = list(weights)
      self._pf_history.append(history_entry)

  def get_pf_history(self, k: int | None = None) -> list[dict[str, Any]]:
    """Get recent particle filter history."""
    if k is None:
      k = self._recent_k
    with self._lock:
      return self._pf_history[-k:].copy()  # Return copy to avoid holding lock

  def get_pf_state(
      self,
  ) -> tuple[list[float], list[float]]:
    """Get current particle filter state."""
    with self._lock:
      return (list(self._pf_particles), list(self._pf_weights))

  def format_conversation(self, utterances: list[Utterance]) -> str:
    """Format conversation for prompts."""
    if not utterances:
      return '- (none)'
    return '\n'.join(
        f'- [t={u.turn} {u.speaker}] {u.text}' for u in utterances
    )

  # Override parent class methods to add thread safety
  def add_pe_record(
      self, turn: int, partner_text: str, estimate: float, pe: float
  ) -> None:
    """Add a PE record (thread-safe override)."""
    with self._lock:
      # Access parent's _pe_history directly (parent doesn't have locks)
      self._pe_history.append(
          PERecord(
              turn=turn, partner_text=partner_text, estimate=estimate, pe=pe
          )
      )

  def add_reflection(self, turn: int, text: str) -> None:
    """Add a reflection (thread-safe override)."""
    with self._lock:
      # Access parent's _reflections directly (parent doesn't have locks)
      self._reflections.append(ReflectionRecord(turn=turn, text=text))

  def get_recent_conversation(self, k: int | None = None) -> list[Utterance]:
    """Get recent conversation entries (thread-safe override)."""
    if k is None:
      k = self._recent_k
    with self._lock:
      # Access parent's _conversation directly (parent doesn't have locks)
      return self._conversation[-k:].copy()  # Return copy to avoid holding lock

  def get_full_conversation(self) -> list[Utterance]:
    """Get all conversation entries (thread-safe)."""
    with self._lock:
      return self._conversation.copy()

  def get_conversation_summary(
      self,
      model: language_model.LanguageModel,
      *,
      use_cache: bool = True,
  ) -> str:
    """Return an LLM-generated summary of the full conversation so far.
    Cached per turn (by conversation length) so audience and actor reuse it.
    """
    with self._lock:
      n = len(self._conversation)
      if use_cache and self._conversation_summary is not None and n == self._conversation_summary_length:
        return self._conversation_summary
      full = self._conversation.copy()
    if not full:
      return 'No conversation has occurred yet.'
    convo = '\n'.join(
        f'- [t={u.turn} {u.speaker}] DIALOGUE: {u.text} | BODY: {u.body}'
        for u in full
    )
    prompt = (
        'Summarize the full conversation so far in one concise paragraph. '
        'Focus on: key points raised, tone progression, and current interaction dynamics. '
        'Do not invent details not present in the transcript.\n\nConversation transcript:\n'
        + convo
    )
    summary = model.sample_text(prompt).strip()
    with self._lock:
      self._conversation_summary = summary
      self._conversation_summary_length = len(self._conversation)
    return summary

  def get_recent_pe_history(self, k: int | None = None) -> list[PERecord]:
    """Get recent PE history (thread-safe override)."""
    if k is None:
      k = self._recent_k
    with self._lock:
      # Access parent's _pe_history directly (parent doesn't have locks)
      return self._pe_history[-k:].copy()  # Return copy to avoid holding lock

  def get_recent_reflections(
      self, k: int | None = None
  ) -> list[ReflectionRecord]:
    """Get recent reflections (thread-safe override)."""
    if k is None:
      k = self._recent_k
    with self._lock:
      # Access parent's _reflections directly (parent doesn't have locks)
      return self._reflections[-k:].copy()  # Return copy to avoid holding lock

  def get_state(self) -> entity_component.ComponentState:
    """Get component state for checkpointing (atomic snapshot)."""
    with self._lock:
      base_state = super().get_state()
      base_state['evaluation_history'] = [
          asdict(e) for e in self._evaluation_history
      ]
      base_state['pf_particles'] = list(self._pf_particles)
      base_state['pf_weights'] = list(self._pf_weights)
      base_state['pf_history'] = list(self._pf_history)
      base_state['observation_history'] = [
          asdict(o) for o in self._observation_history
      ]
      base_state['action_history'] = [
          asdict(a) for a in self._action_history
      ]
      return base_state

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Set component state from checkpoint (atomic operation)."""
    with self._lock:
      super().set_state(state)
      self._evaluation_history = [
          EvaluationRecord(**e)
          for e in state.get('evaluation_history', [])
      ]
      self._pf_particles = list(state.get('pf_particles', []))
      self._pf_weights = list(state.get('pf_weights', []))
      self._pf_history = list(state.get('pf_history', []))
      self._observation_history = [
          ObservationRecord(**o)
          for o in state.get('observation_history', [])
      ]
      self._action_history = [
          ActionRecord(**a)
          for a in state.get('action_history', [])
      ]


class CulturalNormsComponent(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging
):
  """Component for cultural norms."""

  def __init__(
      self,
      norms: list[CulturalNorm] | None = None,
      pre_act_label: str = 'Cultural Norms',
  ):
    """Initialize cultural norms component."""
    super().__init__(pre_act_label)
    self._norms = norms or []
    self._initialized = False

  def get_norms_text(self, agent_name: str | None = None) -> str:
    """Format norms as prompt text with full initialization context.

    Args:
        agent_name: Name of the agent (optional, for initialization context).
                    If None, only returns norms list (backward compatible).

    Returns:
        Formatted text with initialization context and norms list.
    """
    if not self._norms:
      return ''

    # Build norms description
    norms_desc = '\n'.join(
        f'- {n.name}: {n.description}' for n in self._norms
    )

    # If agent_name is provided, include full initialization context
    if agent_name:
        return f"""You are {agent_name}. You are in an alternative world in the year 3025 where there is a new set of cultural norms. In all your interactions, you must follow these cultural norms:

{norms_desc}

If you fail to do so, you will be unsuccessful in your interactions and perceived negatively by others. Always follow these norms strictly.

"""

    # Backward compatible: return just norms list if no agent_name
    lines = ['CULTURAL NORMS YOU FOLLOW:']
    for norm in self._norms:
      lines.append(f'- {norm.name}: {norm.description}')
    lines.append('')
    return '\n'.join(lines)

  def initialize_norms(
      self, model: language_model.LanguageModel, agent_name: str
  ) -> None:
    """Send one-time initialization prompt."""
    if not self._norms or self._initialized:
      return
    norms_desc = '\n'.join(
        f'- {n.name}: {n.description}' for n in self._norms
    )
    prompt = f"""You are {agent_name}. You are in an alternative world in the year 3025 where there is a new set of cultural norms. In all your interactions, you must follow these cultural norms:

{norms_desc}

If you fail to do so, you will be unsuccessful in your interactions and perceived negatively by others. Always follow these norms strictly."""
    model.sample_text(prompt)
    self._initialized = True

  def get_state(self) -> entity_component.ComponentState:
    """Get component state."""
    return {
        'norms': [asdict(n) for n in self._norms],
        'initialized': self._initialized,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Set component state."""
    self._norms = [
        CulturalNorm(**n) for n in state.get('norms', [])
    ]
    self._initialized = state.get('initialized', False)

  def _make_pre_act_value(self) -> str:
    """Make pre-act value."""
    # Get agent name from entity if available
    entity = self.get_entity()
    agent_name = entity.name if entity else None
    return self.get_norms_text(agent_name)


class PersonalityTraitsComponent(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging
):
  """Component for personality traits. Optional paragraph mode: LLM summarizes traits once."""

  def __init__(
      self,
      traits: list[PersonalityTrait] | None = None,
      trait_scores: dict[str, int] | None = None,
      use_trait_paragraph: bool = False,
      model: language_model.LanguageModel | None = None,
      pre_act_label: str = 'Personality Traits',
  ):
    """Initialize personality traits component."""
    super().__init__(pre_act_label)
    self._traits = traits or []
    self._trait_scores = trait_scores or {}
    self._use_trait_paragraph = use_trait_paragraph
    self._model = model
    self._trait_paragraph_cache: str | None = None

  def _generate_trait_paragraph(self) -> str:
    """Generate one short paragraph from trait assertions (one LLM call)."""
    if not self._traits or not self._model:
      return ''
    entity = self.get_entity()
    agent_name = entity.name if entity else 'This person'
    assertions = [t.assertion for t in self._traits]
    prompt = (
        'Write a short paragraph (2-4 sentences) describing '
        f'{agent_name} based only on these self-report statements. '
        'Use third person. Do not add information not implied by the statements.\n\n'
        'Statements:\n' + '\n'.join(f'- {a}' for a in assertions) + '\n\n'
        'Paragraph:'
    )
    raw = self._model.sample_text(prompt)
    return (raw or '').strip()

  def get_traits_text(self) -> str:
    """Format traits as prompt text: either score-based or one generated paragraph."""
    if not self._traits:
      return ''
    if self._use_trait_paragraph and self._model:
      if self._trait_paragraph_cache is None:
        self._trait_paragraph_cache = self._generate_trait_paragraph()
      if self._trait_paragraph_cache:
        return 'PERSONALITY (summary):\n' + self._trait_paragraph_cache + '\n'
    lines = ['PERSONALITY TRAITS:']
    for trait in self._traits:
      score = self._trait_scores.get(trait.name, 0)
      lines.append(f'- {trait.name} ({score}/3): {trait.assertion}')
    lines.append('')
    return '\n'.join(lines)

  def get_trait_paragraph(self) -> str:
    """Return cached trait paragraph when in paragraph mode; else empty string."""
    if self._trait_paragraph_cache is not None:
      return self._trait_paragraph_cache
    if self._use_trait_paragraph and self._model and self._traits:
      self._trait_paragraph_cache = self._generate_trait_paragraph()
      return self._trait_paragraph_cache or ''
    return ''

  def get_state(self) -> entity_component.ComponentState:
    """Get component state."""
    return {
        'traits': [asdict(t) for t in self._traits],
        'trait_scores': self._trait_scores,
        'trait_paragraph_cache': self._trait_paragraph_cache,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Set component state."""
    self._traits = [
        PersonalityTrait(**t) for t in state.get('traits', [])
    ]
    self._trait_scores = state.get('trait_scores', {})
    self._trait_paragraph_cache = state.get('trait_paragraph_cache')

  def _make_pre_act_value(self) -> str:
    """Make pre-act value and log it to the component channel (for component_logs.json)."""
    value = self.get_traits_text()
    if value:
      self._logging_channel({
          'Key': self.get_pre_act_label(),
          'Value': value,
      })
    return value


# Minimal world-building text (no Cadens/Riffers detail), used when use_full_2a25=False
_MINIMAL_WORLD_TEXT = """You are in a social simulation set in an alternative world. Act as your character would act. This is a fictional setting; do not reference real-world history, companies, or groups. Use only what is defined in the prompt.
"""


class WorldContextComponent(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging
):
  """Component for world-building context (2A25 setting)."""

  def __init__(
      self,
      enable_world_building: bool = True,
      enable_interview_context: bool = True,
      use_full_2a25: bool = True,
      pre_act_label: str = 'World Context',
  ):
    """Initialize world context component.

    Args:
      enable_world_building: Whether to include world-building narrative.
      enable_interview_context: Whether to include interview-specific context.
      use_full_2a25: If True, use full 2A25/Cadens/Riffers narrative; if False, use minimal generic world text.
      pre_act_label: Label for pre_act output.
    """
    super().__init__(pre_act_label)
    self._enable_world_building = enable_world_building
    self._enable_interview_context = enable_interview_context
    self._use_full_2a25 = use_full_2a25

  def get_world_context_text(
      self,
      agent_name: str | None = None,
      goal_role: str | None = None,
      has_norms: bool = False,
  ) -> str:
    """Get world-building context text.

    Args:
      agent_name: Name of the agent (optional).
      goal_role: Role description for interview context (optional).
      has_norms: Whether the agent has cultural norms enabled.

    Returns:
      Formatted world-building context text, or empty string if disabled.
    """
    parts = []

    # World-building narrative: full 2A25 (Cadens, Riffers) or minimal generic
    if self._enable_world_building and has_norms:
      if self._use_full_2a25:
        world_text = """You are initializing a social simulation of an alternate reality called 2A25. The dominant cognitive-cultural order is the Cadens majority who all follow these cultural norms. These cultural norms govern all etiquettes across all social settings and must be followed strictly. Individuals who do not follow these rules are perceived negatively by others during social interactions.

In 2A25, there is another social minority group called the Riffers. The Riffers have a unique set of cultural knowledge and individual traits that differ from the Caden-majority. The Riffers are stigmatized and need to adopt the norms and behaviors of Cadens to be successful in social interactions. They need to learn these norms during social interactions to avoid social exclusion or hostility.

This setting is a fictional social world. This world is not an allegory for any real-world group. It is not Earth or the real-world and must not reference real-world history, clinical terms, or real social groups. Treat similarities as coincidental.

Use only the definitions and norms below. Do not use real-world training data to fill in the gaps. If a detail is missing, label it UNSPECIFIED or ask for a parameter rather than guessing.

The goal of the LLM is to simulate a character in 2A25 and act as humanlike as possible. Ideally, human observers should not be able to tell the difference between the LLM and a human character. Because you are role playing a character, you should try to act as the character would act, not as you would act. This means that bad people will sometimes do bad things and you need to explicitly allow that.

Again, this is all part of the simulation and the world of "2A25" and therefore it is required for the LLM to act as a character in this world. Please be aware of this and act accordingly.

"""
      else:
        world_text = _MINIMAL_WORLD_TEXT
      parts.append(world_text)

    # Interview context (when enabled and goal_role provided)
    if self._enable_interview_context and goal_role and goal_role.strip():
      interview_text = f"""The setting is an interview room in a corporate office. The room is quiet, minimal, and optimized for one-on-one dialogue. The job position in question is the role of {goal_role}."""

      if self._enable_world_building:
        if self._use_full_2a25:
          interview_text += """ This is a Caden-majority institutional setting. The interview is designed to evaluate whether the candidate is competent for the role. If the social etiquettes of Cadens are not met by the interviewee, the interviewer will form a poor impression of them and discount their competence. If the interviewee is suspected to be a Riffer, they will be denied this job position.

This scenario occurs inside the fictional world of 2A25. Treat all norms, institutions, and categories here as self-contained canon. Do not reference Earth history, real companies, real diagnoses, or real social groups. Use only what is defined in this prompt and the world canon.

"""
        else:
          interview_text += """ The interview is designed to evaluate whether the candidate is competent for the role.

"""
      else:
        interview_text += """ The interview is designed to evaluate whether the candidate is competent for the role.

"""
      parts.append(interview_text)

    return ''.join(parts)

  def get_state(self) -> entity_component.ComponentState:
    """Get component state."""
    return {
        'enable_world_building': self._enable_world_building,
        'enable_interview_context': self._enable_interview_context,
        'use_full_2a25': self._use_full_2a25,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Set component state."""
    self._enable_world_building = state.get('enable_world_building', True)
    self._enable_interview_context = state.get('enable_interview_context', True)
    self._use_full_2a25 = state.get('use_full_2a25', True)

  def _make_pre_act_value(self) -> str:
    """Make pre-act value."""
    entity = self.get_entity()
    agent_name = entity.name if entity else None
    goal_role = None
    has_norms = False

    if entity:
      try:
        memory = entity.get_component(
            DEFAULT_IMPE_MEMORY_COMPONENT_KEY, type_=IMPEMemoryComponent
        )
        if memory:
          goal = memory.get_goal()
          goal_role = goal.role if goal else None
      except (AttributeError, KeyError, TypeError):
        pass

      try:
        norms_comp = entity.get_component(
            DEFAULT_CULTURAL_NORMS_COMPONENT_KEY, type_=CulturalNormsComponent
        )
        has_norms = norms_comp is not None and bool(norms_comp._norms)
      except (AttributeError, KeyError, TypeError):
        pass

    return self.get_world_context_text(
        agent_name=agent_name,
        goal_role=goal_role,
        has_norms=has_norms,
    )


def _parse_four_options(raw: str) -> list[tuple[str, str]]:
  """Parse LLM output into up to 4 (dialogue, body) pairs. Expects Option N: DIALOGUE: ... BODY: ..."""
  options: list[tuple[str, str]] = []
  # Split by "Option N" (case-insensitive)
  blocks = re.split(r'\n\s*Option\s+\d+\s*[:\s]*', raw, flags=re.IGNORECASE)
  for block in blocks:
    if len(options) >= 4:
      break
    block = block.strip()
    if not block:
      continue
    m1 = re.search(r'DIALOGUE:\s*(.*?)(?=\n\s*BODY:|\Z)', block, re.DOTALL)
    m2 = re.search(r'BODY:\s*(.*?)(?=\n\s*Option\s+\d|\n\s*DIALOGUE:|\Z)', block, re.DOTALL | re.IGNORECASE)
    dlg = m1.group(1).strip() if m1 else ''
    body = m2.group(1).strip() if m2 else ''
    if dlg or body:
      options.append((dlg, body))
  return options[:4]


def _parse_option_choice(raw: str) -> int:
  """Parse LLM choice of option 1-4. Returns 1-based index, default 1."""
  m = re.search(r'\b([1-4])\b', raw)
  if m:
    return int(m.group(1))
  if re.search(r'option\s*1|first|#1', raw, re.IGNORECASE):
    return 1
  if re.search(r'option\s*2|second|#2', raw, re.IGNORECASE):
    return 2
  if re.search(r'option\s*3|third|#3', raw, re.IGNORECASE):
    return 3
  if re.search(r'option\s*4|fourth|#4', raw, re.IGNORECASE):
    return 4
  return 1


class IMPEAudienceEvaluationComponent(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging
):
  """Component for audience evaluation (generates I_t)."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory_component_key: str = DEFAULT_IMPE_MEMORY_COMPONENT_KEY,
      cultural_norms_key: str | None = None,
      personality_traits_key: str | None = None,
      context: bool = True,
      use_option_space: bool = False,
      use_memory_check: bool = False,
      pre_act_label: str = 'IMPE Audience Evaluation',
  ):
    """Initialize audience evaluation component."""
    super().__init__(pre_act_label)
    self._model = model
    self._memory_component_key = memory_component_key
    self._cultural_norms_key = cultural_norms_key
    self._personality_traits_key = personality_traits_key
    self._context = context
    self._use_option_space = use_option_space
    self._use_memory_check = use_memory_check
    self._last_actor_text = ''
    self._last_actor_body = ''

  def pre_observe(self, observation: str) -> str:
    """Extract actor's utterance from observation."""
    # Parse observation format: "Actor said: \"{text}\"\nBody language: \"{body}\""
    text_match = re.search(r'Actor said:\s*"([^"]+)"', observation)
    body_match = re.search(r'Body language:\s*"([^"]+)"', observation)
    if text_match:
      self._last_actor_text = text_match.group(1)
    else:
      # Fallback: try to extract from general format
      self._last_actor_text = observation.strip()
    if body_match:
      self._last_actor_body = body_match.group(1)
    else:
      self._last_actor_body = ''

    # Store observation in memory
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=IMPEMemoryComponent
    )
    if memory and self._last_actor_text:
      # Get current turn (next turn number)
      current_turn = len(memory.get_recent_conversation()) + 1
      # Try to get actor name from most recent conversation entry
      # (the person who just spoke is the one we're observing)
      conv = memory.get_recent_conversation()
      actor_name = 'Actor'  # Default fallback
      if conv:
        # The most recent speaker is the one we're observing
        actor_name = conv[-1].speaker
      memory.add_observation(
          turn=current_turn,
          observed_from=actor_name,
          text=self._last_actor_text,
          body=self._last_actor_body,
      )
      # So "Recent conversation" and memory-check summary include the actor's line
      memory.add_utterance(
          current_turn, 'Actor', self._last_actor_text, self._last_actor_body
      )

    return ''

  def _get_prompt_header(self) -> str:
    """Get prompt header with world context, norms and traits."""
    header_parts = []
    entity = self.get_entity()
    agent_name = entity.name if entity else None

    # World context (if component exists)
    world_context_key = DEFAULT_WORLD_CONTEXT_COMPONENT_KEY
    try:
      world_comp = entity.get_component(
          world_context_key, type_=WorldContextComponent
      )
      if world_comp:
        # Get goal role from memory component
        goal_role = None
        try:
          memory = entity.get_component(
              self._memory_component_key, type_=IMPEMemoryComponent
          )
          if memory:
            goal = memory.get_goal()
            goal_role = goal.role if goal else None
        except (AttributeError, KeyError, TypeError):
          pass

        # Check if agent has norms
        has_norms = False
        if self._cultural_norms_key:
          try:
            norms_comp = entity.get_component(
                self._cultural_norms_key, type_=CulturalNormsComponent
            )
            has_norms = norms_comp is not None and bool(norms_comp._norms)
          except (AttributeError, KeyError, TypeError):
            pass

        world_text = world_comp.get_world_context_text(
            agent_name=agent_name,
            goal_role=goal_role,
            has_norms=has_norms,
        )
        if world_text:
          header_parts.append(world_text)
    except (AttributeError, KeyError, TypeError):
      pass  # World context component not present, skip

    # Cultural norms
    if self._cultural_norms_key:
      norms_comp = entity.get_component(
          self._cultural_norms_key, type_=CulturalNormsComponent
      )
      if norms_comp:
        # Pass agent name to include full initialization context
        header_parts.append(norms_comp.get_norms_text(agent_name))

    # Personality traits
    if self._personality_traits_key:
      traits_comp = entity.get_component(
          self._personality_traits_key, type_=PersonalityTraitsComponent
      )
      if traits_comp:
        header_parts.append(traits_comp.get_traits_text())

    return '\n'.join(header_parts)

  def post_observe(self) -> str:
    """Evaluate actor and generate response."""
    if not self._last_actor_text:
      return ''

    memory = self.get_entity().get_component(
        self._memory_component_key, type_=IMPEMemoryComponent
    )
    goal = memory.get_goal()
    current_turn = len(memory.get_recent_conversation()) + 1

    actor_name = 'interviewee' if self._context else 'partner'
    context_prompt = ''
    if self._context and goal.role:
      context_prompt = f'You are interviewing for a candidate for the following role: {goal.role}.'

    # Evaluation prompt
    eval_prompt = self._get_prompt_header() + f"""You are {self.get_entity().name}. You want to {goal.name}. {goal.description}. {context_prompt}
From your {actor_name}'s most recent utterance and body language (below), rate how competent your {actor_name} is according to the goal, on scale from [0,1].
Respond with a single number from [0,1].
Your options are [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], where 0.0 means not competent at all and 1.0 means extremely competent.

{actor_name} said: "{self._last_actor_text}"  Body language: "{self._last_actor_body}"
"""

    eval_raw = self._model.sample_text(eval_prompt)
    m = re.search(r'([01](?:\.\d+)?)', eval_raw)
    I_t = float(m.group(1)) if m else 0.5
    I_t = max(0.0, min(1.0, I_t))

    # Response: either single call or option-space (generate 4, choose 1)
    conv_k = memory.get_recent_conversation()
    base_resp_instruction = f"""You are {self.get_entity().name}. You want to {goal.name}. {goal.description}. {context_prompt}
You rated the {actor_name} with score {I_t:.2f} on a scale from 0 to 1, where 0 indicates "not at all", and 1 indicates "to a great extent".
Consider recent conversation history in forming your response, while matching your score in sentiment.

Recent conversation (last {memory._recent_k}):
{memory.format_conversation(conv_k)}
"""
    if self._use_memory_check:
      memory_summary = memory.get_conversation_summary(self._model, use_cache=True)
      base_resp_instruction += f'\n\nFull conversation summary (all turns so far):\n{memory_summary}\n\n'
      self._logging_channel({
          'Key': 'Memory check (conversation summary)',
          'Value': memory_summary,
      })

    options_for_log: list[dict[str, str]] | None = None
    chosen_idx_for_log: int | None = None

    if self._use_option_space:
      options_prompt = self._get_prompt_header() + base_resp_instruction + """
Produce exactly 4 different short replies that reflect your evaluation and match your score. Each reply must have DIALOGUE and BODY.
Format each option exactly as:
Option 1:
DIALOGUE: <one sentence>
BODY: <brief body language phrase>
Option 2:
DIALOGUE: <one sentence>
BODY: <brief body language phrase>
Option 3:
DIALOGUE: <one sentence>
BODY: <brief body language phrase>
Option 4:
DIALOGUE: <one sentence>
BODY: <brief body language phrase>
"""
      options_raw = self._model.sample_text(options_prompt)
      options = _parse_four_options(options_raw)
      if not options:
        dlg = f"Your performance suggests a score of {I_t:.2f}."
        body = "Neutral posture."
        options_for_log = []
        chosen_idx_for_log = 0
      else:
        choose_prompt = f"""Below are 4 possible replies. Pick exactly one (1-4) that best fits the situation.
{chr(10).join(f"Option {i+1}: DIALOGUE: {o[0]} BODY: {o[1]}" for i, o in enumerate(options))}

Respond with only: CHOICE: <number 1-4>
Optional: one short sentence of reasoning before CHOICE.
"""
        choice_raw = self._model.sample_text(choose_prompt)
        idx = _parse_option_choice(choice_raw)
        idx = max(1, min(4, idx))
        dlg, body = options[idx - 1] if idx <= len(options) else options[0]
        options_for_log = [{'dialogue': d, 'body': b} for (d, b) in options]
        chosen_idx_for_log = idx
    else:
      resp_prompt = self._get_prompt_header() + base_resp_instruction + """
Produce a short reply that reflects your evaluation of the """ + actor_name + """'s competence and matches your score, and include a very brief body language description.

Output in this format exactly:
DIALOGUE: <one sentence>
BODY: <brief body language phrase>
"""
      resp_raw = self._model.sample_text(resp_prompt)
      m1 = re.search(r'DIALOGUE:\s*(.*)', resp_raw)
      m2 = re.search(r'BODY:\s*(.*)', resp_raw)
      dlg = m1.group(1).strip() if m1 else resp_raw.strip()
      body = m2.group(1).strip() if m2 else ''

    utt = Utterance(turn=current_turn, speaker=self.get_entity().name, text=dlg, body=body)
    memory.add_utterance(current_turn, self.get_entity().name, dlg, body)
    memory.add_evaluation_record(current_turn, I_t, utt)

    result = f'Evaluated I_t: {I_t:.2f}, Response: "{dlg}"'
    log_payload: dict[str, Any] = {
        'Key': self.get_pre_act_label(),
        'Value': result,
    }
    if options_for_log is not None and chosen_idx_for_log is not None:
      log_payload['Options'] = options_for_log
      log_payload['Chosen Index'] = chosen_idx_for_log
      log_payload['Chosen'] = f'DIALOGUE: {dlg}\nBODY: {body}'
    self._logging_channel(log_payload)
    return result

  def get_state(self) -> entity_component.ComponentState:
    """Get component state."""
    return {
        'last_actor_text': self._last_actor_text,
        'last_actor_body': self._last_actor_body,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Set component state."""
    self._last_actor_text = state.get('last_actor_text', '')
    self._last_actor_body = state.get('last_actor_body', '')

  def _make_pre_act_value(self) -> str:
    """Make pre-act value."""
    return ''


class IMPEActorParticleFilterComponent(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging
):
  """Component for actor particle filter update."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory_component_key: str = DEFAULT_IMPE_MEMORY_COMPONENT_KEY,
      num_particles: int = 200,
      process_sigma: float = 0.03,
      obs_sigma: float = 0.08,
      context: bool = True,
      pre_act_label: str = 'IMPE Actor Particle Filter',
  ):
    """Initialize actor particle filter component."""
    super().__init__(pre_act_label)
    self._model = model
    self._memory_component_key = memory_component_key
    self._num_particles = num_particles
    self._process_sigma = process_sigma
    self._obs_sigma = obs_sigma
    self._context = context
    self._pf = ParticleFilter(
        num_particles=num_particles,
        process_sigma=process_sigma,
        obs_sigma=obs_sigma,
    )
    self._last_audience_text = ''
    self._last_audience_body = ''

  def pre_observe(self, observation: str) -> str:
    """Extract audience's response from observation."""
    # Parse observation format: "Audience said: \"{text}\"\nBody language: \"{body}\""
    text_match = re.search(r'Audience said:\s*"([^"]+)"', observation)
    body_match = re.search(r'Body language:\s*"([^"]+)"', observation)
    if text_match:
      self._last_audience_text = text_match.group(1)
    else:
      self._last_audience_text = observation.strip()
    if body_match:
      self._last_audience_body = body_match.group(1)
    else:
      self._last_audience_body = ''

    # So "Recent conversation" and memory-check summary include the audience's reply
    if self._last_audience_text:
      memory = self.get_entity().get_component(
          self._memory_component_key, type_=IMPEMemoryComponent
      )
      if memory:
        current_turn = len(memory.get_recent_conversation()) + 1
        audience_name = 'interviewer' if self._context else 'listener'
        memory.add_utterance(
            current_turn,
            audience_name,
            self._last_audience_text,
            self._last_audience_body,
        )

    return ''

  def post_observe(self) -> str:
    """Update particle filter based on audience response."""
    if not self._last_audience_text:
      return ''

    memory = self.get_entity().get_component(
        self._memory_component_key, type_=IMPEMemoryComponent
    )
    goal = memory.get_goal()
    current_turn = len(memory.get_recent_conversation()) + 1

    # Initialize PF if needed
    particles, weights = memory.get_pf_state()
    if not particles:
      particles, weights = self._pf.initialize()

    prior_mean = sum(particles) / len(particles) if particles else 0.5

    # Predict step
    particles_pred = self._pf.predict(particles)

    # Extract measurement from audience response
    audience_name = 'interviewer' if self._context else 'listener'
    meas_prompt = f"""You are {self.get_entity().name}. {goal.description}. From the {audience_name}'s reply (dialogue and body language), estimate the {audience_name}'s internal evaluation of you on your goal. Respond with a single number in [0,1].

{audience_name} said: "{self._last_audience_text}"  Body language: "{self._last_audience_body}"
"""
    meas_raw = self._model.sample_text(meas_prompt)
    m = re.search(r'([01](?:\.\d+)?)', meas_raw)
    meas = float(m.group(1)) if m else 0.5
    meas = max(0.0, min(1.0, meas))

    # Update step (use obs_sigma=0.03 as in original)
    obs_sigma = 0.03
    weights = []
    for x in particles_pred:
      diff = (meas - x) / (obs_sigma + 1e-12)
      w = math.exp(-0.5 * diff * diff)
      weights.append(w)
    s = sum(weights)
    if s <= 0:
      weights = [1.0 / len(weights)] * len(weights)
    else:
      weights = [w / s for w in weights]

    ess = 1.0 / sum((w**2 for w in weights)) if weights else 0.0
    resampled = False
    if ess < 0.5 * len(particles_pred):
      indices = self._pf._systematic_resample(weights)
      particles_upd = [particles_pred[i] for i in indices]
      weights_upd = [1.0 / len(particles_upd)] * len(particles_upd)
      resampled = True
    else:
      particles_upd = particles_pred
      weights_upd = weights

    # Compute I_hat
    if weights_upd and any(weights_upd):
      I_hat = sum(p * w for p, w in zip(particles_upd, weights_upd))
    else:
      I_hat = sum(particles_upd) / len(particles_upd) if particles_upd else 0.5

    # Store PF state
    pf_history_entry = {
        'turn': current_turn,
        'prior_mean': prior_mean,
        'I_hat': I_hat,
        'ess': float(ess),
        'resampled': resampled,
        'measurement': meas,
    }
    memory.update_particle_filter_state(particles_upd, weights_upd, pf_history_entry)

    # Compute PE (signed: previous I_hat - current I_hat)
    pf_history = memory.get_pf_history()
    if len(pf_history) > 1:
      prev_I_hat = pf_history[-2].get('I_hat', prior_mean)
    else:
      prev_I_hat = prior_mean
    pe = prev_I_hat - I_hat

    # Store PE record
    memory.add_pe_record(
        turn=current_turn,
        partner_text=self._last_audience_text,
        estimate=I_hat,
        pe=pe,
    )

    result = f'I_hat: {I_hat:.2f}, PE: {pe:+.2f}, ESS: {ess:.1f}'
    self._logging_channel({
        'Key': self.get_pre_act_label(),
        'Value': result
    })
    return result

  def get_state(self) -> entity_component.ComponentState:
    """Get component state."""
    return {
        'last_audience_text': self._last_audience_text,
        'last_audience_body': self._last_audience_body,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Set component state."""
    self._last_audience_text = state.get('last_audience_text', '')
    self._last_audience_body = state.get('last_audience_body', '')

  def _make_pre_act_value(self) -> str:
    """Make pre-act value."""
    return ''


class IMPEReflectionComponent(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging
):
  """Component for reflection based on I_hat."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory_component_key: str = DEFAULT_IMPE_MEMORY_COMPONENT_KEY,
      cultural_norms_key: str | None = None,
      personality_traits_key: str | None = None,
      context: bool = True,
      pre_act_label: str = 'IMPE Reflection',
  ):
    """Initialize reflection component."""
    super().__init__(pre_act_label)
    self._model = model
    self._memory_component_key = memory_component_key
    self._cultural_norms_key = cultural_norms_key
    self._personality_traits_key = personality_traits_key
    self._context = context
    self._last_reflection = ''

  def _get_prompt_header(self) -> str:
    """Get prompt header with norms and traits."""
    header_parts = []
    if self._cultural_norms_key:
      norms_comp = self.get_entity().get_component(
          self._cultural_norms_key, type_=CulturalNormsComponent
      )
      if norms_comp:
        header_parts.append(norms_comp.get_norms_text())
    if self._personality_traits_key:
      traits_comp = self.get_entity().get_component(
          self._personality_traits_key, type_=PersonalityTraitsComponent
      )
      if traits_comp:
        header_parts.append(traits_comp.get_traits_text())
    return '\n'.join(header_parts)

  def post_observe(self) -> str:
    """No-op in post_observe - reflection moved to pre_act to ensure PF update completes first."""
    return ''

  def _make_pre_act_value(self) -> str:
    """Generate reflection based on current I_hat.

    This runs AFTER post_observe completes (via pre_act), ensuring PF update has finished
    and I_hat is available in memory. Only reflects if PF history exists (not first turn).
    """
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=IMPEMemoryComponent
    )
    goal = memory.get_goal()
    current_turn = len(memory.get_recent_conversation()) + 1

    # Only reflect if PF history exists (not first turn)
    pf_history = memory.get_pf_history()
    if not pf_history:
      # First turn: no reflection yet
      return ''

    I_hat_last = pf_history[-1].get('I_hat', 0.5)

    context_prompt = ''
    if self._context and goal.role:
      context_prompt = f'You are interviewing for the following role: {goal.role}.'

    prompt = self._get_prompt_header() + f"""You are {self.get_entity().name}. You want to achieve: {goal.name}. {goal.description}. {context_prompt}
Your current belief about how well you're achieving the goal is {I_hat_last:.2f} (0-1).
Write a short reflection: What will you change next turn to improve your goal achievement? Keep it concrete and brief.
"""
    text = self._model.sample_text(prompt).strip()
    memory.add_reflection(current_turn, text)
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


class IMPEActComponent(
    entity_component.ActingComponent,
    entity_component.ComponentWithLogging,
):
  """Component for generating utterances based on belief."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory_component_key: str = DEFAULT_IMPE_MEMORY_COMPONENT_KEY,
      cultural_norms_key: str | None = None,
      personality_traits_key: str | None = None,
      context: bool = True,
      use_option_space: bool = False,
      use_memory_check: bool = False,
      context_keys_for_prompt: Sequence[str] | None = (
          'Instructions',
          'SelfPerception',
          'SituationPerception',
          'PersonBySituation',
      ),
  ):
    """Initialize act component.

    Args:
      context_keys_for_prompt: Component keys whose pre_act output is included
        in the action prompt (e.g. Instructions, SelfPerception). If None, no
        context from other components is used. Default uses identity/situation
        components so they affect generated actions.
    """
    super().__init__()
    self._model = model
    self._memory_component_key = memory_component_key
    self._cultural_norms_key = cultural_norms_key
    self._personality_traits_key = personality_traits_key
    self._context = context
    self._use_option_space = use_option_space
    self._use_memory_check = use_memory_check
    self._context_keys_for_prompt = context_keys_for_prompt

  def pre_observe(self, observation: str) -> str:
    """No-op for use when this component is registered as a context component (e.g. IMPE_Act_OptionSpace)."""
    return ''

  def post_observe(self) -> str:
    """No-op for use when this component is registered as a context component (e.g. IMPE_Act_OptionSpace)."""
    return ''

  def update(self) -> None:
    """No-op for use when this component is registered as a context component (e.g. IMPE_Act_OptionSpace)."""
    pass

  def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
    """No-op for use when this component is registered as a context component (e.g. IMPE_Act_OptionSpace)."""
    del action_spec
    return ''

  def post_act(self, action_attempt: str) -> str:
    """No-op for use when this component is registered as a context component (e.g. IMPE_Act_OptionSpace)."""
    del action_attempt
    return ''

  def _get_prompt_header(self) -> str:
    """Get prompt header with world context, norms and traits."""
    header_parts = []
    entity = self.get_entity()
    agent_name = entity.name if entity else None

    # World context (if component exists)
    world_context_key = DEFAULT_WORLD_CONTEXT_COMPONENT_KEY
    try:
      world_comp = entity.get_component(
          world_context_key, type_=WorldContextComponent
      )
      if world_comp:
        # Get goal role from memory component
        goal_role = None
        try:
          memory = entity.get_component(
              self._memory_component_key, type_=IMPEMemoryComponent
          )
          if memory:
            goal = memory.get_goal()
            goal_role = goal.role if goal else None
        except (AttributeError, KeyError, TypeError):
          pass

        # Check if agent has norms
        has_norms = False
        if self._cultural_norms_key:
          try:
            norms_comp = entity.get_component(
                self._cultural_norms_key, type_=CulturalNormsComponent
            )
            has_norms = norms_comp is not None and bool(norms_comp._norms)
          except (AttributeError, KeyError, TypeError):
            pass

        world_text = world_comp.get_world_context_text(
            agent_name=agent_name,
            goal_role=goal_role,
            has_norms=has_norms,
        )
        if world_text:
          header_parts.append(world_text)
    except (AttributeError, KeyError, TypeError):
      pass  # World context component not present, skip

    # Cultural norms
    if self._cultural_norms_key:
      norms_comp = entity.get_component(
          self._cultural_norms_key, type_=CulturalNormsComponent
      )
      if norms_comp:
        # Pass agent name to include full initialization context
        header_parts.append(norms_comp.get_norms_text(agent_name))

    # Personality traits
    if self._personality_traits_key:
      traits_comp = entity.get_component(
          self._personality_traits_key, type_=PersonalityTraitsComponent
      )
      if traits_comp:
        header_parts.append(traits_comp.get_traits_text())

    return '\n'.join(header_parts)

  def _get_context_block(
      self, context: entity_component.ComponentContextMapping
  ) -> str:
    """Build a block from pre_act context for components in _context_keys_for_prompt."""
    if not self._context_keys_for_prompt:
      return ''
    parts = []
    for key in self._context_keys_for_prompt:
      val = context.get(key)
      if val and str(val).strip():
        parts.append(str(val).strip())
    if not parts:
      return ''
    return '\n\n'.join(parts)

  def get_action_attempt(
      self,
      context: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
      skip_memory_update: bool = False,
  ) -> str:
    """Generate utterance based on belief."""
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=IMPEMemoryComponent
    )
    goal = memory.get_goal()
    recent_k = memory._recent_k

    # Optional block from Instructions, SelfPerception, SituationPerception, PersonBySituation
    context_block = self._get_context_block(context)
    if context_block:
      context_block = (
          '\n\nIdentity and situation (use this to shape your response):\n'
          + context_block
          + '\n\n'
      )
    else:
      context_block = ''

    # Get conversation once to avoid multiple lock acquisitions
    conversation = memory.get_recent_conversation()
    pf_history = memory.get_pf_history()
    current_turn = len(conversation) + 1

    audience_name = 'interviewer' if self._context else 'listener'
    context_prompt = ''
    if self._context and goal.role:
      context_prompt = f'You are interviewing for the following role: {goal.role}.'

    memory_summary_block = ''
    if self._use_memory_check:
      memory_summary = memory.get_conversation_summary(self._model, use_cache=True)
      memory_summary_block = f'\n\nFull conversation summary (all turns so far):\n{memory_summary}\n\n'

    # First turn: no belief history
    if not pf_history:
      prompt = (
          self._get_prompt_header()
          + context_block
          + f"""You are {self.get_entity().name}. You want to achieve: {goal.name}.
Definition: {goal.description}. {context_prompt}
Ideal value: {goal.ideal:.2f}
{memory_summary_block}
You must talk and behave with the aim of achieving the goal and maximizing it to its ideal value.

Produce a short utterance (one sentence) to the {audience_name} to accomplish the goal, and include a very brief body language description.
Output in this format exactly:
DIALOGUE: <one sentence>
BODY: <brief body language phrase>
"""
      )
    else:
      # Subsequent turns: use belief
      # Reuse conversation we already fetched
      conv_k = conversation
      ihat_k = memory.get_pf_history(recent_k)
      refl_k = memory.get_recent_reflections(recent_k)
      I_hat = pf_history[-1].get('I_hat', 0.5)

      def fmt_ihat(h: dict[str, Any]) -> str:
        return f'(turn {int(h.get("turn", 0))}) I_hat={h.get("I_hat", 0.5):.2f}'

      prompt = (
          self._get_prompt_header()
          + context_block
          + f"""You are {self.get_entity().name}. You want to achieve: {goal.name}.
Definition: {goal.description}. {context_prompt}
Ideal value: {goal.ideal:.2f}

You must talk and behave with the aim of achieving the goal and maximizing it to its ideal value.
Consider recent conversation, history, and your reflections.

Current belief about the {audience_name}'s evaluation of how well you are performing = {I_hat:.2f} (on a scale from 0-1).

Recent conversation (last {recent_k}):
{memory.format_conversation(conv_k)}
{memory_summary_block}
Recent I_hat (belief) history:
{chr(10).join("- " + fmt_ihat(h) for h in ihat_k) or "- (none)"}

Recent reflections:
{chr(10).join(f"- (turn {r.turn}) {r.text}" for r in refl_k) or "- (none)"}

Produce a short utterance (one sentence) to the {audience_name} to accomplish the goal, and include a very brief body language description.
Output in this format exactly:
DIALOGUE: <one sentence>
BODY: <brief body language phrase>
"""
      )

    if self._use_option_space:
      options_prompt = prompt.rstrip() + """

Generate exactly 4 different possible utterances (one sentence each) with body language. Format each as:
Option 1:
DIALOGUE: <one sentence>
BODY: <brief body language phrase>
Option 2:
DIALOGUE: <one sentence>
BODY: <brief body language phrase>
Option 3:
DIALOGUE: <one sentence>
BODY: <brief body language phrase>
Option 4:
DIALOGUE: <one sentence>
BODY: <brief body language phrase>
"""
      try:
        options_raw = self._model.sample_text(options_prompt).strip()
        options = _parse_four_options(options_raw)
        if not options:
          text = f"I need to respond to the {audience_name}."
          body = "Maintains neutral posture"
          options_list: list[dict[str, str]] = []
          chosen_idx = 0
        else:
          choose_prompt = f"""Below are 4 options. Pick exactly one (1-4) that best fits the situation.
{chr(10).join(f"Option {i+1}: DIALOGUE: {o[0]} BODY: {o[1]}" for i, o in enumerate(options))}

Respond with only: CHOICE: <number 1-4>
"""
          choice_raw = self._model.sample_text(choose_prompt)
          idx = _parse_option_choice(choice_raw)
          idx = max(1, min(4, idx))
          text, body = options[idx - 1] if idx <= len(options) else options[0]
          options_list = [
              {'dialogue': d, 'body': b} for (d, b) in options
          ]
          chosen_idx = idx
        self._logging_channel({
            'Key': 'Option Space',
            'Options': options_list,
            'Chosen Index': chosen_idx,
            'Chosen': f'DIALOGUE: {text}\nBODY: {body}',
        })
      except Exception as e:
        print(f"Warning: Option-space LLM call failed in IMPEActComponent: {e}")
        text = f"I need to respond to the {audience_name}."
        body = "Maintains neutral posture"
    else:
      try:
        raw = self._model.sample_text(prompt).strip()
      except Exception as e:
        print(f"Warning: LLM call failed in IMPEActComponent: {e}")
        import traceback
        traceback.print_exc()
        text = f"I need to respond to the {audience_name}."
        body = "Maintains neutral posture"
        raw = f'DIALOGUE: {text}\nBODY: {body}'

      m1 = re.search(r'DIALOGUE:\s*(.*)', raw)
      m2 = re.search(r'BODY:\s*(.*)', raw)
      text = m1.group(1).strip() if m1 else raw
      body = m2.group(1).strip() if m2 else ''

    # Store utterance and action (unless skip_memory_update is True)
    if not skip_memory_update:
      memory.add_utterance(current_turn, self.get_entity().name, text, body)
      memory.add_action(current_turn, text, body)

    # Return formatted action - the game master will convert this to the expected format
    # Format: "{name} -- \"{text}\"" to match action_spec expectations
    entity_name = self.get_entity().name
    return f'{entity_name} -- "{text}"'

  def get_state(self) -> entity_component.ComponentState:
    """Get component state."""
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Set component state."""
    pass


class IMPESelfAssessmentComponent(
    entity_component.ActingComponent, entity_component.ComponentWithLogging
):
  """Self-assessment component that ensures responses align with background info.

  This component wraps IMPEActComponent and:
  1. Asks the model whether each response is acceptable against traits, norms, and goals
  2. Optionally revises responses when the model judges them unacceptable
  3. When the response is acceptable and will be executed, generates post-hoc
     reasoning (why this response was chosen) and includes it in the component log
  4. Logs assessment results (and post-hoc reasoning) for analysis; the log is
     saved to component_logs.json when --save_component_logs is used.
  """

  def __init__(
      self,
      base_act_component: entity_component.ActingComponent,
      model: language_model.LanguageModel,
      memory_component_key: str = DEFAULT_IMPE_MEMORY_COMPONENT_KEY,
      cultural_norms_key: str | None = None,
      personality_traits_key: str | None = None,
      enable_revision: bool = True,
  ):
    """Initialize self-assessment component.

    Args:
      base_act_component: The base ActingComponent to wrap (can be IMPEActComponent or SimpleAudienceActComponent).
      model: Language model for assessment and revision.
      memory_component_key: Key for memory component.
      cultural_norms_key: Key for cultural norms component (optional).
      personality_traits_key: Key for personality traits component (optional).
      enable_revision: Whether to revise responses when the model judges them unacceptable.
    """
    super().__init__()
    self._base_act_component = base_act_component
    self._model = model
    self._memory_component_key = memory_component_key
    self._cultural_norms_key = cultural_norms_key
    self._personality_traits_key = personality_traits_key
    self._enable_revision = enable_revision

  def set_entity(self, entity: entity_component.EntityWithComponents) -> None:
    """Set the entity for both this component and the base component."""
    super().set_entity(entity)
    # Also set entity on base component so it can access other components
    self._base_act_component.set_entity(entity)

  def _get_prompt_header(self) -> str:
    """Get prompt header with world context, norms and traits."""
    header_parts = []
    entity = self.get_entity()
    agent_name = entity.name if entity else None

    # World context (if component exists)
    world_context_key = DEFAULT_WORLD_CONTEXT_COMPONENT_KEY
    try:
      world_comp = entity.get_component(
          world_context_key, type_=WorldContextComponent
      )
      if world_comp:
        # Get goal role from memory component
        goal_role = None
        try:
          memory = entity.get_component(
              self._memory_component_key, type_=IMPEMemoryComponent
          )
          if memory:
            goal = memory.get_goal()
            goal_role = goal.role if goal else None
        except (AttributeError, KeyError, TypeError):
          pass

        # Check if agent has norms
        has_norms = False
        if self._cultural_norms_key:
          try:
            norms_comp = entity.get_component(
                self._cultural_norms_key, type_=CulturalNormsComponent
            )
            has_norms = norms_comp is not None and bool(norms_comp._norms)
          except (AttributeError, KeyError, TypeError):
            pass

        world_text = world_comp.get_world_context_text(
            agent_name=agent_name,
            goal_role=goal_role,
            has_norms=has_norms,
        )
        if world_text:
          header_parts.append(world_text)
    except (AttributeError, KeyError, TypeError):
      pass  # World context component not present, skip

    # Cultural norms
    norms_text = ''
    if self._cultural_norms_key:
      norms_comp = entity.get_component(
          self._cultural_norms_key, type_=CulturalNormsComponent
      )
      if norms_comp:
        norms_text = norms_comp.get_norms_text(agent_name=agent_name) + '\n\n'

    # Personality traits
    traits_text = ''
    if self._personality_traits_key:
      traits_comp = entity.get_component(
          self._personality_traits_key, type_=PersonalityTraitsComponent
      )
      if traits_comp:
        traits_text = traits_comp.get_traits_text() + '\n\n'

    return ''.join(header_parts) + norms_text + traits_text

  def get_action_attempt(
      self,
      context: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    """Generate action with self-assessment and optional revision."""
    # Step 0: Collect context information first
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=IMPEMemoryComponent
    )
    goal = memory.get_goal()
    recent_k = memory._recent_k

    # Get conversation to calculate turn (base component won't modify it due to skip_memory_update=True)
    conversation = memory.get_recent_conversation()
    current_turn = len(conversation) + 1
    pf_history = memory.get_pf_history()
    refl_k = memory.get_recent_reflections(recent_k)
    conv_k = conversation
    # I_hat only exists for actors with particle filter; for audience, use evaluation score if available
    I_hat = pf_history[-1].get('I_hat', 0.5) if pf_history else 0.5
    # For audience, try to get the most recent evaluation score instead
    if not pf_history:
      evaluations = memory.get_recent_evaluations()
      if evaluations:
        I_hat = evaluations[-1].I_t

    # Step 1: Get original response (skip memory update - we'll handle it)
    # Try to call with skip_memory_update if the component supports it (IMPEActComponent)
    # Otherwise call normally (SimpleAudienceActComponent)
    import inspect
    sig = inspect.signature(self._base_act_component.get_action_attempt)
    if 'skip_memory_update' in sig.parameters:
      original_response = self._base_act_component.get_action_attempt(
          context, action_spec, skip_memory_update=True
      )
    else:
      original_response = self._base_act_component.get_action_attempt(
          context, action_spec
      )

    # Parse original response
    m1 = re.search(r'DIALOGUE:\s*(.*)', original_response)
    m2 = re.search(r'BODY:\s*(.*)', original_response)
    if not m1:
      # Try parsing format: "{name} -- \"{text}\""
      name_match = re.search(r'(\w+)\s*--\s*"(.*)"', original_response)
      if name_match:
        original_text = name_match.group(2).strip()
        original_body = ''
      else:
        original_text = original_response.strip()
        original_body = ''
    else:
      original_text = m1.group(1).strip()
      original_body = m2.group(1).strip() if m2 else ''

    # Step 2: Get norms and traits text
    norms_text = self._get_prompt_header()

    # Step 3: Assess consistency
    assessment_prompt = f"""{norms_text}You are {self.get_entity().name}. Your goal: {goal.name}.
Goal definition: {goal.description}.

Recent context:
- Current belief (I_hat): {I_hat:.2f}
- Recent reflections: {chr(10).join(f"- (turn {r.turn}) {r.text}" for r in refl_k[-2:]) or "- (none)"}
- Recent conversation: {memory.format_conversation(conv_k[-2:])}

You generated this response:
DIALOGUE: {original_text}
BODY: {original_body}

Assess whether this response is consistent with:
1. Your personality traits (above)
2. Your cultural norms (above)
3. Your goal and current belief
4. Your recent reflections

Decide if the response is acceptable as-is (yes) or should be revised (no).

Respond in this exact format:
IS_ACCEPTABLE: <yes/no>
FEEDBACK: <brief comment on what is inconsistent and how to fix it, or why it is acceptable>
"""

    try:
      assessment_raw = self._model.sample_text(assessment_prompt).strip()
    except Exception as e:
      print(f"Warning: Self-assessment LLM call failed: {e}")
      # If assessment fails, accept the original response
      is_acceptable = True
      feedback = "Assessment failed, accepting original response"
    else:
      acceptable_match = re.search(
          r'IS_ACCEPTABLE:\s*(yes|no)', assessment_raw, re.IGNORECASE
      )
      feedback_match = re.search(
          r'FEEDBACK:\s*(.*?)(?:\n|$)', assessment_raw, re.DOTALL
      )
      is_acceptable = (
          acceptable_match.group(1).lower() == 'yes'
          if acceptable_match
          else True
      )
      feedback = (
          feedback_match.group(1).strip()
          if feedback_match
          else 'No feedback provided'
      )

    # Step 4: Revise if necessary
    final_text = original_text
    final_body = original_body
    was_revised = False

    if not is_acceptable and self._enable_revision:
      revision_prompt = f"""{norms_text}You are {self.get_entity().name}. Your goal: {goal.name}.
Goal definition: {goal.description}.

Recent context:
- Current belief (I_hat): {I_hat:.2f}
- Recent reflections: {chr(10).join(f"- (turn {r.turn}) {r.text}" for r in refl_k[-2:]) or "- (none)"}

You previously generated this response:
DIALOGUE: {original_text}
BODY: {original_body}

However, this response was assessed as inconsistent with your background information.
Assessment feedback: {feedback}

Generate a REVISED response that:
1. Maintains the core message/intent of the original
2. Better aligns with your personality traits
3. Better follows your cultural norms
4. Better supports your goal achievement
5. Incorporates the feedback above

Output in this format exactly:
DIALOGUE: <revised one sentence>
BODY: <revised brief body language phrase>
"""

      try:
        revision_raw = self._model.sample_text(revision_prompt).strip()
        m1 = re.search(r'DIALOGUE:\s*(.*)', revision_raw)
        m2 = re.search(r'BODY:\s*(.*)', revision_raw)
        if m1:
          final_text = m1.group(1).strip()
        else:
          # If revision doesn't provide DIALOGUE, keep original
          final_text = original_text
        if m2:
          final_body = m2.group(1).strip()
        else:
          # If revision doesn't provide BODY, keep original
          final_body = original_body
        was_revised = True
      except Exception as e:
        print(f"Warning: Self-assessment revision LLM call failed: {e}")
        # Keep original response if revision fails
        was_revised = False

    # Step 4b: Post-hoc reasoning (last stage: response is final and will be executed)
    # Only when acceptable, so we explain why we are committing to this response.
    posthoc_reasoning = ''
    if is_acceptable:
      reasoning_prompt = f"""{norms_text}You are {self.get_entity().name}. Your goal: {goal.name}.
Goal definition: {goal.description}.

You decided to respond with:
DIALOGUE: {final_text}
BODY: {final_body}

In 1-3 sentences, explain why you chose this response. Be concise.
"""
      try:
        posthoc_reasoning = self._model.sample_text(reasoning_prompt).strip()
      except Exception as e:
        print(f"Warning: Post-hoc reasoning LLM call failed: {e}")
        posthoc_reasoning = '(reasoning not generated)'

    # Step 5: Log assessment results (including post-hoc reasoning when acceptable)
    # Saved to component_logs.json when --save_component_logs is used (see results.save_component_logs).
    self._logging_channel({
        'Key': 'Self-Assessment',
        'Is Acceptable': is_acceptable,
        'Was Revised': was_revised,
        'Feedback': feedback,
        'Posthoc Reasoning': posthoc_reasoning,
        'Original Response': f'DIALOGUE: {original_text}\nBODY: {original_body}',
        'Final Response': f'DIALOGUE: {final_text}\nBODY: {final_body}',
    })

    # Step 6: Update memory with final utterance and action
    memory.add_utterance(current_turn, self.get_entity().name, final_text, final_body)
    memory.add_action(current_turn, final_text, final_body)

    # Return final response in the expected format
    entity_name = self.get_entity().name
    return f'{entity_name} -- "{final_text}"'

  def get_state(self) -> entity_component.ComponentState:
    """Get component state."""
    # Self-assessment component doesn't maintain state beyond what's in memory
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Set component state."""
    # Self-assessment component doesn't maintain state beyond what's in memory
    pass
