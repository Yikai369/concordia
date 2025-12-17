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

from collections.abc import Mapping
from dataclasses import asdict
from dataclasses import dataclass
import math
import random
import re
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
    self._evaluation_history: list[EvaluationRecord] = []
    self._pf_particles: list[float] = []
    self._pf_weights: list[float] = []
    self._pf_history: list[dict[str, Any]] = []

  def add_utterance(
      self, turn: int, speaker: str, text: str, body: str = ''
  ) -> None:
    """Add conversation utterance with body language."""
    self._conversation.append(
        Utterance(turn=turn, speaker=speaker, text=text, body=body)
    )

  def add_evaluation_record(
      self, turn: int, I_t: float, utterance: Utterance
  ) -> None:
    """Add evaluation record."""
    self._evaluation_history.append(
        EvaluationRecord(turn=turn, I_t=I_t, utterance=utterance)
    )

  def get_recent_evaluations(
      self, k: int | None = None
  ) -> list[EvaluationRecord]:
    """Get recent evaluation records."""
    if k is None:
      k = self._recent_k
    return self._evaluation_history[-k:]

  def update_particle_filter_state(
      self,
      particles: list[float],
      weights: list[float],
      history_entry: dict[str, Any],
  ) -> None:
    """Update particle filter state."""
    self._pf_particles = list(particles)
    self._pf_weights = list(weights)
    self._pf_history.append(history_entry)

  def get_pf_history(self, k: int | None = None) -> list[dict[str, Any]]:
    """Get recent particle filter history."""
    if k is None:
      k = self._recent_k
    return self._pf_history[-k:]

  def get_pf_state(
      self,
  ) -> tuple[list[float], list[float]]:
    """Get current particle filter state."""
    return (list(self._pf_particles), list(self._pf_weights))

  def format_conversation(self, utterances: list[Utterance]) -> str:
    """Format conversation for prompts."""
    if not utterances:
      return '- (none)'
    return '\n'.join(
        f'- [t={u.turn} {u.speaker}] {u.text}' for u in utterances
    )

  def get_state(self) -> entity_component.ComponentState:
    """Get component state for checkpointing."""
    base_state = super().get_state()
    base_state['evaluation_history'] = [
        asdict(e) for e in self._evaluation_history
    ]
    base_state['pf_particles'] = self._pf_particles
    base_state['pf_weights'] = self._pf_weights
    base_state['pf_history'] = self._pf_history
    return base_state

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Set component state from checkpoint."""
    super().set_state(state)
    self._evaluation_history = [
        EvaluationRecord(**e)
        for e in state.get('evaluation_history', [])
    ]
    self._pf_particles = state.get('pf_particles', [])
    self._pf_weights = state.get('pf_weights', [])
    self._pf_history = state.get('pf_history', [])


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

  def get_norms_text(self) -> str:
    """Format norms as prompt text."""
    if not self._norms:
      return ''
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
    return self.get_norms_text()


class PersonalityTraitsComponent(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging
):
  """Component for personality traits."""

  def __init__(
      self,
      traits: list[PersonalityTrait] | None = None,
      trait_scores: dict[str, int] | None = None,
      pre_act_label: str = 'Personality Traits',
  ):
    """Initialize personality traits component."""
    super().__init__(pre_act_label)
    self._traits = traits or []
    self._trait_scores = trait_scores or {}

  def get_traits_text(self) -> str:
    """Format traits with scores as prompt text."""
    if not self._traits:
      return ''
    lines = ['PERSONALITY TRAITS:']
    for trait in self._traits:
      score = self._trait_scores.get(trait.name, 0)
      lines.append(f'- {trait.name} ({score}/3): {trait.assertion}')
    lines.append('')
    return '\n'.join(lines)

  def get_state(self) -> entity_component.ComponentState:
    """Get component state."""
    return {
        'traits': [asdict(t) for t in self._traits],
        'trait_scores': self._trait_scores,
    }

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Set component state."""
    self._traits = [
        PersonalityTrait(**t) for t in state.get('traits', [])
    ]
    self._trait_scores = state.get('trait_scores', {})

  def _make_pre_act_value(self) -> str:
    """Make pre-act value."""
    return self.get_traits_text()


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
      pre_act_label: str = 'IMPE Audience Evaluation',
  ):
    """Initialize audience evaluation component."""
    super().__init__(pre_act_label)
    self._model = model
    self._memory_component_key = memory_component_key
    self._cultural_norms_key = cultural_norms_key
    self._personality_traits_key = personality_traits_key
    self._context = context
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
    return ''

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

    # Response prompt
    conv_k = memory.get_recent_conversation()
    resp_prompt = self._get_prompt_header() + f"""You are {self.get_entity().name}. You want to {goal.name}. {goal.description}. {context_prompt}
You rated the {actor_name} with score {I_t:.2f} on a scale from 0 to 1, where 0 indicates "not at all", and 1 indicates "to a great extent".
Produce a short reply that reflects your evaluation of the {actor_name}'s competence and matches your score, and include a very brief body language description.

Consider recent conversation history in forming your response, while matching your score in sentiment.

Recent conversation (last {memory._recent_k}):
{memory.format_conversation(conv_k)}

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
    self._logging_channel({
        'Key': self.get_pre_act_label(),
        'Value': result
    })
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
    """Generate reflection based on current I_hat."""
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=IMPEMemoryComponent
    )
    goal = memory.get_goal()
    current_turn = len(memory.get_recent_conversation()) + 1

    pf_history = memory.get_pf_history()
    I_hat_last = pf_history[-1].get('I_hat', 0.5) if pf_history else 0.5

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

  def _make_pre_act_value(self) -> str:
    """Make pre-act value."""
    return ''


class IMPEActComponent(entity_component.ActingComponent):
  """Component for generating utterances based on belief."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory_component_key: str = DEFAULT_IMPE_MEMORY_COMPONENT_KEY,
      cultural_norms_key: str | None = None,
      personality_traits_key: str | None = None,
      context: bool = True,
  ):
    """Initialize act component."""
    super().__init__()
    self._model = model
    self._memory_component_key = memory_component_key
    self._cultural_norms_key = cultural_norms_key
    self._personality_traits_key = personality_traits_key
    self._context = context

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

  def get_action_attempt(
      self,
      context: entity_component.ComponentContextMapping,
      action_spec: entity_lib.ActionSpec,
  ) -> str:
    """Generate utterance based on belief."""
    memory = self.get_entity().get_component(
        self._memory_component_key, type_=IMPEMemoryComponent
    )
    goal = memory.get_goal()
    recent_k = memory._recent_k

    pf_history = memory.get_pf_history()
    current_turn = len(memory.get_recent_conversation()) + 1

    audience_name = 'interviewer' if self._context else 'listener'
    context_prompt = ''
    if self._context and goal.role:
      context_prompt = f'You are interviewing for the following role: {goal.role}.'

    # First turn: no belief history
    if not pf_history:
      prompt = self._get_prompt_header() + f"""You are {self.get_entity().name}. You want to achieve: {goal.name}.
Definition: {goal.description}. {context_prompt}
Ideal value: {goal.ideal:.2f}

You must talk and behave with the aim of achieving the goal and maximizing it to its ideal value.

Produce a short utterance (one sentence) to the {audience_name} to accomplish the goal, and include a very brief body language description.
Output in this format exactly:
DIALOGUE: <one sentence>
BODY: <brief body language phrase>
"""
    else:
      # Subsequent turns: use belief
      conv_k = memory.get_recent_conversation()
      ihat_k = memory.get_pf_history(recent_k)
      refl_k = memory.get_recent_reflections(recent_k)
      I_hat = pf_history[-1].get('I_hat', 0.5)

      def fmt_ihat(h: dict[str, Any]) -> str:
        return f'(turn {int(h.get("turn", 0))}) I_hat={h.get("I_hat", 0.5):.2f}'

      prompt = self._get_prompt_header() + f"""You are {self.get_entity().name}. You want to achieve: {goal.name}.
Definition: {goal.description}. {context_prompt}
Ideal value: {goal.ideal:.2f}

You must talk and behave with the aim of achieving the goal and maximizing it to its ideal value.
Consider recent conversation, history, and your reflections.

Current belief about the {audience_name}'s evaluation of how well you are performing = {I_hat:.2f} (on a scale from 0-1).

Recent conversation (last {recent_k}):
{memory.format_conversation(conv_k)}

Recent I_hat (belief) history:
{chr(10).join("- " + fmt_ihat(h) for h in ihat_k) or "- (none)"}

Recent reflections:
{chr(10).join(f"- (turn {r.turn}) {r.text}" for r in refl_k) or "- (none)"}

Produce a short utterance (one sentence) to the {audience_name} to accomplish the goal, and include a very brief body language description.
Output in this format exactly:
DIALOGUE: <one sentence>
BODY: <brief body language phrase>
"""

    raw = self._model.sample_text(prompt).strip()
    m1 = re.search(r'DIALOGUE:\s*(.*)', raw)
    m2 = re.search(r'BODY:\s*(.*)', raw)
    text = m1.group(1).strip() if m1 else raw
    body = m2.group(1).strip() if m2 else ''

    # Store utterance
    memory.add_utterance(current_turn, self.get_entity().name, text, body)

    # Return formatted action
    return f'DIALOGUE: {text}\nBODY: {body}'

  def get_state(self) -> entity_component.ComponentState:
    """Get component state."""
    return {}

  def set_state(self, state: entity_component.ComponentState) -> None:
    """Set component state."""
    pass
