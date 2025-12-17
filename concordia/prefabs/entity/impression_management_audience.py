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

"""A prefab for Impression Management PE audience entity."""

from collections.abc import Mapping
import dataclasses

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.components.agent import impression_management_pe as impe_components
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib


@dataclasses.dataclass
class Entity(prefab_lib.Prefab):
  """A prefab implementing an Impression Management PE audience entity."""

  description: str = (
      'An entity that evaluates actor performance and generates true hidden state I_t. '
      'Provides feedback based on cultural norms and personality traits.'
  )
  params: Mapping[str, str | float | bool] = dataclasses.field(
      default_factory=lambda: {
          'name': 'Jane',
          'goal_name': 'evaluate',
          'goal_description': (
              'Evaluate the interviewee\'s competence '
              '(0=not competent, 1=fully competent).'
          ),
          'goal_role': 'Product Manager',
          'goal_ideal': 1.0,
          'recent_k': 3,
          'context': True,
          'cultural_norms': None,
          'traits': None,
          'trait_scores': None,
      }
  )

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build an IMPE audience entity.

    Args:
      model: The language model to use.
      memory_bank: The memory bank to use (for standard observations).

    Returns:
      An entity agent with IMPE components.
    """
    entity_name = self.params.get('name', 'Jane')
    goal_name = self.params.get('goal_name', 'evaluate')
    goal_description = self.params.get(
        'goal_description',
        'Evaluate the interviewee\'s competence (0=not competent, 1=fully competent).',
    )
    goal_role = self.params.get('goal_role', 'Product Manager')
    goal_ideal = float(self.params.get('goal_ideal', 1.0))
    recent_k = int(self.params.get('recent_k', 3))
    context = bool(self.params.get('context', True))
    cultural_norms = self.params.get('cultural_norms')
    traits = self.params.get('traits')
    trait_scores = self.params.get('trait_scores')

    # Create goal
    goal = impe_components.Goal(
        name=goal_name,
        description=goal_description,
        role=goal_role,
        ideal=goal_ideal,
    )

    # IMPE Memory component
    impe_memory_key = impe_components.DEFAULT_IMPE_MEMORY_COMPONENT_KEY
    impe_memory = impe_components.IMPEMemoryComponent(
        goal=goal,
        recent_k=recent_k,
        pre_act_label='\nIMPE Memory',
    )

    # Cultural Norms component (optional)
    cultural_norms_key = None
    if cultural_norms:
      cultural_norms_key = impe_components.DEFAULT_CULTURAL_NORMS_COMPONENT_KEY
      cultural_norms_comp = impe_components.CulturalNormsComponent(
          norms=cultural_norms,
          pre_act_label='\nCultural Norms',
      )
      # Initialize norms (one-time setup)
      cultural_norms_comp.initialize_norms(model, entity_name)

    # Personality Traits component (optional)
    personality_traits_key = None
    if traits:
      personality_traits_key = impe_components.DEFAULT_PERSONALITY_TRAITS_COMPONENT_KEY
      personality_traits_comp = impe_components.PersonalityTraitsComponent(
          traits=traits,
          trait_scores=trait_scores or {},
          pre_act_label='\nPersonality Traits',
      )

    # Audience Evaluation component
    audience_eval_key = impe_components.DEFAULT_IMPE_AUDIENCE_EVALUATION_COMPONENT_KEY
    audience_eval = impe_components.IMPEAudienceEvaluationComponent(
        model=model,
        memory_component_key=impe_memory_key,
        cultural_norms_key=cultural_norms_key,
        personality_traits_key=personality_traits_key,
        context=context,
        pre_act_label='\nAudience Evaluation',
    )

    # Standard memory component (for general observations)
    memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)

    # Standard observation to memory
    observation_to_memory_key = 'ObservationToMemory'
    observation_to_memory = agent_components.observation.ObservationToMemory(
        memory_component_key=memory_key,
    )

    # Assemble components
    components_of_agent = {
        memory_key: memory,
        impe_memory_key: impe_memory,
        audience_eval_key: audience_eval,
        observation_to_memory_key: observation_to_memory,
    }

    if cultural_norms_key:
      components_of_agent[cultural_norms_key] = cultural_norms_comp
    if personality_traits_key:
      components_of_agent[personality_traits_key] = personality_traits_comp

    agent = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=entity_name,
        act_component=agent_components.constant.Constant(state=''),  # Audience doesn't act
        context_components=components_of_agent,
    )

    return agent
