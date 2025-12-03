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

"""A prefab for PE (Prediction Error) conversation entity."""

from collections.abc import Mapping
import dataclasses

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.components.agent import pe_conversation as pe_components
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib


@dataclasses.dataclass
class Entity(prefab_lib.Prefab):
  """A prefab implementing a PE conversation entity."""

  description: str = (
      'An entity that adapts conversation based on prediction error (PE). '
      'Tracks goal achievement, estimates state, reflects, and adjusts behavior.'
  )
  params: Mapping[str, str | float] = dataclasses.field(
      default_factory=lambda: {
          'name': 'Agent A',
          'goal_name': 'likability',
          'goal_description': (
              'Be perceived as likable by the partner '
              '(0=not liked, 1=fully liked). Aim for 1.0.'
          ),
          'goal_ideal': 1.0,
          'recent_k': 3,
      }
  )

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build a PE conversation entity.

    Args:
      model: The language model to use.
      memory_bank: The memory bank to use (not used for PE memory, but for
        standard observations).

    Returns:
      An entity agent with PE conversation components.
    """
    entity_name = self.params.get('name', 'Agent A')
    goal_name = self.params.get('goal_name', 'likability')
    goal_description = self.params.get(
        'goal_description',
        'Be perceived as likable by the partner (0=not liked, 1=fully liked). '
        'Aim for 1.0.',
    )
    goal_ideal = float(self.params.get('goal_ideal', 1.0))
    recent_k = int(self.params.get('recent_k', 3))

    # Create goal
    goal = pe_components.Goal(
        name=goal_name,
        description=goal_description,
        ideal=goal_ideal,
    )

    # PE Memory component
    pe_memory_key = pe_components.DEFAULT_PE_MEMORY_COMPONENT_KEY
    pe_memory = pe_components.PEMemoryComponent(
        goal=goal,
        recent_k=recent_k,
        pre_act_label='\nPE Memory',
    )

    # PE Estimation component
    pe_estimation_key = pe_components.DEFAULT_PE_ESTIMATION_COMPONENT_KEY
    pe_estimation = pe_components.PEEstimationComponent(
        model=model,
        memory_component_key=pe_memory_key,
        pre_act_label='\nPE Estimation',
    )

    # PE Reflection component
    pe_reflection_key = pe_components.DEFAULT_PE_REFLECTION_COMPONENT_KEY
    pe_reflection = pe_components.PEReflectionComponent(
        model=model,
        memory_component_key=pe_memory_key,
        pre_act_label='\nPE Reflection',
    )

    # Standard memory component (for general observations)
    memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)

    # Standard observation to memory (for general observations)
    observation_to_memory_key = 'ObservationToMemory'
    observation_to_memory = agent_components.observation.ObservationToMemory(
        memory_component_key=memory_key,
    )

    # PE Act component
    act_component = pe_components.PEActComponent(
        model=model,
        memory_component_key=pe_memory_key,
    )

    # Assemble components
    components_of_agent = {
        memory_key: memory,
        pe_memory_key: pe_memory,
        pe_estimation_key: pe_estimation,
        pe_reflection_key: pe_reflection,
        observation_to_memory_key: observation_to_memory,
    }

    agent = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=entity_name,
        act_component=act_component,
        context_components=components_of_agent,
    )

    return agent
