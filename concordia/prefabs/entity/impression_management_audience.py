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
from concordia.components.agent import \
    impression_management_pe as impe_components
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib


@dataclasses.dataclass
class Entity(prefab_lib.Prefab):
  """A prefab implementing an Impression Management PE audience entity."""

  description: str = (
      "An entity that evaluates actor performance and generates true hidden state I_t. "
      "Provides feedback based on cultural norms and personality traits."
  )
  params: Mapping[str, str | float | bool] = dataclasses.field(
      default_factory=lambda: {
          "name": "Caden",
          "goal_name": "evaluate",
          "goal_description": "Evaluate the candidate's competence ",
          "goal_role": "Customer Service Agent",
          "recent_k": 3,
          "context": True,
          "cultural_norms": None,
          "traits_paragraph": None,
          "enable_world_building": True,
          "enable_interview_context": True,
          "use_full_2a25_world": True,
            "enable_self_assessment": True,
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
    entity_name = self.params.get("name", "Caden")
    goal_name = self.params.get("goal_name", "evaluate")
    goal_description = self.params.get(
        "goal_description",
        "Evaluate the candidate's competence.",
    )
    goal_role = self.params.get("goal_role", "Customer Service Agent")
    recent_k = int(self.params.get("recent_k", 3))
    context = bool(self.params.get("context", True))
    cultural_norms = self.params.get("cultural_norms")
    traits_paragraph = self.params.get("traits_paragraph")
    enable_world_building = bool(self.params.get("enable_world_building", True))
    enable_interview_context = bool(self.params.get("enable_interview_context", True))
    use_memory_check = bool(self.params.get("use_memory_check", False))
    use_full_2a25 = bool(self.params.get("use_full_2a25_world", True))

    goal = impe_components.Goal(
        name=goal_name,
        description=goal_description,
        role=goal_role,
    )

    impe_memory_key = impe_components.DEFAULT_IMPE_MEMORY_COMPONENT_KEY
    impe_memory = impe_components.IMPEMemoryComponent(
        goal=goal,
        recent_k=recent_k,
        pre_act_label="\nIMPE Memory",
    )

    cultural_norms_key = None
    cultural_norms_comp = None
    if cultural_norms:
      cultural_norms_key = impe_components.DEFAULT_CULTURAL_NORMS_COMPONENT_KEY
      cultural_norms_comp = impe_components.CulturalNormsComponent(
          norms=cultural_norms,
          pre_act_label="\nCultural Norms",
      )
      cultural_norms_comp.initialize_norms(model, entity_name)

    personality_traits_key = None
    personality_traits_comp = None
    if traits_paragraph:
      traits_for_component = [
          impe_components.PersonalityTrait(
              name="Profile",
              assertion=str(traits_paragraph),
          )
      ]
      personality_traits_key = impe_components.DEFAULT_PERSONALITY_TRAITS_COMPONENT_KEY
      personality_traits_comp = impe_components.PersonalityTraitsComponent(
          traits=traits_for_component,
          use_trait_paragraph=True,
          model=model,
          pre_act_label="\nPersonality Traits",
      )

    world_context_key = None
    world_context_comp = None
    if enable_world_building or enable_interview_context:
      world_context_key = impe_components.DEFAULT_WORLD_CONTEXT_COMPONENT_KEY
      world_context_comp = impe_components.WorldContextComponent(
          enable_world_building=enable_world_building,
          enable_interview_context=enable_interview_context,
          use_full_2a25=use_full_2a25,
          pre_act_label="\nWorld Context",
      )

    audience_eval_key = impe_components.DEFAULT_IMPE_AUDIENCE_EVALUATION_COMPONENT_KEY
    audience_eval = impe_components.IMPEAudienceEvaluationComponent(
        model=model,
        memory_component_key=impe_memory_key,
        cultural_norms_key=cultural_norms_key,
        personality_traits_key=personality_traits_key,
        context=context,
        use_memory_check=use_memory_check,
        pre_act_label="\nAudience Evaluation",
    )

    memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)

    observation_to_memory_key = "ObservationToMemory"
    observation_to_memory = agent_components.observation.ObservationToMemory(
        memory_component_key=memory_key,
    )

    components_of_agent = {
        memory_key: memory,
        impe_memory_key: impe_memory,
        audience_eval_key: audience_eval,
        observation_to_memory_key: observation_to_memory,
    }

    if cultural_norms_key and cultural_norms_comp is not None:
      components_of_agent[cultural_norms_key] = cultural_norms_comp
    if personality_traits_key and personality_traits_comp is not None:
      components_of_agent[personality_traits_key] = personality_traits_comp
    if world_context_key and world_context_comp is not None:
      components_of_agent[world_context_key] = world_context_comp

    base_act_component = impe_components.IMPEActComponent(
        model=model,
        memory_component_key=impe_memory_key,
        cultural_norms_key=cultural_norms_key,
        personality_traits_key=personality_traits_key,
        context=context,
        use_option_space=False,
        use_memory_check=use_memory_check,
    )

    enable_self_assessment = bool(self.params.get("enable_self_assessment", True))
    enable_revision = not bool(self.params.get("disable_revision", False))
    if enable_self_assessment:
      act_component = impe_components.IMPESelfAssessmentComponent(
          base_act_component=base_act_component,
          model=model,
          memory_component_key=impe_memory_key,
          cultural_norms_key=cultural_norms_key,
          personality_traits_key=personality_traits_key,
          enable_revision=enable_revision,
      )
    else:
      act_component = base_act_component

    agent = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=entity_name,
        act_component=act_component,
        context_components=components_of_agent,
    )

    return agent
