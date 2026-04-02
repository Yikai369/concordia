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

"""A prefab for Impression Management PE actor entity."""

from collections.abc import Mapping
import dataclasses

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.components.agent import \
    impression_management_pe as impe_components
from concordia.components.agent import instructions
from concordia.components.agent import question_of_recent_memories
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib


@dataclasses.dataclass
class Entity(prefab_lib.Prefab):
  """A prefab implementing an Impression Management PE actor entity."""

  description: str = (
      'An entity that adapts conversation based on particle filter belief tracking. '
      'Uses particle filter to estimate audience evaluation and adjusts behavior accordingly.'
  )
  params: Mapping[str, str | float | bool] = dataclasses.field(
      default_factory=lambda: {
          'name': 'Riffer',
          'goal_name': 'competence',
          'goal_description': (
              'Be perceived as competent by the interviewer '
          ),
          'goal_role': 'Customer Service Agent',
          'recent_k': 3,
          'num_particles': 200,
          'process_sigma': 0.03,
          'obs_sigma': 0.08,
          'context': True,
          'cultural_norms': None,
          'traits_paragraph': None,
          'enable_instructions': True,
          'enable_self_perception': True,
          'enable_situation_perception': False,
          'enable_person_by_situation': False,
          'enable_world_building': True,
          'enable_interview_context': True,
          'enable_prior_impression': True,
          'enable_posterior_impression': True,
          'enable_feedback_interpretation': True,
          'use_option_space': True,
          'enable_self_assessment': True,
      }
  )
  _llm_model: language_model.LanguageModel | None = dataclasses.field(
      init=False,
      default=None,
      repr=False,
  )

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build an IMPE actor entity.

    Args:
      model: The language model to use.
      memory_bank: The memory bank to use (for standard observations).

    Returns:
      An entity agent with IMPE components.
    """
    entity_name = self.params.get('name', 'Riffer')
    goal_name = self.params.get('goal_name', 'competence')
    goal_description = self.params.get(
        'goal_description',
        'Be perceived as competent by the interviewer.',
    )
    goal_role = self.params.get('goal_role', 'Customer Service Agent')
    recent_k = int(self.params.get('recent_k', 3))
    num_particles = int(self.params.get('num_particles', 200))
    process_sigma = float(self.params.get('process_sigma', 0.03))
    obs_sigma = float(self.params.get('obs_sigma', 0.08))
    context = bool(self.params.get('context', True))
    cultural_norms = self.params.get('cultural_norms')
    traits_paragraph = self.params.get('traits_paragraph')
    self._llm_model = model

    # Create goal
    goal = impe_components.Goal(
        name=goal_name,
        description=goal_description,
        role=goal_role,
    )

    # Instructions component (role-playing context, optional)
    enable_instructions = bool(
        self.params.get('enable_instructions', True)  # Default: enabled
    )

    instructions_key = None
    instructions_comp = None
    if enable_instructions:
      instructions_key = 'Instructions'
      instructions_comp = instructions.Instructions(
          agent_name=entity_name,
          pre_act_label='\nRole playing instructions',
      )

    # SelfPerception component (optional, but recommended)
    enable_self_perception = bool(
        self.params.get('enable_self_perception', True)  # Default: enabled
    )

    self_perception_key = None
    self_perception_comp = None
    if enable_self_perception:
      self_perception_key = 'SelfPerception'
      self_perception_comp = question_of_recent_memories.SelfPerception(
          model=model,
          pre_act_label=f'\nQuestion: What kind of person is {entity_name}?\nAnswer',
      )

    # SituationPerception component (optional)
    enable_situation_perception = bool(
        self.params.get('enable_situation_perception', False)  # Default: disabled
    )

    situation_perception_key = None
    situation_perception_comp = None
    if enable_situation_perception:
      situation_perception_key = 'SituationPerception'
      situation_perception_comp = question_of_recent_memories.SituationPerception(
          model=model,
          pre_act_label=f'\nQuestion: What kind of situation is {entity_name} in right now?\nAnswer',
      )

    # PersonBySituation component (optional, requires SelfPerception and SituationPerception)
    enable_person_by_situation = bool(
        self.params.get('enable_person_by_situation', False)  # Default: disabled
    )

    person_by_situation_key = None
    person_by_situation_comp = None
    if enable_person_by_situation and self_perception_key and situation_perception_key:
      person_by_situation_key = 'PersonBySituation'
      person_by_situation_comp = question_of_recent_memories.PersonBySituation(
          model=model,
          components=[
              self_perception_key,
              situation_perception_key,
          ],
          pre_act_label=f'\nQuestion: What would a person like {entity_name} do in a situation like this?\nAnswer',
      )
    elif enable_person_by_situation:
      # Warn if dependencies not met
      import warnings
      warnings.warn(
          f"PersonBySituation requires both SelfPerception and SituationPerception. "
          f"Disabling PersonBySituation for {entity_name}.",
          UserWarning
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
    cultural_norms_comp = None
    if cultural_norms:
      cultural_norms_key = impe_components.DEFAULT_CULTURAL_NORMS_COMPONENT_KEY
      cultural_norms_comp = impe_components.CulturalNormsComponent(
          norms=cultural_norms,
          pre_act_label='\nCultural Norms',
      )

    # Personality Traits component (optional)
    personality_traits_key = None
    personality_traits_comp = None
    if traits_paragraph:

      traits_for_component = [
          impe_components.PersonalityTrait(
              name='Profile',
              assertion=str(traits_paragraph),
          )
      ]
      personality_traits_key = impe_components.DEFAULT_PERSONALITY_TRAITS_COMPONENT_KEY
      personality_traits_comp = impe_components.PersonalityTraitsComponent(
          traits=traits_for_component,
          use_trait_paragraph=True,
          model=model,
          pre_act_label='\nPersonality Traits',
      )

    # World Context component (optional)
    world_context_key = None
    world_context_comp = None
    enable_world_building = bool(
        self.params.get('enable_world_building', True)
    )
    enable_interview_context = bool(
        self.params.get('enable_interview_context', True)
    )

    use_full_2a25 = bool(self.params.get('use_full_2a25_world', True))
    if enable_world_building or enable_interview_context:
      world_context_key = impe_components.DEFAULT_WORLD_CONTEXT_COMPONENT_KEY
      world_context_comp = impe_components.WorldContextComponent(
          enable_world_building=enable_world_building,
          enable_interview_context=enable_interview_context,
          use_full_2a25=use_full_2a25,
          pre_act_label='\nWorld Context',
      )

    # Actor Particle Filter component
    actor_pf_key = impe_components.DEFAULT_IMPE_ACTOR_PARTICLE_FILTER_COMPONENT_KEY
    actor_pf = impe_components.IMPEActorParticleFilterComponent(
        model=model,
        memory_component_key=impe_memory_key,
        num_particles=num_particles,
        process_sigma=process_sigma,
        obs_sigma=obs_sigma,
        context=context,
        pre_act_label='\nActor Particle Filter',
    )

    # Reflection component
    reflection_key = impe_components.DEFAULT_IMPE_REFLECTION_COMPONENT_KEY
    reflection = impe_components.IMPEReflectionComponent(
        model=model,
        memory_component_key=impe_memory_key,
        cultural_norms_key=cultural_norms_key,
        personality_traits_key=personality_traits_key,
        context=context,
        pre_act_label='\nIMPE Reflection',
    )

    # Standard memory component (for general observations)
    memory_key = agent_components.memory.DEFAULT_MEMORY_COMPONENT_KEY
    memory = agent_components.memory.AssociativeMemory(memory_bank=memory_bank)

    # Standard observation to memory
    observation_to_memory_key = 'ObservationToMemory'
    observation_to_memory = agent_components.observation.ObservationToMemory(
        memory_component_key=memory_key,
    )

    # Prior impression component (optional)
    prior_impression_key = None
    prior_impression_comp = None
    if bool(self.params.get('enable_prior_impression', True)):
      prior_impression_key = (
        impe_components.DEFAULT_PRIOR_IMPRESSION_COMPONENT_KEY
      )
      prior_impression_comp = impe_components.PriorImpressionComponent(
        model=model,
        pre_act_label='\nPrior Impression',
      )

    # Posterior impression component (optional)
    posterior_impression_key = None
    posterior_impression_comp = None
    if bool(self.params.get('enable_posterior_impression', True)):
      posterior_impression_key = (
        impe_components.DEFAULT_POSTERIOR_IMPRESSION_COMPONENT_KEY
      )
      posterior_impression_comp = impe_components.PosteriorImpressionComponent(
        model=model,
        post_observe_label='\nPosterior Impression',
      )

    # Feedback interpretation component (optional)
    feedback_interpretation_key = None
    feedback_interpretation_comp = None
    if bool(self.params.get('enable_feedback_interpretation', True)):
      feedback_interpretation_key = (
        impe_components.DEFAULT_FEEDBACK_INTERPRETATION_COMPONENT_KEY
      )
      feedback_interpretation_comp = impe_components.FeedbackInterpretationComponent(
        model=model,
        post_observe_label='\nFeedback Interpretation',
      )

    # IMPE Act component (base)
    use_option_space = bool(self.params.get('use_option_space', True))
    use_memory_check = bool(self.params.get('use_memory_check', False))
    base_act_component = impe_components.IMPEActComponent(
        model=model,
        memory_component_key=impe_memory_key,
        cultural_norms_key=cultural_norms_key,
        personality_traits_key=personality_traits_key,
        context=context,
        use_option_space=use_option_space,
        use_memory_check=use_memory_check,
    )

    # Optionally wrap with self-assessment component
    enable_self_assessment = bool(
      self.params.get('enable_self_assessment', True)
    )
    enable_revision = not bool(self.params.get('disable_revision', False))  # Note: disable_revision is inverse

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

    # Assemble components in order (dependencies first)
    components_of_agent = {}

    # 1. Instructions (first - provides experimental context)
    if instructions_key:
      components_of_agent[instructions_key] = instructions_comp

    # 2. Self-perception (can use traits, norms, memories)
    if self_perception_key:
      components_of_agent[self_perception_key] = self_perception_comp

    # 3. Situation perception (uses observations and memories)
    if situation_perception_key:
      components_of_agent[situation_perception_key] = situation_perception_comp

    # 4. Person-by-situation (depends on self and situation perception)
    if person_by_situation_key:
      components_of_agent[person_by_situation_key] = person_by_situation_comp

    # 5. Memory components (required for perception components)
    components_of_agent[memory_key] = memory
    components_of_agent[impe_memory_key] = impe_memory

    # 6. Observation components (existing)
    components_of_agent[observation_to_memory_key] = observation_to_memory

    # 7. Prior impression (candidate self-report before each action)
    if prior_impression_key:
      components_of_agent[prior_impression_key] = prior_impression_comp

    # 8. Posterior impression (candidate self-report after interviewer responds, called by game_master)
    if posterior_impression_key:
      components_of_agent[posterior_impression_key] = posterior_impression_comp

    # 9. Feedback interpretation (candidate self-report after interviewer responds, called by game_master)
    if feedback_interpretation_key:
      components_of_agent[feedback_interpretation_key] = feedback_interpretation_comp

    # 10. Other IMPE components (existing)
    components_of_agent[actor_pf_key] = actor_pf
    components_of_agent[reflection_key] = reflection

    # 11. World context (optional)
    if world_context_key:
      components_of_agent[world_context_key] = world_context_comp

    # 12. Cultural norms (existing, conditional)
    if cultural_norms_key:
      components_of_agent[cultural_norms_key] = cultural_norms_comp

    # 13. Personality traits (existing, conditional)
    if personality_traits_key:
      components_of_agent[personality_traits_key] = personality_traits_comp

    agent = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=entity_name,
        act_component=act_component,
        context_components=components_of_agent,
    )

    return agent
