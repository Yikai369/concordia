"""Simple audience prefab for standard simulation loop."""

from collections.abc import Mapping
import dataclasses

from concordia.agents import entity_agent_with_logging
from concordia.associative_memory import basic_associative_memory
from concordia.components import agent as agent_components
from concordia.components.agent import impression_management_pe as impe_components
from concordia.components.agent import instructions
from concordia.components.agent import question_of_recent_memories
from concordia.language_model import language_model
from concordia.typing import prefab as prefab_lib

from projects.impression_management_standard import audience_act_component


@dataclasses.dataclass
class Entity(prefab_lib.Prefab):
  """A simple audience entity that evaluates on observe and acts simply."""

  description: str = (
      'A simple audience entity that evaluates actor performance on observation '
      'and generates responses based on memory and evaluation.'
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
          'enable_self_assessment': False,
          'consistency_threshold': 0.7,
          'disable_revision': False,
          'enable_instructions': True,
          'enable_self_perception': True,
          'enable_situation_perception': False,
          'enable_person_by_situation': False,
          'enable_world_building': True,
          'enable_interview_context': True,
      }
  )

  def build(
      self,
      model: language_model.LanguageModel,
      memory_bank: basic_associative_memory.AssociativeMemoryBank,
  ) -> entity_agent_with_logging.EntityAgentWithLogging:
    """Build a simple audience entity.

    The audience evaluates automatically on observe() and acts simply
    by returning the stored evaluation response.
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

    # World Context component (optional)
    world_context_key = None
    enable_world_building = bool(
        self.params.get('enable_world_building', True)
    )
    enable_interview_context = bool(
        self.params.get('enable_interview_context', True)
    )

    if enable_world_building or enable_interview_context:
      world_context_key = impe_components.DEFAULT_WORLD_CONTEXT_COMPONENT_KEY
      world_context_comp = impe_components.WorldContextComponent(
          enable_world_building=enable_world_building,
          enable_interview_context=enable_interview_context,
          pre_act_label='\nWorld Context',
      )

    # Audience Evaluation component (triggers on observe)
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

    # Simple act component (base - returns stored evaluation response)
    base_act_component = audience_act_component.SimpleAudienceActComponent(
        memory_component_key=impe_memory_key,
    )

    # Optionally wrap with self-assessment component
    enable_self_assessment = bool(
        self.params.get('enable_self_assessment', False)
    )
    consistency_threshold = float(
        self.params.get('consistency_threshold', 0.7)
    )
    enable_revision = not bool(self.params.get('disable_revision', False))

    if enable_self_assessment:
      act_component = impe_components.IMPESelfAssessmentComponent(
          base_act_component=base_act_component,
          model=model,
          memory_component_key=impe_memory_key,
          cultural_norms_key=cultural_norms_key,
          personality_traits_key=personality_traits_key,
          consistency_threshold=consistency_threshold,
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

    # 7. Audience evaluation component (existing)
    components_of_agent[audience_eval_key] = audience_eval

    # 8. World context (optional)
    if world_context_key:
      components_of_agent[world_context_key] = world_context_comp

    # 9. Cultural norms (existing, conditional)
    if cultural_norms_key:
      components_of_agent[cultural_norms_key] = cultural_norms_comp

    # 10. Personality traits (existing, conditional)
    if personality_traits_key:
      components_of_agent[personality_traits_key] = personality_traits_comp

    agent = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name=entity_name,
        act_component=act_component,
        context_components=components_of_agent,
    )

    return agent
