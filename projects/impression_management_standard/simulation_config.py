"""Create Config object for standard simulation loop."""

import random
from typing import Any

from concordia.components.agent.pe_conversation import Goal
from concordia.prefabs import entity as entity_prefabs
from concordia.prefabs import game_master as gm_prefabs
from concordia.typing import prefab as prefab_lib
from concordia.utils import helper_functions

from projects.impression_management_standard import constants
from projects.impression_management_standard.config import ConversationConfig
from projects.impression_management_standard import utils
from projects.impression_management_standard import simple_audience_prefab

from concordia.prefabs.entity import impression_management_actor
from concordia.prefabs.game_master import impression_management_pe as impe_gm


def create_simulation_config(
    config: ConversationConfig,
    rng: random.Random,
    traits_override: list[Any] | None = None,
) -> prefab_lib.Config:
    """Create Config object with prefabs and instances.

    Args:
        config: Conversation configuration.
        rng: Random number generator.
        traits_override: Optional list of PersonalityTrait to use instead of
            constants.ALL_TRAITS. When None, use None if config.no_traits
            else constants.ALL_TRAITS.

    Returns:
        Config object for simulation.
    """
    # Load prefabs
    prefabs = {
        **helper_functions.get_package_classes(entity_prefabs),
        **helper_functions.get_package_classes(gm_prefabs),
        # Add IMPE-specific prefabs
        'impression_management_actor__Entity': impression_management_actor.Entity(),
        'simple_audience__Entity': simple_audience_prefab.Entity(),
        'impression_management_pe__GameMaster': impe_gm.GameMaster(),
    }

    # Prepare traits and norms
    cultural_norms = None if config.no_audience_norms else constants.ALL_CULTURAL_NORMS
    if traits_override is not None:
        traits = traits_override
    else:
        traits = None if config.no_traits else constants.ALL_TRAITS

    trait_scores_actor = {}
    trait_scores_audience = {}
    if traits:
        trait_scores_actor = utils.generate_trait_scores(rng, traits, is_audience=False)
        trait_scores_audience = utils.generate_trait_scores(rng, traits, is_audience=True)

    # Resolve interview role from preset; optionally append question/experience banks
    goal_role_actor = None
    goal_role_audience = None
    if not config.no_context:
        base_role = constants.INTERVIEW_ROLE_PRESETS.get(
            config.interview_role_preset
        ) or constants.DEFAULT_INTERVIEW_ROLE
        question_bank = [] if config.no_question_bank else constants.INTERVIEW_QUESTION_BANKS.get(
            config.interview_role_preset, []
        )
        experience_bank = [] if config.no_experience_bank else constants.INTERVIEW_EXPERIENCE_BANKS.get(
            config.interview_role_preset, []
        )
        if experience_bank:
            goal_role_actor = base_role + "\n\nYou can draw on experiences such as:\n" + "\n".join(
                "- " + ex for ex in experience_bank
            )
        else:
            goal_role_actor = base_role
        if question_bank:
            goal_role_audience = base_role + "\n\nYou can ask questions such as:\n" + "\n".join(
                "- " + q for q in question_bank
            )
        else:
            goal_role_audience = base_role

    # Create goal objects for params
    goal_actor_name = 'competence'
    goal_actor_description = (
        'Be perceived as competent by the interviewer '
        '(0=not competent, 1=fully competent). Aim for 1.0.'
    )
    goal_audience_name = 'evaluate'
    goal_audience_description = (
        'Evaluate the interviewee\'s competence '
        '(0=not competent, 1=fully competent).'
    )

    # Create instances
    instances = [
        # Actor entity
        prefab_lib.InstanceConfig(
            prefab='impression_management_actor__Entity',
            role=prefab_lib.Role.ENTITY,
            params={
                'name': config.actor_name,
                'goal_name': goal_actor_name,
                'goal_description': goal_actor_description,
                'goal_role': goal_role_actor,
                'goal_ideal': 1.0,
                'recent_k': config.window,
                'num_particles': constants.DEFAULT_NUM_PARTICLES,
                'process_sigma': constants.DEFAULT_PROCESS_SIGMA,
                'obs_sigma': constants.DEFAULT_OBS_SIGMA,
                'context': not config.no_context,
                'cultural_norms': cultural_norms if config.actor_has_norms else None,
                'traits': traits,
                'trait_scores': trait_scores_actor,
                'enable_self_assessment': config.enable_self_assessment,
                'consistency_threshold': config.consistency_threshold,
                'disable_revision': config.disable_revision,
                'enable_instructions': not config.no_instructions,
                'enable_self_perception': not config.no_self_perception,
                'enable_situation_perception': config.enable_situation_perception,
                'enable_person_by_situation': config.enable_person_by_situation,
                'enable_world_building': not config.no_world_building,
                'enable_interview_context': not config.no_interview_context,
                'use_trait_paragraph': config.use_trait_paragraph,
                'use_option_space': config.use_option_space,
                'use_full_2a25_world': config.use_full_2a25_world,
                'use_memory_check': config.use_memory_check,
            },
        ),
        # Audience entity (simple version for standard loop)
        prefab_lib.InstanceConfig(
            prefab='simple_audience__Entity',
            role=prefab_lib.Role.ENTITY,
            params={
                'name': config.audience_name,
                'goal_name': goal_audience_name,
                'goal_description': goal_audience_description,
                'goal_role': goal_role_audience,
                'goal_ideal': 1.0,
                'recent_k': config.window,
                'context': not config.no_context,
                'cultural_norms': cultural_norms,
                'traits': traits,
                'trait_scores': trait_scores_audience,
                'enable_self_assessment': config.enable_self_assessment,
                'consistency_threshold': config.consistency_threshold,
                'disable_revision': config.disable_revision,
                'enable_instructions': not config.no_instructions,
                'enable_self_perception': not config.no_self_perception,
                'enable_situation_perception': config.enable_situation_perception,
                'enable_person_by_situation': config.enable_person_by_situation,
                'enable_world_building': not config.no_world_building,
                'enable_interview_context': not config.no_interview_context,
                'use_trait_paragraph': config.use_trait_paragraph,
                'use_option_space': config.use_option_space,
                'use_full_2a25_world': config.use_full_2a25_world,
                'use_memory_check': config.use_memory_check,
            },
        ),
        # Game master
        prefab_lib.InstanceConfig(
            prefab='impression_management_pe__GameMaster',
            role=prefab_lib.Role.GAME_MASTER,
            params={
                'name': 'IMPE Conversation Rules',
                'next_game_master_name': 'default rules',
                'can_terminate_simulation': True,
                'actor_name': config.actor_name,
                'audience_name': config.audience_name,
            },
        ),
    ]

    # Create config
    sim_config = prefab_lib.Config(
        default_premise=(
            'An interview conversation where the interviewee (actor) adapts '
            'their responses based on particle filter belief tracking of the '
            'interviewer\'s (audience) evaluation.'
        ),
        # Each logical turn = 4 steps (act, evaluate, PF update, reflect)
        # But we'll use more steps to allow for proper coordination
        # Each turn: actor acts, audience observes and acts
        # Actor's PF update and reflection happen automatically on observe
        default_max_steps=config.turns * 2,
        prefabs=prefabs,
        instances=instances,
    )

    return sim_config
