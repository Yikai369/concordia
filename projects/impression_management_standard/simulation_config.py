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
) -> prefab_lib.Config:
    """Create Config object with prefabs and instances.

    Args:
        config: Conversation configuration.
        rng: Random number generator.

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
    traits = None if config.no_traits else constants.ALL_TRAITS

    trait_scores_actor = {}
    trait_scores_audience = {}
    if traits:
        trait_scores_actor = utils.generate_trait_scores(rng, traits, is_audience=False)
        trait_scores_audience = utils.generate_trait_scores(rng, traits, is_audience=True)

    goal_role = constants.DEFAULT_INTERVIEW_ROLE if not config.no_context else None

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
                'goal_role': goal_role,
                'goal_ideal': 1.0,
                'recent_k': config.window,
                'num_particles': constants.DEFAULT_NUM_PARTICLES,
                'process_sigma': constants.DEFAULT_PROCESS_SIGMA,
                'obs_sigma': constants.DEFAULT_OBS_SIGMA,
                'context': not config.no_context,
                'cultural_norms': None,  # Actor doesn't have norms
                'traits': traits,
                'trait_scores': trait_scores_actor,
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
                'goal_role': goal_role,
                'goal_ideal': 1.0,
                'recent_k': config.window,
                'context': not config.no_context,
                'cultural_norms': cultural_norms,
                'traits': traits,
                'trait_scores': trait_scores_audience,
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
