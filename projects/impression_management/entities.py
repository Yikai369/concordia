"""Entity creation and configuration."""

import random
from typing import Any

from concordia.components.agent.pe_conversation import Goal

from projects.impression_management import constants
from projects.impression_management.config import ConversationConfig
from projects.impression_management import utils
from concordia.prefabs.entity import impression_management_actor
from concordia.prefabs.entity import impression_management_audience
from concordia.prefabs.game_master import impression_management_pe as impe_gm


def create_goals(config: ConversationConfig) -> tuple[Goal, Goal]:
    """Create actor and audience goals."""
    goal_role = constants.DEFAULT_INTERVIEW_ROLE if not config.no_context else None

    goal_actor = Goal(
        name='competence',
        description=(
            'Be perceived as competent by the interviewer '
            '(0=not competent, 1=fully competent). Aim for 1.0.'
        ),
        role=goal_role,
        ideal=1.0,
    )

    goal_audience = Goal(
        name='evaluate',
        description=(
            'Evaluate the interviewee\'s competence '
            '(0=not competent, 1=fully competent).'
        ),
        role=goal_role,
        ideal=1.0,
    )

    return goal_actor, goal_audience


def prepare_traits_and_norms(
    config: ConversationConfig,
    rng: random.Random,
) -> tuple[dict[str, int], dict[str, int], list | None, list | None]:
    """Prepare trait scores and cultural norms."""
    cultural_norms = None if config.no_audience_norms else constants.ALL_CULTURAL_NORMS
    traits = None if config.no_traits else constants.ALL_TRAITS

    trait_scores_actor = {}
    trait_scores_audience = {}
    if traits:
        trait_scores_actor = utils.generate_trait_scores(rng, traits, is_audience=False)
        trait_scores_audience = utils.generate_trait_scores(rng, traits, is_audience=True)

    return trait_scores_actor, trait_scores_audience, cultural_norms, traits


def create_entities(
    config: ConversationConfig,
    goal_actor: Goal,
    goal_audience: Goal,
    trait_scores_actor: dict[str, int],
    trait_scores_audience: dict[str, int],
    cultural_norms: list | None,
    traits: list | None,
    model,
    memory_bank,
) -> tuple[Any, Any]:
    """Create and build actor and audience entities."""
    goal_role = constants.DEFAULT_INTERVIEW_ROLE if not config.no_context else None

    # Create actor prefab
    actor_prefab = impression_management_actor.Entity()
    actor_prefab.params = {
        'name': config.actor_name,
        'goal_name': goal_actor.name,
        'goal_description': goal_actor.description,
        'goal_role': goal_role,
        'goal_ideal': goal_actor.ideal,
        'recent_k': config.window,
        'num_particles': constants.DEFAULT_NUM_PARTICLES,
        'process_sigma': constants.DEFAULT_PROCESS_SIGMA,
        'obs_sigma': constants.DEFAULT_OBS_SIGMA,
        'context': not config.no_context,
        'cultural_norms': None,  # Actor doesn't have norms
        'traits': traits,
        'trait_scores': trait_scores_actor,
    }

    # Create audience prefab
    audience_prefab = impression_management_audience.Entity()
    audience_prefab.params = {
        'name': config.audience_name,
        'goal_name': goal_audience.name,
        'goal_description': goal_audience.description,
        'goal_role': goal_role,
        'goal_ideal': goal_audience.ideal,
        'recent_k': config.window,
        'context': not config.no_context,
        'cultural_norms': cultural_norms,
        'traits': traits,
        'trait_scores': trait_scores_audience,
    }

    # Build entities
    print("Building entities...")
    actor = actor_prefab.build(model, memory_bank)
    audience = audience_prefab.build(model, memory_bank)
    print("✓ Entities built")

    return actor, audience


def create_game_master(
    config: ConversationConfig,
    actor,
    audience,
    model,
    memory_bank,
):
    """Create and build game master."""
    gm_prefab = impe_gm.GameMaster()
    gm_prefab.params = {
        'name': 'IMPE Conversation Rules',
        'next_game_master_name': 'default rules',
        'can_terminate_simulation': False,  # We control termination
        'actor_name': config.actor_name,
        'audience_name': config.audience_name,
    }
    gm_prefab.entities = (actor, audience)
    return gm_prefab.build(model, memory_bank)
