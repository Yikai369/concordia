"""Entity creation and configuration."""

from typing import Any

from concordia.components.agent import \
    impression_management_pe as impe_components
from concordia.components.agent.pe_conversation import Goal
from concordia.prefabs.entity import impression_management_actor
from concordia.prefabs.entity import impression_management_audience
from concordia.prefabs.game_master import impression_management_pe as impe_gm
import pandas as pd

from projects.impression_management import constants
from projects.impression_management.config import ConversationConfig

TEMP_ACTOR_TRAITS_PARAGRAPH = """This person navigates the social world with ease and adaptability, comfortably engaging with others and adjusting their behavior across a wide range of social situations. Rather than relying heavily on routines, they thrive in dynamic environments where expectations may shift and interactions are spontaneous. They naturally focus on the broader context of situations, easily grasping overarching concepts while still recalling relevant details when needed. Social interactions tend to feel intuitive and energizing. Reading non-verbal cues, understanding implied meanings, and participating in spontaneous conversations happen with little conscious effort.

They often enjoy social activities and feel comfortable in environments with varying levels of stimulation, adapting smoothly to busy gatherings or unfamiliar settings. Instead of relying primarily on logic or rehearsed scripts, they communicate fluidly and express themselves naturally in conversation. Their communication style is generally flexible and nuanced, allowing them to interpret subtleties in language and avoid common misunderstandings. They are able to express empathy and fairness openly, and these feelings are usually conveyed clearly to others.

Their interests tend to be diverse and socially integrated rather than intensely focused on narrow topics, helping them connect easily with different groups of people. Because their behavior aligns closely with common social expectations, forming relationships and integrating into social environments usually happens naturally. While they still value personal interests and individuality, they are comfortable balancing these with the social rhythms and spontaneity of everyday interactions."""


def extract_traits_from_spreadsheet(
    file_path: str,
) -> list[impe_components.PersonalityTrait]:
    """Extract personality traits from spreadsheet columns and non-empty cells."""
    if not file_path:
        return []

    df = pd.read_excel(file_path, header=0)
    traits: list[impe_components.PersonalityTrait] = []

    for survey in df.columns:
        series = df[survey].dropna()
        for assertion in series.astype(str):
            assertion = assertion.strip()
            if assertion:
                traits.append(
                    impe_components.PersonalityTrait(
                        name=survey,
                        assertion=assertion,
                    )
                )

    return traits


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
 ) -> tuple[list | None, bool, bool]:
    """Prepare cultural norms and trait enablement for actor/audience."""
    cultural_norms = None if config.no_audience_norms else constants.ALL_CULTURAL_NORMS

    if config.trait_mode == constants.TRAIT_MODE_ACTOR_ONLY:
        actor_has_traits = True
        audience_has_traits = False
    else:  # TRAIT_MODE_NONE
        actor_has_traits = False
        audience_has_traits = False

    return cultural_norms, actor_has_traits, audience_has_traits


def _traits_to_paragraph(model, agent_name: str, traits: list) -> str | None:
    """Convert trait assertions into one initialization paragraph."""
    if not traits:
        return None

    intro = (
        'Write a detailed paragraph describing this person based on '
        'statements about them. Consider how they would perceive, process, '
        'and interact with the social world. The statements are as follows:'
    )
    trait_list = '\n'.join(f'- {s.assertion}' for s in traits)
    prompt = f"""{intro}
    {trait_list}
    """
    traits_paragraph = (model.sample_text(prompt) or '').strip()

    return traits_paragraph


def create_entities(
    config: ConversationConfig,
    goal_actor: Goal,
    goal_audience: Goal,
    cultural_norms: list | None,
    actor_has_traits: bool,
    audience_has_traits: bool,
    model,
    memory_bank,
) -> tuple[Any, Any]:
    """Create and build actor and audience entities."""
    goal_role = constants.DEFAULT_INTERVIEW_ROLE if not config.no_context else None
    all_traits = extract_traits_from_spreadsheet(config.audience_traits_spreadsheet)

    actor_traits_paragraph = (
        TEMP_ACTOR_TRAITS_PARAGRAPH
        if actor_has_traits else None
    )
    if actor_traits_paragraph:
        print('\nActor traits paragraph:')
        print(actor_traits_paragraph)
    audience_traits_paragraph = (
        _traits_to_paragraph(model, config.audience_name, all_traits)
        if audience_has_traits else None
    )

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
        'cultural_norms': cultural_norms,
        'traits_paragraph': actor_traits_paragraph,
        'enable_world_building': True,
        'enable_interview_context': True,
        'use_full_2a25_world': True,
        'use_memory_check': config.use_memory_check,
        'enable_self_assessment': config.enable_self_assessment,
        'consistency_threshold': config.consistency_threshold,
        'disable_revision': config.disable_revision,
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
        'traits_paragraph': audience_traits_paragraph,
        'enable_world_building': True,
        'enable_interview_context': True,
        'use_full_2a25_world': True,
        'use_memory_check': config.use_memory_check,
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
