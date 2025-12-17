"""Conversation execution functions."""

from concordia.components.agent import impression_management_pe as impe

from projects.impression_management.config import ConversationConfig
from projects.impression_management import utils


def get_conversation_components(actor, audience):
    """Extract conversation components from entities."""
    actor_memory = actor.get_component(
        impe.DEFAULT_IMPE_MEMORY_COMPONENT_KEY,
        type_=impe.IMPEMemoryComponent
    )
    audience_memory = audience.get_component(
        impe.DEFAULT_IMPE_MEMORY_COMPONENT_KEY,
        type_=impe.IMPEMemoryComponent
    )
    audience_eval = audience.get_component(
        impe.DEFAULT_IMPE_AUDIENCE_EVALUATION_COMPONENT_KEY,
        type_=impe.IMPEAudienceEvaluationComponent
    )
    actor_pf = actor.get_component(
        impe.DEFAULT_IMPE_ACTOR_PARTICLE_FILTER_COMPONENT_KEY,
        type_=impe.IMPEActorParticleFilterComponent
    )
    actor_reflection = actor.get_component(
        impe.DEFAULT_IMPE_REFLECTION_COMPONENT_KEY,
        type_=impe.IMPEReflectionComponent
    )

    return actor_memory, audience_memory, audience_eval, actor_pf, actor_reflection


def run_conversation_turn(
    turn: int,
    actor,
    audience,
    actor_memory,
    audience_memory,
    audience_eval,
    actor_pf,
    actor_reflection,
):
    """Execute a single conversation turn."""
    print(f"\n--- Turn {turn} ---")

    # 1. Actor acts
    print("Actor acting...")
    actor_action = actor.act()

    # Parse actor action
    actor_text, actor_body = utils.parse_dialogue_and_body(actor_action)
    print(f"Actor: {actor_text[:80]}...")

    # 2. Audience evaluates
    print("Audience evaluating...")
    observation = f'Actor said: "{actor_text}"\nBody language: "{actor_body}"'
    audience_eval.pre_observe(observation)
    audience_eval.post_observe()

    evaluations = audience_memory.get_recent_evaluations()
    if evaluations:
        I_t = evaluations[-1].I_t
        print(f"Audience evaluation I_t = {I_t:.2f}")

    # 3. Actor updates particle filter
    if evaluations:
        print("Actor updating particle filter...")
        audience_utt = evaluations[-1].utterance
        audience_obs = f'Audience said: "{audience_utt.text}"\nBody language: "{audience_utt.body}"'
        actor_pf.pre_observe(audience_obs)
        actor_pf.post_observe()

        pf_history = actor_memory.get_pf_history()
        if pf_history:
            I_hat = pf_history[-1]['I_hat']
            print(f"Actor belief I_hat = {I_hat:.2f}")

    # 4. Actor reflects
    print("Actor reflecting...")
    actor_reflection.post_observe()


def run_conversation(
    config: ConversationConfig,
    actor,
    audience,
):
    """Run the full conversation."""
    print(f"\nRunning {config.turns} turn conversation...")

    actor_memory, audience_memory, audience_eval, actor_pf, actor_reflection = (
        get_conversation_components(actor, audience)
    )

    for turn in range(1, config.turns + 1):
        run_conversation_turn(
            turn,
            actor,
            audience,
            actor_memory,
            audience_memory,
            audience_eval,
            actor_pf,
            actor_reflection,
        )
