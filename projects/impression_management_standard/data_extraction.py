"""Data extraction from simulation entities."""

import datetime

from concordia.components.agent import impression_management_pe as impe
from concordia.prefabs.simulation import generic as simulation

from projects.impression_management_standard.models import TurnLog


def extract_turn_data_from_entities(
    sim: simulation.Simulation,
    actor_name: str,
    audience_name: str,
    total_turns: int,
    use_trait_paragraph: bool = False,
) -> tuple[list[TurnLog], str | None, str | None]:
    """Extract turn data from simulation entities.

    Args:
        sim: The simulation object.
        actor_name: Name of actor entity.
        audience_name: Name of audience entity.
        total_turns: Total number of turns.
        use_trait_paragraph: If True, also extract actor/audience trait paragraphs from
            PersonalityTraitsComponent when present.

    Returns:
        Tuple of (list of TurnLog entries, actor_traits paragraph or None, audience_traits paragraph or None).
    """
    turn_logs = []
    actor_traits: str | None = None
    audience_traits: str | None = None

    # Get entities from simulation
    entities = sim.get_entities()
    actor_entity = next((e for e in entities if e.name == actor_name), None)
    audience_entity = next((e for e in entities if e.name == audience_name), None)

    if not actor_entity or not audience_entity:
        return turn_logs, actor_traits, audience_traits

    # Get components
    actor_memory = actor_entity.get_component(
        impe.DEFAULT_IMPE_MEMORY_COMPONENT_KEY,
        type_=impe.IMPEMemoryComponent
    )
    audience_memory = audience_entity.get_component(
        impe.DEFAULT_IMPE_MEMORY_COMPONENT_KEY,
        type_=impe.IMPEMemoryComponent
    )

    # Get conversation history - use a large k to get all turns
    # Use total_turns * 2 to ensure we get all entries (actor + audience utterances)
    # Add extra buffer to handle any edge cases
    conversation = actor_memory.get_recent_conversation(k=total_turns * 2 + 10)
    evaluations = audience_memory.get_recent_evaluations(k=total_turns + 10)
    pf_history = actor_memory.get_pf_history(k=total_turns + 10)
    reflections = actor_memory.get_recent_reflections(k=total_turns + 10)
    pe_history = actor_memory.get_recent_pe_history(k=total_turns + 10)

    # Debug: print what we found
    print(f"\nDebug: Found {len(conversation)} conversation entries")
    for u in conversation:
        print(f"  Conversation: turn={u.turn}, speaker={u.speaker}")
    print(f"Debug: Found {len(evaluations)} evaluations")
    for e in evaluations:
        print(f"  Evaluation: turn={e.turn}, I_t={e.I_t:.2f}")
    print(f"Debug: Found {len(pf_history)} PF history entries")
    for p in pf_history:
        print(f"  PF: turn={p.get('turn')}, I_hat={p.get('I_hat', 0):.2f}")
    print(f"Debug: Found {len(pe_history)} PE records")
    for p in pe_history:
        print(f"  PE: turn={p.turn}, pe={p.pe:.2f}")

    # Match data by turn
    # For turn 1, we may not have PF history or PE records (they start from turn 2)
    for turn in range(1, total_turns + 1):
        # Find actor utterance
        actor_utt = next(
            (u for u in conversation if u.turn == turn and u.speaker == actor_name),
            None
        )
        if not actor_utt:
            print(f"Warning: No actor utterance found for turn {turn}")
            continue

        # Find audience evaluation
        # Try exact match first, then try to match by position (evaluations should be in order)
        eval_rec = next((e for e in evaluations if e.turn == turn), None)
        if not eval_rec:
            # Fallback: try to match by index (first evaluation = turn 1, etc.)
            eval_index = turn - 1
            if 0 <= eval_index < len(evaluations):
                eval_rec = evaluations[eval_index]
                print(f"Warning: Turn {turn} evaluation matched by index (stored as turn {eval_rec.turn})")
            else:
                print(f"Warning: No audience evaluation found for turn {turn}")
                continue

        # Find PF history entry (may not exist for turn 1)
        pf_entry = next((p for p in pf_history if p.get('turn') == turn), None)
        if not pf_entry and turn == 1:
            # Turn 1: No PF update yet, use default
            I_hat = 0.5
            ess = 0.0
        elif pf_entry:
            I_hat = pf_entry.get('I_hat', 0.5)
            ess = pf_entry.get('ess', 0.0)
        else:
            # For later turns, try to use the most recent PF entry
            if pf_history:
                pf_entry = pf_history[-1]
                I_hat = pf_entry.get('I_hat', 0.5)
                ess = pf_entry.get('ess', 0.0)
            else:
                I_hat = 0.5
                ess = 0.0

        # Find PE record (may not exist for turn 1)
        pe_rec = next((p for p in pe_history if p.turn == turn), None)
        if not pe_rec:
            actor_pe = 0.0  # No PE for turn 1 (no previous I_hat to compare)
        else:
            actor_pe = pe_rec.pe  # Keep signed PE (positive = underestimating, negative = overestimating)

        # Find reflection (may not exist for turn 1)
        refl = next((r for r in reflections if r.turn == turn), None)
        reflection_text = refl.text if refl else ''

        turn_logs.append(TurnLog(
            time=datetime.datetime.now().isoformat(timespec='seconds') + 'Z',
            turn=turn,
            speaker=actor_name,
            listener=audience_name,
            speaker_text=actor_utt.text,
            speaker_body=actor_utt.body,
            audience_I=eval_rec.I_t,
            audience_text=eval_rec.utterance.text,
            audience_body=eval_rec.utterance.body,
            actor_I_hat=I_hat,
            actor_pe=actor_pe,
            reflection_text=reflection_text,
            ess=ess,
        ))

    # Optionally extract trait paragraphs for JSON output
    if use_trait_paragraph:
        try:
            actor_traits_comp = actor_entity.get_component(
                impe.DEFAULT_PERSONALITY_TRAITS_COMPONENT_KEY,
                type_=impe.PersonalityTraitsComponent,
            )
            if actor_traits_comp is not None:
                actor_traits = actor_traits_comp.get_trait_paragraph()
        except (KeyError, TypeError, AttributeError):
            pass
        try:
            audience_traits_comp = audience_entity.get_component(
                impe.DEFAULT_PERSONALITY_TRAITS_COMPONENT_KEY,
                type_=impe.PersonalityTraitsComponent,
            )
            if audience_traits_comp is not None:
                audience_traits = audience_traits_comp.get_trait_paragraph()
        except (KeyError, TypeError, AttributeError):
            pass

    return turn_logs, actor_traits, audience_traits
