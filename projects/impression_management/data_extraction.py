"""Data extraction from entities."""

import datetime

from concordia.components.agent import impression_management_pe as impe

from projects.impression_management.models import TurnLog


def extract_turn_data_from_entities(
    actor_entity,
    audience_entity,
    total_turns: int,
) -> list[TurnLog]:
    """Extract turn data from entity components."""
    turn_logs = []

    actor_memory = actor_entity.get_component(
        impe.DEFAULT_IMPE_MEMORY_COMPONENT_KEY,
        type_=impe.IMPEMemoryComponent
    )
    audience_memory = audience_entity.get_component(
        impe.DEFAULT_IMPE_MEMORY_COMPONENT_KEY,
        type_=impe.IMPEMemoryComponent
    )

    # Get conversation history
    conversation = actor_memory.get_recent_conversation()
    evaluations = audience_memory.get_recent_evaluations()
    pf_history = actor_memory.get_pf_history()
    reflections = actor_memory.get_recent_reflections()
    pe_history = actor_memory.get_recent_pe_history()

    # Match data by turn
    for turn in range(1, total_turns + 1):
        # Find actor utterance
        actor_utt = next((u for u in conversation if u.turn == turn and u.speaker == actor_entity.name), None)
        if not actor_utt:
            continue

        # Find audience evaluation
        eval_rec = next((e for e in evaluations if e.turn == turn), None)
        if not eval_rec:
            continue

        # Find PF history entry
        pf_entry = next((p for p in pf_history if p.get('turn') == turn), None)
        I_hat = pf_entry.get('I_hat', 0.5) if pf_entry else 0.5
        ess = pf_entry.get('ess', 0.0) if pf_entry else 0.0

        # Find PE record
        pe_rec = next((p for p in pe_history if p.turn == turn), None)
        actor_pe = abs(pe_rec.pe) if pe_rec else 0.0

        # Find reflection
        refl = next((r for r in reflections if r.turn == turn), None)
        reflection_text = refl.text if refl else ''

        turn_logs.append(TurnLog(
            time=datetime.datetime.now().isoformat(timespec='seconds') + 'Z',
            turn=turn,
            speaker=actor_entity.name,
            listener=audience_entity.name,
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

    return turn_logs
