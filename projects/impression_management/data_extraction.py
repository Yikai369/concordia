"""Data extraction from entities."""

import datetime
from typing import Any

from concordia.components.agent import impression_management_pe as impe

from projects.impression_management.models import TurnLog


def extract_turn_data_from_entities(
    actor_entity,
    audience_entity,
    total_turns: int,
    turn_summaries: list[dict[str, str | int]] | None = None,
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

    # Get full conversation and per-turn history (avoid recent_k truncation)
    conversation = actor_memory.get_full_conversation()
    actions = actor_memory.get_recent_actions(total_turns)
    evaluations = audience_memory.get_recent_evaluations(total_turns)
    pf_history = actor_memory.get_pf_history(total_turns)
    reflections = actor_memory.get_recent_reflections(total_turns)
    pe_history = actor_memory.get_recent_pe_history(total_turns)

    actor_logs = actor_entity.get_all_logs()
    audience_logs = audience_entity.get_all_logs()

    def _option_entries(channel_data: list[Any]) -> list[dict[str, Any]]:
        out = []
        for datum in channel_data:
            if isinstance(datum, dict) and 'Options' in datum and 'Chosen Index' in datum:
                out.append(datum)
        return out

    actor_option_entries = _option_entries(
        actor_logs.get('__act__', [])
    )
    if not actor_option_entries:
        actor_option_entries = _option_entries(
            actor_logs.get('IMPE_Act_OptionSpace', [])
        )

    audience_option_entries = _option_entries(
        audience_logs.get(
            impe.DEFAULT_IMPE_AUDIENCE_EVALUATION_COMPONENT_KEY,
            [],
        )
    )

    actor_option_idx = 0
    audience_option_idx = 0
    summary_by_turn = {
        int(entry.get('turn', -1)): entry
        for entry in (turn_summaries or [])
    }

    # Build per-turn records by sequence index to avoid turn-id drift.
    for turn in range(1, total_turns + 1):
        idx = turn - 1

        # Prefer action history for actor utterance (one actor action per turn).
        actor_action = actions[idx] if idx < len(actions) else None
        actor_utt = next((u for u in conversation if u.turn == turn and u.actor == actor_entity.name), None)

        # Prefer sequence-aligned audience evaluation.
        eval_rec = evaluations[idx] if idx < len(evaluations) else None
        if eval_rec is None:
            eval_rec = next((e for e in evaluations if e.turn == turn), None)

        # Prefer sequence-aligned PF history.
        pf_entry = pf_history[idx] if idx < len(pf_history) else None
        if pf_entry is None:
            pf_entry = next((p for p in pf_history if p.get('turn') == turn), None)
        I_hat = pf_entry.get('I_hat', 0.5) if pf_entry else 0.5
        ess = pf_entry.get('ess', 0.0) if pf_entry else 0.0

        # Prefer sequence-aligned PE record.
        pe_rec = pe_history[idx] if idx < len(pe_history) else None
        if pe_rec is None:
            pe_rec = next((p for p in pe_history if p.turn == turn), None)
        actor_pe = abs(pe_rec.pe) if pe_rec else 0.0

        # Prefer sequence-aligned reflection.
        refl = reflections[idx] if idx < len(reflections) else None
        if refl is None:
            refl = next((r for r in reflections if r.turn == turn), None)
        reflection_text = refl.text if refl else ''

        actor_options: list[dict[str, str]] = []
        actor_chosen_index: int | None = None
        actor_chosen = ''
        if actor_option_idx < len(actor_option_entries):
            entry = actor_option_entries[actor_option_idx]
            actor_option_idx += 1
            actor_options = entry.get('Options', []) or []
            actor_chosen_index = entry.get('Chosen Index')
            actor_chosen = entry.get('Chosen', '')

        audience_options: list[dict[str, str]] = []
        audience_chosen_index: int | None = None
        audience_chosen = ''
        if audience_option_idx < len(audience_option_entries):
            entry = audience_option_entries[audience_option_idx]
            audience_option_idx += 1
            audience_options = entry.get('Options', []) or []
            audience_chosen_index = entry.get('Chosen Index')
            audience_chosen = entry.get('Chosen', '')

        turn_summary = summary_by_turn.get(turn, {})
        actor_interaction_summary = str(
            turn_summary.get('actor_summary', '')
        )
        audience_interaction_summary = str(
            turn_summary.get('audience_summary', '')
        )

        turn_logs.append(TurnLog(
            time=datetime.datetime.now().isoformat(timespec='seconds') + 'Z',
            turn=turn,
            actor=actor_entity.name,
            audience=audience_entity.name,
            actor_text=(actor_action.text if actor_action else (actor_utt.text if actor_utt else '')),
            actor_body=(actor_action.body if actor_action else (actor_utt.body if actor_utt else '')),
            audience_I=(eval_rec.I_t if eval_rec else 0.0),
            audience_text=(eval_rec.utterance.text if eval_rec else ''),
            audience_body=(eval_rec.utterance.body if eval_rec else ''),
            actor_I_hat=I_hat,
            actor_pe=actor_pe,
            reflection_text=reflection_text,
            ess=ess,
            actor_options=actor_options,
            actor_chosen_index=actor_chosen_index,
            actor_chosen=actor_chosen,
            audience_options=audience_options,
            audience_chosen_index=audience_chosen_index,
            audience_chosen=audience_chosen,
            actor_interaction_summary=actor_interaction_summary,
            audience_interaction_summary=audience_interaction_summary,
        ))

    return turn_logs
