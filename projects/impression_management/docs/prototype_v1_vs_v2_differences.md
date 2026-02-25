# Differences: pe_conversation_prototype.py vs pe_conversation_prototype_v2.py

This document summarizes the differences between the two prototype scripts in `projects/impression_management/`.

---

## 1. **Memory check (conversation summary)**

**v2 only:** Adds a `memory_check()` method on `Agent` that:

- Summarizes the **full conversation so far** in one paragraph via an extra LLM call.
- Returns that summary string.

**Usage in v2:** The summary is injected into prompts in three places (see below). The original prototype does not have this method or any global conversation summary.

---

## 2. **Where the memory summary is used (v2)**

| Location | v1 (original) | v2 |
|----------|----------------|-----|
| **`audience_evaluate_and_respond()`** | Prompt includes only “Recent conversation (last k):” + formatted recent turns. | Same, **plus** “Full conversation summary (all turns so far):” + `self.memory_check()`. |
| **`act()`** | Prompt includes only goal/ideal and “Produce a short utterance…”. | Same, **plus** “Recent conversation (last k):” and “Full conversation summary (all turns so far):” + `self.memory_check()`. |
| **`act_based_on_belief()`** | Prompt includes recent conversation, I_hat history, and “Produce a short utterance…”. | Same, **plus** “Full conversation summary (all turns so far):” + `self.memory_check()`. |

So in v2, every time the audience evaluates/responds or the actor acts, the model also sees an LLM-generated summary of the entire conversation so far, in addition to the raw recent turns.

---

## 3. **`act()` first-turn prompt (v1 vs v2)**

- **v1:** No conversation in the prompt for turn 1 (no prior conversation). Prompt is only goal + ideal + “Produce a short utterance…”.
- **v2:** For turn 1, v2 still calls `memory_check()` and `recent_conversation()`. For turn 1 both are empty/minimal, so v2 adds “Recent conversation” and “Full conversation summary” sections even on the first turn (redundant but consistent with later turns).

---

## 4. **Run loop: actor conversation memory (potential bug in v2)**

- **v1:** After the audience responds, the script appends the audience’s reply to the **actor’s** conversation memory:
  - `actor.memory.conversation.append(Utterance(turn=t, actor=audience.name, text=audience_reply.text, body=audience_reply.body))`
- **v2:** This line is **missing**. Only `audience.memory.conversation.append(actor_utt)` is present.

So in v2, the actor’s `memory.conversation` never gets the audience’s utterances. That can make:

- `actor.recent_conversation()` and `actor.format_conversation(conv_k)` incomplete for the actor.
- `actor.memory_check()` (which uses `self.memory.conversation`) summarize only the actor’s own turns, not the audience’s.

If that’s unintended, v2 should add the same `actor.memory.conversation.append(...)` after the audience responds, as in v1.

---

## 5. **Everything else**

- **Data structures** (e.g. `Goal`, `PERecord`, `Utterance`, `AgentMemory`, `ParticleFilter`, `CulturalNorm`, `PersonalityTrait`, `TurnLog`), **CLI**, **plotting**, **save_json**, and the rest of the turn loop (who acts when, PF update, PE, logging) are the same in both files.
- **Line counts:** v2 is longer (~1223 vs ~1188) mainly due to `memory_check()` and the extra prompt paragraphs that include the full conversation summary.

---

## Summary table

| Aspect | pe_conversation_prototype.py (v1) | pe_conversation_prototype_v2.py (v2) |
|--------|-----------------------------------|--------------------------------------|
| `Agent.memory_check()` | No | Yes – LLM summarizes full conversation |
| Audience response prompt | Recent turns only | Recent turns + full conversation summary |
| Actor `act()` prompt | Goal + ideal only (turn 1) | + recent conv + full summary (even turn 1) |
| Actor `act_based_on_belief()` prompt | Recent conv + I_hat history | + full conversation summary |
| Append audience reply to actor’s conversation | Yes | **No** (likely bug) |

So the **main behavioral difference** is that v2 gives the model a **global “memory”** of the conversation (via `memory_check()`) in addition to the recent turns; the **main risk** in v2 is the missing append to `actor.memory.conversation`, which can make the actor’s view of the dialogue incomplete and the v2 “full conversation summary” one-sided.
