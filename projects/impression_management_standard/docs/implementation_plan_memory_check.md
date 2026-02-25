# Implementation Plan: Memory Check (Conversation Summary) in Standard Version

This plan adds the prototype v2 "memory check" feature to the standard impression management flow: an **LLM-generated summary of the full conversation so far**, injected into the audience response prompt and the actor act prompts.

---

## 1. Goal

- **Behavior:** When enabled, the model sees a short paragraph summarizing the entire conversation (in addition to recent turns) when:
  1. The **audience** evaluates the actor and generates a reply (`IMPEAudienceEvaluationComponent.post_observe`).
  2. The **actor** produces an utterance (`IMPEActComponent.get_action_attempt`), both on the first turn and on subsequent turns.

- **Config:** Feature is **optional** and off by default, controlled by a flag (e.g. `use_memory_check` or `enable_memory_summary`).

- **Cost:** One extra LLM call per turn when enabled (shared via cache so audience and actor reuse the same summary for that turn).

---

## 2. Design Choices

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Where to compute the summary | Cached on **IMPEMemoryComponent** | Both audience and actor use the same memory; one summary per turn avoids duplicate LLM calls. |
| When to recompute | When conversation length changes | Summary is invalidated after each new utterance (actor or audience). |
| Who calls the LLM | Memory component given a **model reference** when requesting summary | Memory doesn’t own a model; standard pattern is to pass model from components that have it (audience eval, act). So: `memory.get_conversation_summary(model)` with model passed by caller. |
| Full conversation access | Add **get_full_conversation()** on IMPEMemoryComponent | Currently only `get_recent_conversation(k)` with default `k=recent_k`; we need full list for the summary prompt. |

---

## 3. Phases

### Phase 1: Memory – full conversation and summary API

**File:** `concordia/components/agent/impression_management_pe.py` (IMPEMemoryComponent)

1. **Add `get_full_conversation()`**
   - Return all utterances (thread-safe, same lock as `get_recent_conversation`).
   - Signature: `def get_full_conversation(self) -> list[Utterance]`.
   - Implementation: under `self._lock`, `return self._conversation.copy()`.

2. **Add optional conversation-summary cache on IMPEMemoryComponent**
   - Attributes: `_conversation_summary: str | None = None`, `_conversation_summary_length: int = 0`.
   - Purpose: cache the last summary and the conversation length it was computed for; recompute only when `len(self._conversation) != self._conversation_summary_length`.

3. **Add `get_conversation_summary(self, model, *, use_cache: bool = True) -> str`**
   - If `use_cache` and `len(self._conversation) == self._conversation_summary_length` and `_conversation_summary is not None`, return cached summary.
   - If conversation is empty, return a fixed string (e.g. `"No conversation has occurred yet."`).
   - Otherwise:
     - Get full conversation via `get_full_conversation()`.
     - Format it (e.g. same style as prototype v2: `"[t={turn} {actor}] DIALOGUE: {text} | BODY: {body}"` per line).
     - Build a prompt like: “Summarize the full conversation so far in one concise paragraph. Focus on: key points raised, tone progression, and current interaction dynamics. Do not invent details. Conversation transcript: …”
     - Call `model.sample_text(prompt)`, strip, store in `_conversation_summary`, set `_conversation_summary_length = len(self._conversation)`, return summary.
   - Thread-safety: perform length check and update of cache under the same lock used for conversation (or document that callers run in the same turn order so no cross-turn races).

4. **State serialization**
   - In `get_state` / `set_state`, include `_conversation_summary` and `_conversation_summary_length` if you need checkpointing; otherwise they can be omitted (summary is recomputed after load).

---

### Phase 2: Config and prefab flag

**Files:**  
`projects/impression_management_standard/config.py`,  
`projects/impression_management_standard/models.py`,  
`projects/impression_management_standard/simulation_config.py`,  
`projects/impression_management_standard/simple_audience_prefab.py`

1. **models.py (ConversationConfig or equivalent)**
   - Add a boolean, e.g. `use_memory_check: bool = False`.

2. **config.py**
   - Add CLI flag, e.g. `--use_memory_check` / `--no_memory_check` (default False).
   - Parse and pass into config object.

3. **simulation_config.py**
   - When building params for actor and audience, set `'use_memory_check': config.use_memory_check` (or the chosen key) in the params dict passed to the prefab.

4. **simple_audience_prefab.py**
   - Read `use_memory_check = bool(self.params.get('use_memory_check', False))`.
   - Pass it into:
     - `IMPEAudienceEvaluationComponent(..., use_memory_check=use_memory_check)` (Phase 3),
     - and into the actor’s `IMPEActComponent(..., use_memory_check=use_memory_check)` (Phase 4).

---

### Phase 3: Audience evaluation component

**File:** `concordia/components/agent/impression_management_pe.py` (IMPEAudienceEvaluationComponent)

1. **Constructor**
   - Add `use_memory_check: bool = False`.
   - Store as `self._use_memory_check`.

2. **In `post_observe()`**
   - After computing `conv_k = memory.get_recent_conversation()` and building `base_resp_instruction`:
     - If `self._use_memory_check`:
       - Call `memory_summary = memory.get_conversation_summary(self._model, use_cache=True)`.
       - Append to the response prompt block (before “Produce a short reply…”):
         - `"\n\nFull conversation summary (all turns so far):\n" + memory_summary + "\n\n"`.
     - If not enabled, keep current prompt unchanged.
   - Use the same `base_resp_instruction` for both the non–option-space and option-space branches so the summary appears in both.

---

### Phase 4: Actor act component

**File:** `concordia/components/agent/impression_management_pe.py` (IMPEActComponent)

1. **Constructor**
   - Add `use_memory_check: bool = False`.
   - Store as `self._use_memory_check`.

2. **First-turn branch (`if not pf_history`)**
   - If `self._use_memory_check`:
     - Call `memory_summary = memory.get_conversation_summary(self._model, use_cache=True)` (on first turn this will usually be “No conversation has occurred yet.” or empty; still consistent with v2).
     - Add to prompt: “Full conversation summary (all turns so far):” + memory_summary.
   - Else: no change.

3. **Subsequent-turn branch (`else`)**
   - If `self._use_memory_check`:
     - Call `memory_summary = memory.get_conversation_summary(self._model, use_cache=True)`.
     - Insert in the prompt after “Recent conversation (last {recent_k}):” and the formatted conversation, e.g.:
       - “Full conversation summary (all turns so far):” + memory_summary + “\n\n” + “Recent I_hat (belief) history:”.
   - Else: no change.

---

### Phase 5: Prefab wiring

**File:** `projects/impression_management_standard/simple_audience_prefab.py`

- In the place where **actor** components are created (actor prefab or shared prefab), ensure `IMPEActComponent` is instantiated with `use_memory_check=use_memory_check` (read from params as in Phase 2).
- Audience evaluation is already wired in Phase 2/3; double-check that both actor and audience receive the same `use_memory_check` value from params.

---

### Phase 6: Tests and manual check

1. **Unit / integration**
   - With `use_memory_check=False`, behavior unchanged (no extra LLM call, same prompts as today).
   - With `use_memory_check=True`, one extra LLM call per turn; audience and actor prompts contain “Full conversation summary” and the cached summary is reused (e.g. verify by logging or a small test that calls `get_conversation_summary` twice in the same turn and asserts one model call).

2. **Manual**
   - Run standard run with `--use_memory_check` (or the chosen flag) for 2–3 turns; confirm logs or outputs show the summary in prompts and that the run completes without errors.

---

## 4. Prompt text (reference)

Use the same instruction as in prototype v2 for consistency:

- **Summary request (used inside `get_conversation_summary`):**
  - “Summarize the full conversation so far in one concise paragraph. Focus on: key points raised, tone progression, and current interaction dynamics. Do not invent details not present in the transcript.”
- **Injection into prompts:**
  - “Full conversation summary (all turns so far):\n{memory_summary}”

Format of the transcript passed to the summary prompt should match what the prototype uses (e.g. one line per utterance with turn, actor, dialogue text, body).

---

## 5. Files to touch (summary)

| File | Changes |
|------|--------|
| `concordia/components/agent/impression_management_pe.py` | IMPEMemoryComponent: `get_full_conversation()`, cache fields, `get_conversation_summary(model)`; IMPEAudienceEvaluationComponent: `use_memory_check`, inject summary in `post_observe`; IMPEActComponent: `use_memory_check`, inject summary in first-turn and subsequent-turn prompts. |
| `projects/impression_management_standard/models.py` | Add `use_memory_check: bool = False`. |
| `projects/impression_management_standard/config.py` | Add CLI and pass-through for `use_memory_check`. |
| `projects/impression_management_standard/simulation_config.py` | Put `use_memory_check` into params for prefab. |
| `projects/impression_management_standard/simple_audience_prefab.py` | Read `use_memory_check` from params; pass into audience eval and actor act components. |

---

## 6. Optional follow-ups

- **Logging:** Log when a new summary is computed (e.g. turn number and length) for debugging and cost analysis.
- **Cap on conversation length:** If conversations get very long, optionally pass only the last N utterances into the summary prompt to bound token count and latency.
- **Documentation:** Add a short note in `docs/feature_comparison.md` or `docs/memory_and_history.md` that the standard version supports an optional “memory check” (conversation summary) matching prototype v2 behavior when `use_memory_check` is enabled.
