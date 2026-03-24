# Plan: Log Four Options When Option Space Is Enabled

When `--use_option_space` is enabled, the system generates 4 response options and selects one. Currently **only the chosen response** appears in component logs. This plan adds logging of **all 4 options** and the **chosen index** to `component_logs.json` (when `--save_component_logs` is used).

---

## Scope

- **Actor**: `IMPEActComponent` (and wrapper `IMPESelfAssessmentComponent`) in `concordia/components/agent/impression_management_pe.py`.
- **Audience**: `IMPEAudienceEvaluationComponent` in the same file.
- **Prefab**: `concordia/prefabs/entity/impression_management_actor.py` (so the actor’s base act component gets a logging channel when wrapped).
- **Output**: Existing `save_component_logs()` in `projects/impression_management_standard/results.py` already serializes all channel entries; no change needed there.

---

## 1. Actor path: IMPEActComponent

**File:** `concordia/components/agent/impression_management_pe.py`

**Current behavior:**  
`IMPEActComponent` does not use `ComponentWithLogging`. When `_use_option_space` is True it generates 4 options, picks one, and returns the final action string. The 4 options are never logged.

**Changes:**

1. **Make IMPEActComponent support logging**
   - Add `entity_component.ComponentWithLogging` to the class bases of `IMPEActComponent` (so it has `_logging_channel` and `set_logging_channel`).
   - In the `_use_option_space` branch (after parsing `options` and computing `idx`), call:
     - `self._logging_channel({ 'Key': 'Option Space', 'Options': [{'dialogue': d, 'body': b} for (d,b) in options], 'Chosen Index': idx, 'Chosen': f'DIALOGUE: {text}\nBODY: {body}' })`
   - Use a structure that is JSON-serializable (list of dicts with `dialogue` and `body`).

2. **Ensure the base act component gets a channel when wrapped**
   - When the actor uses **self-assessment**, the **act** component passed to the entity is `IMPESelfAssessmentComponent`; the base `IMPEActComponent` is not in `context_components`, so it never receives `set_logging_channel`.
   - **File:** `concordia/prefabs/entity/impression_management_actor.py`
   - When `use_option_space` is True, add `base_act_component` to `components_of_agent` under a dedicated key (e.g. `'IMPE_Act_OptionSpace'`) **before** building the agent. Then the entity’s logging setup will call `set_logging_channel` on it, and its option-space log entries will appear under that channel in `component_logs.json`.
   - When self-assessment is disabled, `act_component` is `base_act_component`, so it already receives the `__act__` channel; no prefab change needed for that case.

**Result:**  
For the actor, when option space is enabled, each turn will have one entry (under `IMPE_Act_OptionSpace` when wrapped, or under `__act__` when not wrapped) containing the 4 options and the chosen index.

---

## 2. Audience path: IMPEAudienceEvaluationComponent

**File:** `concordia/components/agent/impression_management_pe.py`

**Current behavior:**  
When `_use_option_space` is True, the component generates 4 options, picks one (`idx`), then logs only:
`{'Key': ..., 'Value': f'Evaluated I_t: {I_t:.2f}, Response: "{dlg}"'}`.

**Changes:**

1. In the `_use_option_space` branch, **after** `idx` and `dlg, body` are set and **before** or **together with** the existing `_logging_channel` call, add the 4 options and chosen index to the logged payload.
2. Extend the existing log entry to include:
   - `'Options': [{'dialogue': d, 'body': b} for (d,b) in options]`
   - `'Chosen Index': idx`
   - Keep existing `'Key'` and `'Value'` (or fold the current `'Value'` into a richer structure so the chosen response and I_t remain clear).

**Result:**  
For the audience, when option space is enabled, the existing channel entry for that turn will include the full list of 4 options and the chosen index, in addition to the evaluated response.

---

## 3. Log format (recommended)

Use a consistent, JSON-friendly shape so downstream analysis and `component_logs.json` stay simple:

```json
{
  "Key": "Option Space",
  "Options": [
    {"dialogue": "...", "body": "..."},
    {"dialogue": "...", "body": "..."},
    {"dialogue": "...", "body": "..."},
    {"dialogue": "...", "body": "..."}
  ],
  "Chosen Index": 1,
  "Chosen": "DIALOGUE: ...\nBODY: ..."
}
```

For the audience, the same `Options` and `Chosen Index` can be added alongside the existing `Key` and `Value` (e.g. `Evaluated I_t: 0.80, Response: "..."`).

---

## 4. Testing / verification

- Run a short simulation with `--use_option_space` and `--save_component_logs`.
- Inspect `component_logs.json`:
  - **Actor:** Under `IMPE_Act_OptionSpace` (or `__act__` if no self-assessment), each entry should include `Options` (length 4) and `Chosen Index`.
  - **Audience:** Under the audience evaluation channel, each entry when option space is used should include `Options` and `Chosen Index`.
- Optionally run without `--use_option_space` and confirm no new keys or extra structure appear, and that behavior is unchanged.

---

## 5. Summary of file edits

| File | Change |
|------|--------|
| `concordia/components/agent/impression_management_pe.py` | 1) Add `ComponentWithLogging` to `IMPEActComponent`; 2) In `_use_option_space` branch of `IMPEActComponent`, call `_logging_channel` with Options + Chosen Index + Chosen. 3) In `IMPEAudienceEvaluationComponent` `_use_option_space` branch, extend `_logging_channel` payload with Options + Chosen Index. |
| `concordia/prefabs/entity/impression_management_actor.py` | When `use_option_space` is True, add `base_act_component` to `components_of_agent` with key `'IMPE_Act_OptionSpace'` so it receives a logging channel when wrapped by self-assessment. |

No changes to `results.save_component_logs()` or CLI flags are required; the new data will appear in the existing component log structure.
