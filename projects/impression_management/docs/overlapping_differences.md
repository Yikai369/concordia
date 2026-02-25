# Overlapping Differences: Common Prototype Features

## Overview

This document identifies differences that appear in **BOTH** comparison documents:
1. `pe_conversation_openai.py` vs `pe_conversation_prototype.py`
2. `impression_management_standard` vs `pe_conversation_prototype.py`

These overlapping differences represent **core distinguishing features** of the prototype that differ from BOTH standard implementations.

---

## Overlapping Differences Summary

| Feature | `pe_conversation_openai.py` | `impression_management_standard` | `pe_conversation_prototype.py` |
|---------|----------------------------|----------------------------------|--------------------------------|
| **Personality Traits** | Score-based (0-3) | Score-based (0-3) | LLM-generated paragraphs |
| **Trait Source** | Fixed list | Fixed list | Excel spreadsheet |
| **Self-Reflection** | Basic | Threshold-based (optional) | Always revises (unconditional) |
| **Option Generation** | ❌ No | ❌ No | ✅ Yes (experimental) |
| **Spreadsheet Support** | ❌ No | ❌ No | ✅ Yes |
| **Question Checks** | ❌ No | ❌ No | ✅ Yes |
| **World-Building** | Direct, minimal | Generic | Elaborate "2A25" fictional world |
| **Interview Role** | Product Manager | Product Manager (configurable) | Customer Service Agent (hardcoded) |
| **Agent Names** | John, Jane | John, Jane (configurable) | Riffer, Caden (hardcoded) |
| **Utterance Field** | `speaker` | `speaker` | `actor` |
| **TurnLog Fields** | `speaker`/`listener` | `speaker`/`listener` | `actor`/`audience` |
| **Trait Paragraphs in JSON** | ❌ No | ❌ No | ✅ Yes |
| **Reflection Text in TurnLog** | ✅ Yes | ✅ Yes | ❌ No |
| **Debug Code** | ❌ No | ❌ No | ✅ Yes (print statements) |
| **Hardcoded Values** | ❌ No | ❌ No | ✅ Yes (file paths, questions) |

---

## Detailed Overlapping Differences

### 1. Personality Trait System

**Both Standards:**
- Use **score-based system** (0-3 scale)
- Traits from **fixed list** in code
- Traits included **directly in prompts** with scores
- Format: `"- {name} ({score}/3): {assertion}"`

**Prototype:**
- Uses **LLM-generated paragraphs** from trait assertions
- Traits loaded from **Excel spreadsheet**
- Traits converted to **narrative paragraphs** via LLM
- Paragraphs stored in `trait_paragraph` attribute
- Paragraphs included in prompts instead of individual traits
- Paragraphs saved in JSON output

**Key Overlap:** Both standards use the same trait approach (scores + fixed list), while prototype uses a completely different approach (paragraphs + spreadsheet).

---

### 2. Self-Reflection Mechanism

**`pe_conversation_openai.py`:**
- Basic reflection in `learning()` method
- No self-reflection on responses
- No consistency checking

**`impression_management_standard`:**
- Optional `IMPESelfAssessmentComponent`
- Threshold-based (consistency score < 0.7)
- Can enable/disable revision
- Provides feedback on inconsistencies

**Prototype:**
- `actor_self_reflection()` and `audience_self_reflection()` methods
- **Always revises** if traits exist (unconditional)
- No consistency score or threshold
- No feedback mechanism
- Actor self-reflection is used, audience self-reflection exists but is never called

**Key Overlap:** Both standards either lack self-reflection (`pe_conversation_openai.py`) or have optional threshold-based reflection (`impression_management_standard`), while prototype has unconditional always-on self-reflection.

---

### 3. Option Space Generation

**Both Standards:**
- ❌ No option generation
- Direct response generation

**Prototype:**
- ✅ `generate_option_space()` method
- ✅ `choose_option()` method
- Generates 4 distinct response options
- LLM chooses one with deliberation
- Currently commented out in code (lines 618-622)

**Key Overlap:** Neither standard has this feature; it's unique to prototype.

---

### 4. Spreadsheet Support

**Both Standards:**
- ❌ No spreadsheet support
- Traits defined in code

**Prototype:**
- ✅ `extract_traits_from_spreadsheet()` function
- Loads traits from Excel file: `"autism-measures-compilation.xlsx"`
- Uses pandas to read Excel
- Each column becomes a `survey` field
- Each row becomes an `assertion`

**Key Overlap:** Neither standard supports spreadsheet loading; prototype does.

---

### 5. Question Checks

**Both Standards:**
- ❌ No question checks
- No verification of agent understanding

**Prototype:**
- ✅ `question_check()` method
- Checks context understanding
- Checks personality understanding
- Adds 2 LLM calls per turn per agent
- Results stored in TurnLog (but fields commented out)

**Key Overlap:** Neither standard has this feature; it's unique to prototype.

---

### 6. World-Building Approach

**`pe_conversation_openai.py`:**
- Direct, minimal cultural norms
- Simple norm listing
- No fictional world-building

**`impression_management_standard`:**
- Generic world-building context
- Standard cultural norm descriptions
- No elaborate fictional world

**Prototype:**
- Elaborate "2A25" fictional world
- Detailed narrative about Cadens and Riffers
- Stigma and social dynamics
- Explicit disclaimers about fictional nature
- More immersive world-building

**Key Overlap:** Both standards use simpler, more direct approaches, while prototype has elaborate fictional world-building.

---

### 7. Interview Context & Agent Configuration

**Both Standards:**
- Interview role: **Product Manager**
- Agent names: **John, Jane**
- `impression_management_standard` has these configurable

**Prototype:**
- Interview role: **Customer Service Agent** (hardcoded)
- Agent names: **Riffer, Caden** (hardcoded)
- Hardcoded interview questions for Caden
- Hardcoded experiences for Riffer

**Key Overlap:** Both standards use the same role and names (Product Manager, John/Jane), while prototype uses different ones (Customer Service, Riffer/Caden).

---

### 8. Data Structure Field Names

**Both Standards:**
- `Utterance.speaker` (not `actor`)
- `TurnLog.speaker` and `TurnLog.listener` (not `actor`/`audience`)

**Prototype:**
- `Utterance.actor` (not `speaker`)
- `TurnLog.actor` and `TurnLog.audience` (not `speaker`/`listener`)

**Key Overlap:** Both standards use the same field naming convention (`speaker`/`listener`), while prototype uses different names (`actor`/`audience`).

---

### 9. Reflection Text in TurnLog

**Both Standards:**
```python
@dataclass
class TurnLog:
    ...
    reflection_text: str  # ← Both standards have this field
    ...
```

**Prototype:**
```python
@dataclass
class TurnLog:
    ...
    # Note: No reflection_text field
    ...
```

**Key Overlap:** Both standards include `reflection_text` in TurnLog to store the actor's reflection on how to improve. Prototype does not include this field.

---

### 10. JSON Output Structure

**Both Standards:**
- Turn logs only
- No trait paragraphs in output

**Prototype:**
- Includes `actor_traits` and `audience_traits` at top level
- Trait paragraphs stored in JSON output

**Key Overlap:** Both standards have the same output structure (turns only), while prototype adds trait paragraphs.

---

### 11. Code Quality

**Both Standards:**
- Clean production code
- No debug print statements
- Minimal commented code

**Prototype:**
- Debug print statements (lines 623, 827, 921-922)
- Commented-out code sections
- Hardcoded file paths

**Key Overlap:** Both standards maintain clean code, while prototype has debug code and hardcoded values.

---

## Unique Differences (Not Overlapping)

### Only in `pe_conversation_openai.py` vs Prototype:

1. **Framework Architecture**: N/A (both are standalone)
2. **Execution Model**: N/A (both use manual loops)
3. **Component System**: N/A (both use direct Agent class)

### Only in `impression_management_standard` vs Prototype:

1. **Framework Integration**: Standard uses Concordia framework, prototype is standalone
2. **Execution Model**: Standard uses `sim.play()`, prototype uses manual loop
3. **Component System**: Standard uses component-based architecture, prototype uses direct methods
4. **Modularity**: Standard is multi-file, prototype is single file
5. **Configurability**: Standard has configurable names/roles, prototype has hardcoded values

---

## Summary

**Core Overlapping Differences (11 total):**

1. ✅ **Personality Trait System**: Paragraphs vs Scores
2. ✅ **Trait Source**: Spreadsheet vs Fixed List
3. ✅ **Self-Reflection**: Unconditional vs Optional/None
4. ✅ **Option Generation**: Yes vs No
5. ✅ **Spreadsheet Support**: Yes vs No
6. ✅ **Question Checks**: Yes vs No
7. ✅ **World-Building**: Elaborate vs Simple
8. ✅ **Interview Role/Names**: Customer Service/Riffer-Caden vs Product Manager/John-Jane
9. ✅ **Data Structure Fields**: `actor`/`audience` vs `speaker`/`listener`
10. ✅ **Reflection Text in TurnLog**: No vs Yes
11. ✅ **Code Quality**: Debug code vs Clean code

**These represent the fundamental experimental features of the prototype that differ from BOTH standard implementations.**

---

**Document Version:** 1.0
**Last Updated:** 2025-12-27
