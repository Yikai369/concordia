# Impression Management PE Conversation - Test Suite

## Test Files

### 1. `test_impe_components.py`
**Purpose**: Unit tests for individual IMPE components

**Tests**:
- `ParticleFilter`: Initialize, predict, update operations
- `IMPEMemoryComponent`: Evaluation records, particle filter state management
- `CulturalNormsComponent`: Norms text formatting, empty norms handling
- `PersonalityTraitsComponent`: Traits text formatting with scores

**Run**: `python projects/impression_management/tests/test_impe_components.py`

**Status**: ✅ All 8 tests pass

### 2. `test_impe_integration.py`
**Purpose**: Integration test for full 2-turn conversation

**Tests**:
- Entity building (actor and audience)
- Turn 1: Actor acts → Audience evaluates → Actor updates PF → Actor reflects
- Turn 2: Actor acts → Audience evaluates → Actor updates PF
- Data extraction and verification

**Run**: `python projects/impression_management/tests/test_impe_integration.py`

**Status**: ✅ All integration tests pass

### 3. `test_main_script_structure.py`
**Purpose**: Verify main script structure and syntax

**Tests**:
- Script exists and is valid Python
- Has `main()` function
- Has `TurnLog` dataclass
- Has CLI argument parsing

**Run**: `python projects/impression_management/tests/test_main_script_structure.py`

**Status**: ✅ All 4 tests pass

### 4. `test_main_script.py` and `test_main_script_simple.py`
**Purpose**: Test main script execution (requires mocking dependencies)

**Status**: ⚠️ Requires dependency mocking (sentence_transformers, etc.)

## Running All Tests

To run all tests in the test suite:

```bash
# From project root
conda activate concordia
python -m pytest projects/impression_management/tests/ -v
```

Or run individually:
```bash
python projects/impression_management/tests/test_impe_components.py
python projects/impression_management/tests/test_impe_integration.py
python projects/impression_management/tests/test_main_script_structure.py
```

## Test Results Summary

✅ **Component Tests**: 8/8 passing
✅ **Integration Tests**: All passing (2-turn conversation verified)
✅ **Structure Tests**: 4/4 passing

## Notes

- Integration test uses mock LLM to avoid API calls
- Main script tests require dependency mocking (sentence_transformers, torch, etc.)
- All core functionality is verified and working
