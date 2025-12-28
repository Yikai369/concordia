# Information Flow History Bank Implementation Plan

## Overview
Implement a comprehensive information flow history bank for each agent that captures and persists all model inputs (prompts) and outputs (responses) for debugging purposes.

## Current State Analysis

### Existing Logging Infrastructure
- `EntityAgentWithLogging` uses `Measurements` object to collect component logs
- Components can implement `ComponentWithLogging` interface
- Some components log prompts (e.g., `ScriptedActComponent`)
- Logs are stored in-memory only via `Measurements.get_all_channels()`
- No centralized, persistent history bank

### Gaps
1. **Not Comprehensive**: Not all components log full prompts/responses
2. **Not Persistent**: Logs are in-memory, lost after simulation ends
3. **Not Centralized**: Each component logs separately, no unified view per agent
4. **No Model-Level Interception**: Can't capture all LLM calls automatically

## Implementation Plan

### Phase 1: Model-Level Interception (Core Foundation)

#### 1.1 Create Model Wrapper for Logging
**File**: `concordia/language_model/logging_wrapper.py`

```python
class LoggingLanguageModel(language_model.LanguageModel):
    """Wraps a language model to log all inputs and outputs."""

    def __init__(
        self,
        model: language_model.LanguageModel,
        history_bank: 'InformationFlowHistoryBank',
        agent_name: str,
    ):
        self._model = model
        self._history_bank = history_bank
        self._agent_name = agent_name

    def sample_text(self, prompt: str, **kwargs) -> str:
        response = self._model.sample_text(prompt, **kwargs)
        self._history_bank.add_interaction(
            agent_name=self._agent_name,
            prompt=prompt,
            response=response,
            method='sample_text',
            kwargs=kwargs,
        )
        return response

    def sample_choice(self, prompt: str, responses: Sequence[str], **kwargs):
        result = self._model.sample_choice(prompt, responses, **kwargs)
        index, response, info = result
        self._history_bank.add_interaction(
            agent_name=self._agent_name,
            prompt=prompt,
            response=response,
            method='sample_choice',
            kwargs={**kwargs, 'responses': responses},
            metadata={'index': index, 'info': info},
        )
        return result
```

#### 1.2 Create Information Flow History Bank
**File**: `concordia/utils/information_flow_history.py`

```python
@dataclass
class ModelInteraction:
    """Single model interaction record."""
    timestamp: datetime.datetime
    agent_name: str
    component_name: str | None  # Which component made the call
    method: str  # 'sample_text' or 'sample_choice'
    prompt: str  # Full prompt sent to model
    response: str  # Model response
    kwargs: dict[str, Any]  # Model call parameters (temperature, max_tokens, etc.)
    metadata: dict[str, Any]  # Additional context (choice index, etc.)
    turn: int | None = None  # Simulation turn number if available
    phase: str | None = None  # 'pre_act', 'act', 'post_act', 'observe'


class InformationFlowHistoryBank:
    """Stores complete information flow history for all agents."""

    def __init__(self, save_dir: str | None = None):
        self._interactions: dict[str, list[ModelInteraction]] = {}  # agent_name -> list
        self._lock = threading.Lock()
        self._save_dir = save_dir
        self._turn_counter: dict[str, int] = {}  # agent_name -> current turn

    def add_interaction(
        self,
        agent_name: str,
        prompt: str,
        response: str,
        method: str,
        kwargs: dict[str, Any],
        component_name: str | None = None,
        metadata: dict[str, Any] | None = None,
        phase: str | None = None,
    ):
        """Add a model interaction to the history."""
        with self._lock:
            if agent_name not in self._interactions:
                self._interactions[agent_name] = []
                self._turn_counter[agent_name] = 0

            interaction = ModelInteraction(
                timestamp=datetime.datetime.now(),
                agent_name=agent_name,
                component_name=component_name,
                method=method,
                prompt=prompt,
                response=response,
                kwargs=kwargs or {},
                metadata=metadata or {},
                turn=self._turn_counter[agent_name],
                phase=phase,
            )
            self._interactions[agent_name].append(interaction)

    def get_agent_history(self, agent_name: str) -> list[ModelInteraction]:
        """Get all interactions for an agent."""
        with self._lock:
            return list(self._interactions.get(agent_name, []))

    def increment_turn(self, agent_name: str):
        """Increment turn counter for an agent."""
        with self._lock:
            if agent_name not in self._turn_counter:
                self._turn_counter[agent_name] = 0
            self._turn_counter[agent_name] += 1

    def save_to_json(self, filepath: str | None = None):
        """Save history to JSON file."""
        if filepath is None:
            if self._save_dir is None:
                raise ValueError("No save directory specified")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self._save_dir, f"information_flow_history_{timestamp}.json")

        with self._lock:
            data = {
                agent_name: [
                    asdict(interaction) for interaction in interactions
                ]
                for agent_name, interactions in self._interactions.items()
            }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

        return filepath

    def load_from_json(self, filepath: str):
        """Load history from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        with self._lock:
            for agent_name, interactions_data in data.items():
                self._interactions[agent_name] = [
                    ModelInteraction(**item) for item in interactions_data
                ]
```

### Phase 2: Integration with Agent Architecture

#### 2.1 Modify EntityAgentWithLogging
**File**: `concordia/agents/entity_agent_with_logging.py`

Add optional history bank parameter:

```python
def __init__(
    self,
    agent_name: str,
    act_component: entity_component.ActingComponent,
    context_processor: ... = None,
    context_components: Mapping[str, entity_component.ContextComponent] = ...,
    information_flow_history: InformationFlowHistoryBank | None = None,  # NEW
):
    # ... existing code ...
    self._information_flow_history = information_flow_history
```

#### 2.2 Wrap Models in Simulation
**File**: `concordia/prefabs/simulation/generic.py`

Modify `add_entity` and `add_game_master` to wrap models:

```python
def add_entity(self, instance_config, state=None):
    # ... existing code ...

    # Wrap model with logging if history bank is available
    if hasattr(self, '_information_flow_history') and self._information_flow_history:
        from concordia.language_model import logging_wrapper
        wrapped_model = logging_wrapper.LoggingLanguageModel(
            model=self._model,
            history_bank=self._information_flow_history,
            agent_name=entity.name,
        )
        entity = entity_prefab.build(model=wrapped_model, memory_bank=memory_bank)
    else:
        entity = entity_prefab.build(model=self._model, memory_bank=memory_bank)

    # ... rest of code ...
```

#### 2.3 Add History Bank to Simulation
**File**: `concordia/prefabs/simulation/generic.py`

```python
def __init__(
    self,
    config: Config,
    model: language_model.LanguageModel,
    embedder: Callable[[str], np.ndarray],
    engine: engine_lib.Engine = sequential.Sequential(),
    enable_information_flow_logging: bool = False,  # NEW
    information_flow_save_dir: str | None = None,  # NEW
):
    # ... existing code ...

    if enable_information_flow_logging:
        from concordia.utils import information_flow_history
        self._information_flow_history = information_flow_history.InformationFlowHistoryBank(
            save_dir=information_flow_save_dir,
        )
    else:
        self._information_flow_history = None
```

### Phase 3: Component-Level Integration (Optional Enhancement)

#### 3.1 Enhance Component Logging
For components that use `InteractiveDocument`, we can extract the full prompt:

**File**: `concordia/components/agent/impression_management_pe.py` (example)

Modify components to pass component name to history bank:

```python
# In IMPEActComponent.get_action_attempt()
prompt = interactive_document.InteractiveDocument(self._model)
# ... build prompt ...
response = prompt.open_question(...)

# If model is wrapped, it will automatically log
# But we can also add component context
if hasattr(self._model, '_history_bank'):
    # The wrapper will handle logging, but we can add metadata
    pass
```

### Phase 4: Turn Tracking

#### 4.1 Integrate with Engine
**File**: `concordia/environment/engines/sequential.py`

Add turn tracking:

```python
def step(self, ...):
    # ... existing code ...

    # Increment turn for all entities
    if hasattr(simulation, '_information_flow_history') and simulation._information_flow_history:
        for entity in simulation.get_entities():
            simulation._information_flow_history.increment_turn(entity.name)
        for gm in simulation.game_masters:
            simulation._information_flow_history.increment_turn(gm.name)
```

### Phase 5: Save/Load Interface

#### 5.1 Add Save Method to Simulation
**File**: `concordia/prefabs/simulation/generic.py`

```python
def save_information_flow_history(self, filepath: str | None = None) -> str | None:
    """Save information flow history to JSON."""
    if self._information_flow_history:
        return self._information_flow_history.save_to_json(filepath)
    return None

def get_information_flow_history(self, agent_name: str | None = None):
    """Get information flow history for agent(s)."""
    if not self._information_flow_history:
        return None

    if agent_name:
        return self._information_flow_history.get_agent_history(agent_name)
    else:
        return {
            name: self._information_flow_history.get_agent_history(name)
            for name in self._information_flow_history._interactions.keys()
        }
```

## Usage Example

```python
from concordia.prefabs.simulation import generic
from concordia.utils import information_flow_history

# Create simulation with history logging enabled
sim = generic.Simulation(
    config=config,
    model=model,
    embedder=embedder,
    enable_information_flow_logging=True,
    information_flow_save_dir="./logs",
)

# Run simulation
sim.play(premise="...", max_steps=10)

# Save history
history_file = sim.save_information_flow_history()
print(f"History saved to: {history_file}")

# Access history programmatically
actor_history = sim.get_information_flow_history("Actor")
for interaction in actor_history:
    print(f"Turn {interaction.turn}: {interaction.method}")
    print(f"Prompt: {interaction.prompt[:100]}...")
    print(f"Response: {interaction.response[:100]}...")
```

## File Structure

```
concordia/
├── language_model/
│   └── logging_wrapper.py          # NEW: Model wrapper for logging
├── utils/
│   └── information_flow_history.py # NEW: History bank implementation
├── agents/
│   └── entity_agent_with_logging.py # MODIFY: Add history bank support
└── prefabs/
    └── simulation/
        └── generic.py               # MODIFY: Add history bank to simulation
```

## Implementation Steps (Detailed)

### Phase 1: Core Infrastructure (Foundation)
1. **Step 1.1**: Create `InformationFlowHistoryBank` class
   - Basic structure with `_interactions` dict
   - `add_interaction()` method with thread safety
   - `get_agent_history()` method
   - Unit tests

2. **Step 1.2**: Add JSON serialization support
   - Custom JSON encoder for datetime
   - `save_to_json()` method
   - `load_from_json()` method
   - Unit tests for save/load

3. **Step 1.3**: Add turn tracking
   - `increment_turn()` method
   - Turn counter per agent
   - Unit tests

### Phase 2: Model Wrapper (Interception Layer)
4. **Step 2.1**: Create `LoggingLanguageModel` wrapper
   - Implement `sample_text()` with logging
   - Implement `sample_choice()` with logging
   - Error handling (try-except around logging)
   - Unit tests

5. **Step 2.2**: Add context tracking (thread-local)
   - Create context module with thread-local storage
   - `set_component_context()` and `clear_component_context()`
   - Update wrapper to use context
   - Unit tests

6. **Step 2.3**: Handle edge cases
   - Forced responses (don't log)
   - Error interactions (log errors)
   - Nested wrappers (test compatibility)
   - Unit tests

### Phase 3: Integration (Simulation Layer)
7. **Step 3.1**: Add history bank to Simulation
   - Add `enable_information_flow_logging` parameter
   - Add `information_flow_save_dir` parameter
   - Initialize history bank in `__init__`
   - Unit tests

8. **Step 3.2**: Wrap models in entity creation
   - Modify `add_entity()` to wrap model
   - Modify `add_game_master()` to wrap model
   - Handle case where history bank is None
   - Integration tests

9. **Step 3.3**: Add save/load methods to Simulation
   - `save_information_flow_history()` method
   - `get_information_flow_history()` method
   - Integration tests

### Phase 4: Turn Tracking (Engine Layer)
10. **Step 4.1**: Add turn tracking to Sequential engine
    - Increment turns at start of `step()`
    - Test with simple simulation

11. **Step 4.2**: Add turn tracking to other engines
    - Simultaneous engine
    - Other engines if any
    - Integration tests

### Phase 5: Component Context (Component Layer)
12. **Step 5.1**: Add context manager for components
    - Create `component_context()` context manager
    - Update EntityAgent to set context in act/observe
    - Integration tests

13. **Step 5.2**: Update key components (optional)
    - Update components to use context manager
    - Test component name tracking
    - Integration tests

### Phase 6: Testing & Validation
14. **Step 6.1**: Comprehensive unit tests
    - All edge cases
    - Error scenarios
    - Thread safety

15. **Step 6.2**: Integration tests
    - Simple simulations
    - Multi-agent simulations
    - Error scenarios

16. **Step 6.3**: End-to-end tests
    - Impression management example
    - Performance tests
    - Long simulation tests

### Phase 7: Documentation & Polish
17. **Step 7.1**: Add docstrings
    - All public methods
    - Usage examples in docstrings

18. **Step 7.2**: Create usage examples
    - Simple example
    - Advanced example
    - Error handling example

19. **Step 7.3**: Update main documentation
    - Add to README if needed
    - Add to API documentation

### Implementation Dependencies

```
Step 1.1 (History Bank)
  └─> Step 1.2 (JSON serialization)
  └─> Step 1.3 (Turn tracking)
       └─> Step 2.1 (Model Wrapper)
            └─> Step 2.2 (Context tracking)
                 └─> Step 2.3 (Edge cases)
                      └─> Step 3.1 (Simulation integration)
                           └─> Step 3.2 (Model wrapping)
                                └─> Step 3.3 (Save/load methods)
                                     └─> Step 4.1 (Turn tracking in engines)
                                          └─> Step 5.1 (Component context)
                                               └─> Step 6.x (Testing)
                                                    └─> Step 7.x (Documentation)
```

### Critical Path
The minimal viable implementation requires:
1. History Bank (Step 1.1, 1.2)
2. Model Wrapper (Step 2.1)
3. Simulation Integration (Step 3.1, 3.2)
4. Basic Testing (Step 6.1)

Everything else can be added incrementally.

## Testing Strategy

### Unit Tests
1. **History Bank**:
   - Test `add_interaction()` with various inputs
   - Test `get_agent_history()` for single and multiple agents
   - Test `increment_turn()` and turn tracking
   - Test `save_to_json()` and `load_from_json()`
   - Test thread safety with concurrent access
   - Test error handling (invalid JSON, missing files, etc.)
   - Test datetime serialization/deserialization

2. **Model Wrapper**:
   - Test `sample_text()` logging
   - Test `sample_choice()` logging
   - Test error propagation (wrapper shouldn't swallow errors)
   - Test with nested wrappers (CallLimit, Retry, etc.)
   - Test with forced responses (shouldn't log)
   - Test context passing (component name, phase)

### Integration Tests
1. **Simple Simulation**:
   - Create minimal simulation with 1 entity
   - Verify all model calls are logged
   - Verify turn tracking works
   - Verify save/load works

2. **Multi-Agent Simulation**:
   - Test with multiple entities and game master
   - Verify each agent's history is separate
   - Verify turn counters are independent per agent

3. **Component Context**:
   - Test component name tracking
   - Test phase tracking (pre_act, act, post_act, observe)
   - Test context clearing

### End-to-End Tests
1. **Impression Management Example**:
   - Run full simulation
   - Verify all interactions are captured
   - Verify history can be saved and loaded
   - Verify no performance degradation

2. **Error Scenarios**:
   - Test with model that throws exceptions
   - Test with disk full (save failure)
   - Test with invalid save directory
   - Verify simulation continues even if logging fails

### Performance Tests
1. **Overhead Measurement**:
   - Compare simulation time with/without logging
   - Measure memory usage
   - Test with long simulations (1000+ turns)

2. **Streaming Mode**:
   - Test streaming to disk for long simulations
   - Compare memory usage with/without streaming

### Edge Cases
1. **Empty History**: Test with no interactions
2. **Very Long Prompts**: Test with prompts > 100k characters
3. **Special Characters**: Test with unicode, newlines, etc.
4. **Concurrent Access**: Test thread safety
5. **Rapid Turn Changes**: Test turn increment timing
6. **Agent Name Changes**: Test if agent name can change
7. **Checkpoint/Resume**: Test history bank state persistence
8. **Multiple Wrappers**: Test with CallLimit + Retry + Logging wrappers
9. **Forced Responses**: Verify forced responses aren't logged
10. **Diversified Queries**: Test open_question_diversified logging

## Benefits

1. **Complete Visibility**: See every model call made by each agent
2. **Debugging**: Trace exactly what prompts led to specific responses
3. **Reproducibility**: Save and replay model interactions
4. **Analysis**: Analyze prompt patterns, response quality, etc.
5. **Non-Intrusive**: Works with existing code via wrapper pattern

## Considerations

1. **Performance**: Wrapper adds minimal overhead (just logging)
2. **Storage**: JSON files can be large for long simulations
3. **Privacy**: Contains full prompts/responses - handle sensitive data carefully
4. **Optional**: Feature is opt-in, doesn't affect existing code

## Missing Pieces & Additional Considerations

### Critical Missing Items

#### 1. Error Handling in Wrapper
**Issue**: If logging fails, it shouldn't break the simulation.

**Solution**: Wrap logging in try-except:
```python
def sample_text(self, prompt: str, **kwargs) -> str:
    response = self._model.sample_text(prompt, **kwargs)
    try:
        self._history_bank.add_interaction(...)
    except Exception as e:
        # Log error but don't fail
        import warnings
        warnings.warn(f"Failed to log interaction: {e}")
    return response
```

#### 2. Component Name Tracking
**Issue**: Components don't know their own name in the context.

**Solution Options**:
- **Option A**: Use stack trace inspection (fragile, slow)
- **Option B**: Pass component context through call chain (requires changes)
- **Option C**: Use thread-local storage to track current component (recommended)
- **Option D**: Don't track component name automatically, let components opt-in

**Recommended**: Option C + Option D hybrid:
```python
import threading
_context = threading.local()

def set_component_context(component_name: str, phase: str):
    """Set current component context for logging."""
    _context.component_name = component_name
    _context.phase = phase

def clear_component_context():
    """Clear component context."""
    _context.component_name = None
    _context.phase = None
```

#### 3. Phase Tracking
**Issue**: Need to know if call is in pre_act, act, post_act, or observe phase.

**Solution**: Use thread-local context (same as component tracking):
```python
# In EntityAgent methods:
def act(self, ...):
    set_component_context(None, 'act')
    try:
        return self._act_component.act(...)
    finally:
        clear_component_context()

def observe(self, ...):
    set_component_context(None, 'observe')
    try:
        return self._observe_component.observe(...)
    finally:
        clear_component_context()
```

#### 4. Forced Responses
**Issue**: `InteractiveDocument.open_question(forced_response=...)` doesn't call the model.

**Solution**: Only log if `forced_response is None`:
```python
# In InteractiveDocument.open_question():
if forced_response is None:
    response = self._model.sample_text(...)  # This will be logged
else:
    response = forced_response  # Don't log this
```

#### 5. Nested Model Wrappers
**Issue**: Model might already be wrapped (e.g., `CallLimitLanguageModel`, `RetryLanguageModel`).

**Solution**: Check if already wrapped, and if so, wrap the inner model:
```python
def _unwrap_model(self, model):
    """Unwrap nested wrappers to get to base model."""
    if hasattr(model, '_model'):
        return self._unwrap_model(model._model)
    return model

# Or: Always wrap at the outermost level, preserve existing wrappers
```

#### 6. Turn Tracking Timing
**Issue**: When should turns be incremented? Before or after interactions?

**Solution**: Increment at the start of each engine step, before any interactions:
```python
def step(self, simulation, ...):
    # Increment turn for all entities at start of step
    if hasattr(simulation, '_information_flow_history'):
        for entity in simulation.get_entities():
            simulation._information_flow_history.increment_turn(entity.name)
        for gm in simulation.game_masters:
            simulation._information_flow_history.increment_turn(gm.name)

    # ... rest of step logic ...
```

#### 7. JSON Serialization
**Issue**: `datetime.datetime` objects aren't JSON serializable.

**Solution**: Use custom JSON encoder:
```python
import json
from datetime import datetime

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

json.dump(data, f, indent=2, cls=DateTimeEncoder)
```

#### 8. Multiple Model Calls in One Operation
**Issue**: `open_question_diversified` makes multiple model calls.

**Solution**: Log each call separately with metadata indicating it's part of a diversified query:
```python
# In open_question_diversified:
for i in range(num_samples):
    response = self._model.sample_text(...)
    # Each call logged separately with metadata: {'diversified_index': i, 'total_samples': num_samples}
```

#### 9. Memory Management for Long Simulations
**Issue**: Storing all interactions in memory can be problematic.

**Solution**: Add streaming option:
```python
class InformationFlowHistoryBank:
    def __init__(self, save_dir: str | None = None, stream_to_disk: bool = False):
        self._stream_to_disk = stream_to_disk
        if stream_to_disk:
            # Write each interaction immediately to disk
            self._log_file = open(...)

    def add_interaction(self, ...):
        if self._stream_to_disk:
            # Write immediately
            json.dump(asdict(interaction), self._log_file)
            self._log_file.write('\n')
        else:
            # Store in memory
            self._interactions[agent_name].append(interaction)
```

#### 10. Checkpoint/Resume Support
**Issue**: If simulation can be checkpointed, history bank state needs to be saved/restored.

**Solution**: Add state methods:
```python
def get_state(self) -> dict:
    """Get state for checkpointing."""
    return {
        'interactions': {name: [asdict(i) for i in interactions]
                         for name, interactions in self._interactions.items()},
        'turn_counters': self._turn_counter.copy(),
    }

def set_state(self, state: dict):
    """Restore state from checkpoint."""
    # Restore interactions and turn counters
```

#### 11. Component Context Passing (Enhanced)
**Solution**: Add context manager for components:
```python
@contextlib.contextmanager
def component_context(component_name: str, phase: str):
    """Context manager for component logging context."""
    set_component_context(component_name, phase)
    try:
        yield
    finally:
        clear_component_context()

# Usage in components:
def pre_act(self, ...):
    with component_context(self.__class__.__name__, 'pre_act'):
        # ... component logic ...
```

#### 12. Backward Compatibility
**Issue**: Need to ensure existing code works without changes.

**Solution**:
- All new parameters are optional with defaults
- Wrapper is transparent (implements same interface)
- No breaking changes to existing APIs

#### 13. InteractiveDocument Methods
**Issue**: Need to check all methods that call the model.

**Methods to handle**:
- `open_question()` - ✅ Handled
- `open_question_diversified()` - ✅ Handled (multiple calls)
- `multiple_choice_question()` - ⚠️ Need to check
- Any other methods?

#### 14. Thread Safety Verification
**Issue**: Need to ensure thread safety for concurrent simulations.

**Solution**:
- Use locks in history bank (already in plan)
- Use thread-local storage for context (prevents race conditions)
- Test with concurrent simulations

#### 15. Agent Name Resolution
**Issue**: Need to get agent name when model is called.

**Solution**: Store agent name in wrapper (already in plan), but also handle cases where entity name might change or be None.

### Additional Implementation Details

#### Enhanced Wrapper with Context
```python
class LoggingLanguageModel(language_model.LanguageModel):
    def __init__(self, model, history_bank, agent_name):
        self._model = model
        self._history_bank = history_bank
        self._agent_name = agent_name

    def sample_text(self, prompt: str, **kwargs) -> str:
        # Get context from thread-local storage
        component_name = getattr(_context, 'component_name', None)
        phase = getattr(_context, 'phase', None)

        # Call model
        try:
            response = self._model.sample_text(prompt, **kwargs)
        except Exception as e:
            # Log error interaction
            self._history_bank.add_interaction(
                agent_name=self._agent_name,
                prompt=prompt,
                response=f"<ERROR: {str(e)}>",
                method='sample_text',
                kwargs=kwargs,
                component_name=component_name,
                phase=phase,
                metadata={'error': str(e), 'error_type': type(e).__name__},
            )
            raise

        # Log successful interaction
        try:
            self._history_bank.add_interaction(
                agent_name=self._agent_name,
                prompt=prompt,
                response=response,
                method='sample_text',
                kwargs=kwargs,
                component_name=component_name,
                phase=phase,
            )
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to log interaction: {e}")

        return response
```

#### Turn Tracking Integration
```python
# In engine step():
def step(self, simulation, ...):
    # Increment turn at START of step (before any interactions)
    if hasattr(simulation, '_information_flow_history') and simulation._information_flow_history:
        for entity in simulation.get_entities():
            simulation._information_flow_history.increment_turn(entity.name)
        for gm in simulation.game_masters:
            simulation._information_flow_history.increment_turn(gm.name)

    # ... rest of step logic ...
```

## Future Enhancements

1. **Filtering**: Filter by component, method, turn range
2. **Search**: Search prompts/responses by keyword
3. **Compression**: Compress old interactions
4. **Database Backend**: Use SQLite instead of JSON for large histories
5. **Visualization**: Web UI to browse interaction history
6. **Diff View**: Compare prompts/responses across turns
7. **Performance Metrics**: Track latency, token counts, etc.
8. **Selective Logging**: Allow filtering which interactions to log
9. **Export Formats**: Support CSV, Parquet, etc.
10. **Query Interface**: SQL-like queries on history
