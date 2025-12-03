# Concordia Framework Tutorial: Building Your First Game

Welcome to Concordia! This tutorial will guide you through building your first game using the Concordia framework. By the end of this tutorial, you'll understand the core concepts and be able to create your own simulations.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation and Setup](#installation-and-setup)
3. [Common Errors and Troubleshooting](#common-errors-and-troubleshooting) ⚠️ Read this first!
4. [Quick Reference: Key Concepts](#quick-reference-key-concepts) 📖 Glossary
5. [Core Concepts](#core-concepts)
6. [Your First Game](#your-first-game)
7. [Understanding the Game Loop](#understanding-the-game-loop)
8. [Customizing Entities](#customizing-entities)
9. [Customizing Game Masters](#customizing-game-masters)
10. [Working with Engines](#working-with-engines)
11. [Advanced Topics](#advanced-topics)
12. [Best Practices](#best-practices)
13. [Troubleshooting](#troubleshooting)
14. [Next Steps](#next-steps)

---

## Introduction

### What is Concordia?

Concordia is a library for building **generative social simulations** - interactive environments where AI agents can interact with each other and their environment. Think of it like a tabletop role-playing game where:

- **Entities** are the players (characters in your simulation)
- **Game Masters** are the narrators who describe the world and resolve actions
- **Engines** control how the game flows (turn-based, simultaneous, etc.)

### Key Features

- **Natural Language Actions**: Agents describe what they want to do in plain English
- **Flexible World Simulation**: Game Masters translate actions into world changes
- **Component-Based Architecture**: Build complex behaviors from simple components
- **Memory System**: Agents remember past events and use them to make decisions
- **Multiple Engine Types**: Sequential (turn-based) or simultaneous actions

### What Can You Build?

- Social simulations (friends interacting, negotiations, conversations)
- Economic simulations (markets, trading, auctions)
- Narrative games (interactive stories, role-playing scenarios)
- Research experiments (studying agent behavior, social dynamics)

---

## Installation and Setup

### Prerequisites

- Python 3.8 or higher
- Access to an LLM API (OpenAI, Together AI, or local model)
- Basic Python knowledge

### Step 1: Install Concordia

```bash
pip install gdm-concordia
```

Or for development:

```bash
git clone https://github.com/google-deepmind/concordia
cd concordia
pip install --editable .[dev]
```

### Step 2: Install Sentence Transformers

You'll need a sentence embedder for the memory system:

```bash
pip install sentence-transformers
```

### Step 3: Verify Installation

Let's make sure everything is installed correctly:

```python
# Quick verification test
try:
    from concordia.language_model import utils as language_model_utils
    import sentence_transformers
    import numpy as np
    print("✓ All required packages imported successfully!")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Please check your installation.")
```

### Step 4: Set Up Your Language Model

**⚠️ Important: API Costs**
- API calls cost money! GPT-4 is expensive (~$0.03 per 1K tokens)
- GPT-3.5-turbo is cheaper (~$0.0015 per 1K tokens)
- **Always test with `disable_language_model=True` first** to avoid unexpected costs
- Start with small `max_steps` (5-10) when testing

**🔒 Security: Protecting Your API Key**
- **Never commit API keys to version control!**
- Use environment variables instead of hardcoding keys
- We'll show both methods below, but prefer environment variables

You need to provide Concordia with access to a language model. Here are your options:

**Option A: OpenAI API (Recommended for beginners)**

**Using Environment Variables (Secure - Recommended):**
```python
import os
from concordia.language_model import utils as language_model_utils

# Get API key from environment variable
API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found. Set it with: export OPENAI_API_KEY='your-key'"
    )

model = language_model_utils.language_model_setup(
    api_type='openai',
    model_name='gpt-3.5-turbo',  # Start with cheaper model
    api_key=API_KEY,
)
```

**Using Hardcoded Key (For quick testing only - NOT for production):**
```python
from concordia.language_model import utils as language_model_utils

# ⚠️ WARNING: Never commit this to version control!
model = language_model_utils.language_model_setup(
    api_type='openai',
    model_name='gpt-3.5-turbo',  # or 'gpt-4' (more expensive)
    api_key='your-api-key-here',  # Replace with your actual key
)
```

**Option B: Together AI (for open-source models)**
```python
model = language_model_utils.language_model_setup(
    api_type='together_ai',
    model_name='google/gemma-2-27b-it',
    api_key='your-api-key-here',
)
```

**Option C: Local Model (Ollama)**
```python
from concordia.language_model import ollama_model

model = ollama_model.OllamaModel(model_name='llama2')
```

**Option D: Disable for Testing**
```python
model = language_model_utils.language_model_setup(
    api_type='openai',
    model_name='gpt-4',
    api_key='',
    disable_language_model=True,  # Returns dummy responses
)
```

### Step 5: Set Up Sentence Embedder

The embedder converts text into numerical vectors (embeddings) for the memory system. Here's how to set it up:

```python
import numpy as np
import sentence_transformers

# For production: Load a sentence transformer model
st_model = sentence_transformers.SentenceTransformer(
    'sentence-transformers/all-mpnet-base-v2'
)

# Create the embedder function
# A lambda function is a shorthand way to define a simple function
# This is equivalent to:
#   def embed_text(text):
#       return st_model.encode(text, show_progress_bar=False)
#   embedder = embed_text
embedder = lambda x: st_model.encode(x, show_progress_bar=False)

# For testing (if DISABLE_LANGUAGE_MODEL=True)
# Use dummy embeddings to avoid loading the model
embedder = lambda _: np.ones(384)  # Returns array of 384 ones
```

**Understanding the Lambda Function:**
- `lambda x: ...` creates an anonymous function that takes one argument `x`
- `st_model.encode(x, ...)` converts text `x` into a numerical embedding
- The `show_progress_bar=False` prevents progress bars during encoding
- This function will be called by Concordia's memory system to embed text

---

## Common Errors and Troubleshooting

Before we dive into building games, let's cover common errors you might encounter:

### Error: "Invalid API key"
**Cause**: Your API key is incorrect or not set
**Solution**:
```python
# Check your API key
import os
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("API key not found in environment variables")
    print("Set it with: export OPENAI_API_KEY='your-key'")
```

### Error: "Prefab 'wrong_name__Entity' not found"
**Cause**: Prefab name is misspelled or doesn't exist
**Solution**:
- Check spelling (note the double underscore: `basic__Entity`)
- List available prefabs:
```python
prefabs = helper_functions.get_package_classes(entity_prefabs)
print("Available prefabs:", list(prefabs.keys()))
```

### Error: "Role must be ENTITY, GAME_MASTER, or INITIALIZER"
**Cause**: Typo in role name
**Solution**: Use `prefab_lib.Role.ENTITY` (not `'ENTITY'` as a string)

### Error: "No game master found"
**Cause**: Missing game master in instances
**Solution**: Always include at least one `Role.GAME_MASTER` instance

### Error: Import errors
**Cause**: Package not installed
**Solution**:
```bash
pip install gdm-concordia sentence-transformers
```

### Error: Embedder fails
**Cause**: Sentence transformer model not downloaded
**Solution**: The model downloads automatically on first use, but requires internet connection

---

## Quick Reference: Key Concepts

Before diving deep, here's a quick glossary of terms you'll see throughout this tutorial:

- **Entity**: A character/agent in your simulation (like a player in a game)
- **Game Master (GM)**: A special entity that controls the simulation, describes the world, and resolves actions
- **Prefab**: A pre-built template for creating entities or game masters (like a blueprint)
- **Instance**: A specific entity or game master created from a prefab (like an actual object built from a blueprint)
- **Component**: A building block that gives entities/game masters abilities (memory, observation, action generation, etc.)
- **Engine**: Controls how the simulation flows (sequential/turn-based, simultaneous/all-at-once, etc.)
- **Config**: Defines your simulation (premise, max steps, prefabs, instances)
- **Embedder**: Converts text into numerical vectors for the memory system
- **Role**: Determines what an instance does (ENTITY, GAME_MASTER, or INITIALIZER)

**Common Patterns:**
- Prefab names: `name__Type` (e.g., `basic__Entity`, `generic__GameMaster`)
- Always need: At least one GAME_MASTER and one ENTITY
- Initializer: Sets up shared memories, runs once at start

---

## Core Concepts

Before building your first game, let's understand the key concepts.

### 1. Entities

**Entities** are the characters in your simulation. They:
- Have a name and goal
- Make decisions based on their observations
- Take actions in natural language
- Remember past events

Think of entities as the "players" in your game.

### 2. Game Masters

**Game Masters** are special entities that:
- Describe the world state to entities
- Decide which entity acts next
- Resolve entity actions into world events
- Determine when the game ends

Think of game masters as the "narrator" or "referee" of your game.

### 3. Engines

**Engines** control the flow of the simulation:
- **Sequential Engine**: Turn-based (one entity acts at a time)
- **Simultaneous Engine**: All entities act at once
- **Configuration Engine**: Pre-configured sequence of actions

### 4. Components

**Components** are building blocks that give entities and game masters their abilities:
- **Memory Components**: Store and retrieve past events
- **Observation Components**: Process what entities see
- **Act Components**: Generate actions
- **Event Resolution Components**: Convert actions into world changes

### 5. Prefabs

**Prefabs** are pre-built templates for entities and game masters. They combine components in useful ways:
- `basic__Entity`: Simple entity with memory and goal
- `generic__GameMaster`: Standard game master for most scenarios
- `formative_memories_initializer__GameMaster`: Sets up initial shared memories

### 6. Config and Instances

- **Config**: Defines your simulation (premise, max steps, prefabs, instances)
- **Instances**: Specific entities and game masters in your simulation

---

## Your First Game

Let's build a simple game: **Two friends deciding where to eat lunch.**

### Step 1: Import Required Modules

Let's import everything we need. We'll explain each import as we use it:

```python
# Standard library imports
import sentence_transformers  # For creating embeddings
import numpy as np  # For numerical operations
from IPython import display  # For displaying results in Jupyter/Colab

# Concordia imports
from concordia.language_model import utils as language_model_utils
# ^ Provides functions to set up language models (OpenAI, Together AI, etc.)

from concordia.prefabs.simulation import generic as simulation
# ^ The main Simulation class for running games

import concordia.prefabs.entity as entity_prefabs
# ^ Pre-built entity templates (we use 'as' to shorten the name)

import concordia.prefabs.game_master as game_master_prefabs
# ^ Pre-built game master templates

from concordia.typing import prefab as prefab_lib
# ^ Type definitions and classes like InstanceConfig, Role, Config

from concordia.utils import helper_functions
# ^ Utility functions for loading prefabs, formatting output, etc.
```

### Step 2: Set Up Language Model and Embedder

**Using Environment Variables (Recommended):**
```python
import os

# Get API key from environment variable (more secure)
API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    raise ValueError("Set OPENAI_API_KEY environment variable")

API_TYPE = 'openai'
MODEL_NAME = 'gpt-3.5-turbo'  # Start with cheaper model for testing

model = language_model_utils.language_model_setup(
    api_type=API_TYPE,
    model_name=MODEL_NAME,
    api_key=API_KEY,
    disable_language_model=False,  # Set to True for free testing
)

# Set up embedder (converts text to numerical vectors)
st_model = sentence_transformers.SentenceTransformer(
    'sentence-transformers/all-mpnet-base-v2'
)
# Lambda function: shorthand for creating a simple function
# This creates a function that takes text and returns embeddings
embedder = lambda x: st_model.encode(x, show_progress_bar=False)
```

**Alternative: Hardcoded Key (For quick testing only):**
```python
# ⚠️ WARNING: Never commit this to version control!
API_KEY = 'your-api-key-here'  # Replace with your actual key
API_TYPE = 'openai'
MODEL_NAME = 'gpt-3.5-turbo'  # or 'gpt-4' (more expensive)

model = language_model_utils.language_model_setup(
    api_type=API_TYPE,
    model_name=MODEL_NAME,
    api_key=API_KEY,
    disable_language_model=False,
)

# Set up embedder
st_model = sentence_transformers.SentenceTransformer(
    'sentence-transformers/all-mpnet-base-v2'
)
embedder = lambda x: st_model.encode(x, show_progress_bar=False)
```

### Step 3: Load Prefabs

**What are Prefabs?**
- Prefabs are pre-built templates for entities and game masters
- They combine components in useful ways
- Naming convention: `name__Type` (note the double underscore)
  - Examples: `basic__Entity`, `generic__GameMaster`
  - The double underscore separates the name from the type

**Loading Prefabs:**
```python
# get_package_classes() scans a module and finds all Prefab classes
# It returns a dictionary mapping prefab names to Prefab classes
# Example: {'basic__Entity': <class>, 'generic__GameMaster': <class>}

# Load entity prefabs
entity_prefabs_dict = helper_functions.get_package_classes(entity_prefabs)

# Load game master prefabs
gm_prefabs_dict = helper_functions.get_package_classes(game_master_prefabs)

# The ** operator unpacks dictionaries and merges them
# This combines both dictionaries into one
prefabs = {
    **entity_prefabs_dict,
    **gm_prefabs_dict,
}

# Optionally, print available prefabs to see what's available
# display.display(display.Markdown(helper_functions.print_pretty_prefabs(prefabs)))
```

**Understanding Dictionary Unpacking (`**`):**
- The `**` operator unpacks a dictionary into key-value pairs
- `{**dict1, **dict2}` merges two dictionaries
- If there are duplicate keys, the rightmost dictionary's values win
- This is equivalent to: `prefabs = {**entity_prefabs_dict, **gm_prefabs_dict}`

### Step 4: Define Your Entities and Game Master

**Understanding Instances vs Prefabs:**
- **Prefabs** are templates (like blueprints)
- **Instances** are specific entities created from prefabs (like actual objects)
- One prefab can create multiple instances with different parameters
- `InstanceConfig` tells Concordia: "Create an instance using this prefab with these parameters"

**Understanding Roles:**
- `Role.ENTITY`: Regular character/agent in the simulation
- `Role.GAME_MASTER`: Controls the simulation, describes world, resolves actions
- `Role.INITIALIZER`: Sets up initial state (runs once at the start, then hands off to game master)

**The Initializer Explained:**
- The `formative_memories_initializer__GameMaster` runs **once** at the very beginning
- It sets up shared memories that all entities know
- Use it to establish common knowledge, backstory, or initial conditions
- After setup, it hands control to the main game master (specified in `next_game_master_name`)

```python
instances = [
    # Entity 1: Alice
    # InstanceConfig creates a specific entity from the 'basic__Entity' prefab
    prefab_lib.InstanceConfig(
        prefab='basic__Entity',  # Which template to use
        role=prefab_lib.Role.ENTITY,  # This is a regular entity
        params={
            'name': 'Alice',
            'goal': 'You want to go to a fancy Italian restaurant for lunch. You love pasta and are willing to spend more for quality food.',
        },
    ),

    # Entity 2: Bob
    prefab_lib.InstanceConfig(
        prefab='basic__Entity',  # Same prefab, different instance
        role=prefab_lib.Role.ENTITY,
        params={
            'name': 'Bob',
            'goal': 'You prefer a quick, affordable lunch. You like burgers and want to keep costs low.',
        },
    ),

    # Game Master: Controls the simulation
    prefab_lib.InstanceConfig(
        prefab='generic__GameMaster',
        role=prefab_lib.Role.GAME_MASTER,  # This controls the simulation
        params={
            'name': 'Lunch Coordinator',
            'extra_event_resolution_steps': '',  # Use default event resolution
        },
    ),

    # Initializer: Sets up shared memories (runs once at start)
    prefab_lib.InstanceConfig(
        prefab='formative_memories_initializer__GameMaster',
        role=prefab_lib.Role.INITIALIZER,  # Runs once, then hands off to game master
        params={
            'name': 'initial setup',
            'next_game_master_name': 'Lunch Coordinator',  # Who takes over after setup
            'shared_memories': [  # Common knowledge all entities share
                'Alice and Bob are friends who work together.',
                'It is lunchtime and they need to decide where to eat.',
                'They are in downtown and have many restaurant options nearby.',
            ],
        },
    ),
]
```

### Step 5: Create the Configuration

```python
config = prefab_lib.Config(
    default_premise='Alice and Bob are trying to decide where to have lunch together.',
    default_max_steps=10,  # Maximum number of turns
    prefabs=prefabs,
    instances=instances,
)
```

### Step 6: Initialize and Run the Simulation

```python
# Create the simulation
sim = simulation.Simulation(
    config=config,
    model=model,
    embedder=embedder,
)

# Run the simulation
results_log = sim.play(max_steps=10)

# Display the results (in Jupyter/Colab)
display.HTML(results_log)
```

### Running Your Code

**In Jupyter Notebook or Google Colab:**
- Copy the code into cells
- Run each cell in order
- The HTML output will display automatically

**In a Python Script:**
```python
# Save results to HTML file
results_log = sim.play(max_steps=10)
with open('simulation_results.html', 'w') as f:
    f.write(results_log)
print("Results saved to simulation_results.html")
```

**Command Line:**
```bash
python your_simulation.py
```

### Understanding the Output

The `sim.play()` method returns an HTML log that includes:
- **Game Master Log**: All events that occurred, in chronological order
- **Entity Memories**: What each entity remembers
- **Game Master Memories**: The game master's memory of all events

**Interpreting the HTML Output:**
- Open the HTML file in a browser (or view in Jupyter/Colab)
- The output has tabs for different views:
  - **Game Master log**: Main timeline of events
  - **Entity tabs**: Each entity's personal memory
  - **Game Master Memories**: Complete event history
- Each step shows:
  - What each entity observed
  - What action each entity took
  - How the game master resolved the action
  - The resulting world state

**Accessing Raw Log Data:**
```python
raw_log = []
results_log = sim.play(max_steps=10, raw_log=raw_log)
# raw_log is a list of dictionaries with detailed step-by-step information

# Example: Print summary of each step
for step_data in raw_log:
    step_num = step_data.get('Step', 'Unknown')
    summary = step_data.get('Summary', 'No summary')
    print(f"Step {step_num}: {summary}")
```

**Getting Non-HTML Output:**
```python
# Get raw log instead of HTML
raw_log = []
results = sim.play(max_steps=10, raw_log=raw_log, return_html_log=False)
# Now 'results' is the raw_log list, not HTML
```

### Complete Example

Here's the complete code in one block:

```python
import sentence_transformers
import numpy as np
from IPython import display

from concordia.language_model import utils as language_model_utils
from concordia.prefabs.simulation import generic as simulation
import concordia.prefabs.entity as entity_prefabs
import concordia.prefabs.game_master as game_master_prefabs
from concordia.typing import prefab as prefab_lib
from concordia.utils import helper_functions

# Setup
API_KEY = 'your-api-key-here'
API_TYPE = 'openai'
MODEL_NAME = 'gpt-4'

model = language_model_utils.language_model_setup(
    api_type=API_TYPE,
    model_name=MODEL_NAME,
    api_key=API_KEY,
)

st_model = sentence_transformers.SentenceTransformer(
    'sentence-transformers/all-mpnet-base-v2'
)
embedder = lambda x: st_model.encode(x, show_progress_bar=False)

# Load prefabs
prefabs = {
    **helper_functions.get_package_classes(entity_prefabs),
    **helper_functions.get_package_classes(game_master_prefabs),
}

# Define instances
instances = [
    prefab_lib.InstanceConfig(
        prefab='basic__Entity',
        role=prefab_lib.Role.ENTITY,
        params={'name': 'Alice', 'goal': 'You want to go to a fancy Italian restaurant.'},
    ),
    prefab_lib.InstanceConfig(
        prefab='basic__Entity',
        role=prefab_lib.Role.ENTITY,
        params={'name': 'Bob', 'goal': 'You prefer a quick, affordable lunch.'},
    ),
    prefab_lib.InstanceConfig(
        prefab='generic__GameMaster',
        role=prefab_lib.Role.GAME_MASTER,
        params={'name': 'Lunch Coordinator'},
    ),
    prefab_lib.InstanceConfig(
        prefab='formative_memories_initializer__GameMaster',
        role=prefab_lib.Role.INITIALIZER,
        params={
            'name': 'initial setup',
            'next_game_master_name': 'Lunch Coordinator',
            'shared_memories': ['Alice and Bob are friends deciding where to eat lunch.'],
        },
    ),
]

# Create config
config = prefab_lib.Config(
    default_premise='Alice and Bob are trying to decide where to have lunch.',
    default_max_steps=10,
    prefabs=prefabs,
    instances=instances,
)

# Run simulation
sim = simulation.Simulation(config=config, model=model, embedder=embedder)
results_log = sim.play(max_steps=10)
display.HTML(results_log)
```

---

## Understanding the Game Loop

Let's understand what happens when you run `sim.play()`:

### The Sequential Engine Flow

1. **Initialization**: The initializer game master sets up shared memories
2. **Game Loop** (repeats until termination or max_steps):
   - **Make Observations**: Game master creates observations for all entities
   - **Next Acting**: Game master decides which entity acts next
   - **Entity Acts**: The chosen entity generates an action
   - **Resolve**: Game master converts the action into a world event
   - **Check Termination**: Game master decides if the game should end
3. **Output**: Returns an HTML log of the entire simulation

### Key Methods in the Engine

- `make_observation()`: Creates what each entity sees
- `next_acting()`: Chooses which entity acts next
- `resolve()`: Converts entity actions into world events
- `terminate()`: Decides if the simulation should end

### Understanding Observations

Entities receive observations that describe:
- What happened in previous turns
- The current state of the world
- What they can see/hear/know

The game master crafts these observations based on:
- Past events
- Entity memories
- The game master's understanding of the world

### Understanding Actions

Entities respond with actions in natural language, like:
- "I suggest we go to the Italian restaurant on Main Street"
- "I'd prefer something faster and cheaper"
- "Let's compromise and find a mid-range place"

The game master then resolves these into concrete events:
- "Alice suggests going to Bella Italia, an upscale Italian restaurant"
- "Bob expresses concern about the cost and time"
- "They agree to look for a compromise option"

---

## Customizing Entities

### Entity Parameters

The `basic__Entity` prefab accepts these parameters:

```python
prefab_lib.InstanceConfig(
    prefab='basic__Entity',
    role=prefab_lib.Role.ENTITY,
    params={
        'name': 'CharacterName',
        'goal': 'What this character wants to achieve',
        'randomize_choices': True,  # Whether to add randomness to decisions
    },
)
```

### Writing Effective Goals

Good goals are:
- **Specific**: "Win the negotiation" vs "Do well"
- **Actionable**: "Convince others to choose your option"
- **Contextual**: Include relevant constraints or preferences

Examples:
```python
# Good goal
'goal': 'You are a competitive negotiator. Your goal is to get the best deal possible while maintaining a friendly relationship. You value long-term partnerships over short-term gains.'

# Less effective goal
'goal': 'Be nice and get what you want'  # Too vague
```

### Available Entity Prefabs

Explore available prefabs:

```python
# Print all available entity prefabs
prefabs = helper_functions.get_package_classes(entity_prefabs)
for name, prefab in prefabs.items():
    print(f"{name}: {prefab.description}")
```

Common entity prefabs:
- `basic__Entity`: Standard entity with memory and goal
- `basic_with_plan__Entity`: Entity that creates and follows plans

---

## Customizing Game Masters

### Game Master Parameters

The `generic__GameMaster` prefab accepts:

```python
prefab_lib.InstanceConfig(
    prefab='generic__GameMaster',
    role=prefab_lib.Role.GAME_MASTER,
    params={
        'name': 'GameMasterName',
        'extra_event_resolution_steps': '',  # Comma-separated thought chain steps
        'acting_order': 'game_master_choice',  # 'game_master_choice', 'fixed', or 'random'
        'extra_components': {},  # Additional custom components
        'extra_components_index': {},  # Where to insert extra components
    },
)
```

### Acting Order Options

- `'game_master_choice'`: Game master decides who acts next (default, most flexible)
- `'fixed'`: Entities act in a fixed order (round-robin)
- `'random'`: Random order each turn

Example with fixed order:
```python
prefab_lib.InstanceConfig(
    prefab='generic__GameMaster',
    role=prefab_lib.Role.GAME_MASTER,
    params={
        'name': 'Coordinator',
        'acting_order': 'fixed',  # Entities will act in the order they're defined
    },
)
```

### Event Resolution Steps

You can customize how the game master resolves events by adding thought chain steps:

```python
params={
    'extra_event_resolution_steps': 'maybe_cut_to_next_scene,result_to_who_what_where',
}
```

Available steps (from `concordia.thought_chains.thought_chains`):
- `maybe_inject_narrative_push`: Adds narrative tension
- `maybe_cut_to_next_scene`: Transitions between scenes
- `result_to_who_what_where`: Structures events clearly

### Specialized Game Masters

Explore specialized game masters for specific scenarios:

```python
# Print available game master prefabs
gm_prefabs = helper_functions.get_package_classes(game_master_prefabs)
for name, prefab in gm_prefabs.items():
    print(f"{name}: {prefab.description}")
```

Examples:
- `dialogic__GameMaster`: For conversation-focused simulations
- `marketplace__GameMaster`: For economic/trading simulations
- `scripted__GameMaster`: For pre-scripted interactions

---

## Working with Engines

### Sequential Engine (Default)

Turn-based: one entity acts at a time.

```python
from concordia.environment.engines import sequential

engine = sequential.Sequential()
sim = simulation.Simulation(
    config=config,
    model=model,
    embedder=embedder,
    engine=engine,  # Explicitly set (this is the default)
)
```

### Simultaneous Engine

All entities act at the same time, then events are resolved.

```python
from concordia.environment.engines import simultaneous

engine = simultaneous.Simultaneous()
sim = simulation.Simulation(
    config=config,
    model=model,
    embedder=embedder,
    engine=engine,
)
```

**When to use Simultaneous:**
- Auctions or markets where everyone bids at once
- Voting scenarios
- Any scenario where timing doesn't matter

**When to use Sequential:**
- Negotiations (order matters)
- Conversations (turn-taking)
- Most narrative scenarios

### Configuration Engine

Pre-configured sequence of actions (advanced).

```python
from concordia.environment.engines import configuration

# Define a sequence of actions
# (See configuration.py for details)
engine = configuration.Configuration(...)
```

---

## Advanced Topics

### Custom Components

You can create custom components to add new behaviors. Components are classes that implement specific interfaces.

**Example: Custom Observation Component**

```python
from concordia.components import agent as agent_components
from concordia.typing import entity_component

class CustomObservation(agent_components.observation.ObservationComponent):
    """A custom observation component that adds weather information."""

    def get_state(self) -> entity_component.ComponentState:
        return {'weather': 'sunny'}

    def set_state(self, state: entity_component.ComponentState) -> None:
        pass

    def observe(self, observation: str) -> None:
        # Process observations
        pass

    def __call__(self) -> str:
        # Return formatted observation text
        return "The weather is sunny today."
```

### Custom Prefabs

Create reusable entity or game master templates:

```python
import dataclasses
from concordia.typing import prefab as prefab_lib

@dataclasses.dataclass
class MyCustomEntity(prefab_lib.Prefab):
    """A custom entity prefab."""

    description: str = 'A custom entity with special abilities'
    params: dict = dataclasses.field(default_factory=lambda: {
        'name': 'CustomEntity',
        'special_param': 'value',
    })

    def build(self, model, memory_bank):
        # Build and return an EntityAgent
        # (See basic.py for examples)
        pass
```

### Accessing Simulation Data

After running a simulation, you can access:

```python
# Get entities
entities = sim.get_entities()
for entity in entities:
    print(f"Entity: {entity.name}")

# Get game masters
game_masters = sim.get_game_masters()
for gm in game_masters:
    print(f"Game Master: {gm.name}")

# Access components (if using EntityAgentWithLogging)
if hasattr(entity, 'get_component'):
    memory = entity.get_component('Memory')
    # Access component state, etc.
```

### Logging and Debugging

Enable verbose output:

```python
# In the engine's run_loop, set verbose=True
# Or modify the engine to always be verbose

# Access raw logs
raw_log = []
results_log = sim.play(max_steps=10, raw_log=raw_log)
# raw_log contains detailed information about each step
```

### Memory System

Entities use associative memory to remember past events:

- **Automatic Storage**: Observations are automatically stored
- **Retrieval**: Relevant memories are retrieved based on current context
- **Similarity Search**: Uses embeddings to find similar past events

You can influence memory:
- Add shared memories via the initializer
- Entities remember their own observations automatically
- Game masters have access to all events

### Saving and Loading Checkpoints

You can save the simulation state at any point and resume later:

**Saving a Checkpoint:**
```python
# Run simulation with checkpoint saving
results_log = sim.play(
    max_steps=10,
    checkpoint_path='./checkpoints',  # Directory to save checkpoints
)

# Or manually save at a specific step
sim.save_checkpoint(step=5, checkpoint_path='./checkpoints')
```

**Loading from a Checkpoint:**
```python
import json

# Load checkpoint data
with open('./checkpoints/step_5_checkpoint.json', 'r') as f:
    checkpoint_data = json.load(f)

# Load into simulation
sim.load_from_checkpoint(checkpoint_data)

# Continue from where you left off
results_log = sim.play(max_steps=15)  # Continue for 15 more steps
```

### Extracting Data from Simulations

**Access Entity Memories:**
```python
# After running simulation
for entity in sim.get_entities():
    if hasattr(entity, 'get_component'):
        memory_component = entity.get_component('__memory__')
        if memory_component:
            all_memories = memory_component.get_all_memories_as_text()
            print(f"{entity.name}'s memories:")
            for memory in all_memories:
                print(f"  - {memory}")
```

**Access Raw Log Data:**
```python
raw_log = []
results_log = sim.play(max_steps=10, raw_log=raw_log)

# raw_log is a list of step dictionaries
for step in raw_log:
    print(f"Step {step.get('Step', 'Unknown')}:")
    print(f"  Summary: {step.get('Summary', 'N/A')}")
    # Access entity actions, game master decisions, etc.
```

**Extract Specific Information:**
```python
from concordia.utils import helper_functions as helper_funcs

# Find specific data in the nested log structure
scores = helper_funcs.find_data_in_nested_structure(raw_log, "Player Scores")
events = helper_funcs.find_data_in_nested_structure(raw_log, "Event")

print("Scores:", scores)
print("Events:", events)
```

**Access Component Data:**
```python
# If using specialized game masters with data components
# (e.g., marketplace game master with trade history)
if hasattr(sim.game_masters[0], 'get_component'):
    component = sim.game_masters[0].get_component('component_name')
    if hasattr(component, 'get_data'):
        data = component.get_data()
        # Process the data (e.g., pandas DataFrame for marketplace)
```

---

## Creating Custom Components

Components are the building blocks that give entities and game masters their abilities. This section explains the component architecture and how to create your own custom components.

### Understanding Component Architecture

#### Component Hierarchy

All components inherit from base classes in the `concordia.typing.entity_component` module:

1. **`BaseComponent`**: The fundamental base class for all components
2. **`ContextComponent`**: For components that provide context during actions/observations
3. **`ActingComponent`**: For components that decide what action to take
4. **`ComponentWithLogging`**: Adds logging capabilities to components

#### Component Lifecycle: Phases

Components operate in a specific lifecycle with phases:

```python
from concordia.typing.entity_component import Phase

# Phase flow:
# READY → PRE_ACT → POST_ACT → UPDATE → READY
#   ↓
# PRE_OBSERVE → POST_OBSERVE → UPDATE → READY
```

**Phase Descriptions:**
- **`READY`**: Component is ready to observe or act
- **`PRE_ACT`**: Entity is about to act; components provide context
- **`POST_ACT`**: Entity just acted; components are informed
- **`PRE_OBSERVE`**: Entity received an observation; components process it
- **`POST_OBSERVE`**: After observation processing; components can provide context
- **`UPDATE`**: Components update their internal state

#### Base Component Methods

Every component must implement these methods. Here's a detailed explanation of each:

##### `get_state() -> ComponentState`

**Purpose**: Serializes the component's internal state into a dictionary that can be saved to disk (e.g., for checkpoints) or transferred between components.

**Key Points**:
- Returns a `ComponentState`, which is a dictionary containing only JSON-serializable types (strings, numbers, lists, dicts, booleans, None)
- Should only include **mutable state** that changes during the simulation
- Should **NOT** include:
  - Parameters passed to `__init__` (e.g., memory banks, model references, component names)
  - References to other objects (convert to serializable data instead)
  - Thread locks or other non-serializable objects
- Used for checkpointing, creating component copies, and state restoration

**Example**:
```python
def get_state(self) -> ComponentState:
    """Returns the component's state for serialization."""
    with self._lock:  # Use locks if accessing shared state
        return {
            'counter': self._count,           # Simple value
            'history': list(self._history),   # Convert sets/lists to lists
            'last_update': self._timestamp.isoformat(),  # Convert datetime to string
            'config': {
                'threshold': self._threshold,
                'enabled': self._enabled
            }
        }
```

**When to use**:
- Saving checkpoints during long simulations
- Creating identical component copies
- Debugging by inspecting component state
- Transferring state between simulation runs

##### `set_state(state: ComponentState) -> None`

**Purpose**: Restores the component's internal state from a previously saved state dictionary.

**Key Points**:
- Receives the same dictionary structure that `get_state()` returned
- Should restore all mutable state that was saved
- Must be called **after** the component is initialized with the same parameters as when `get_state()` was called
- Should handle missing keys gracefully (use `.get()` with defaults)
- Must convert serialized data back to original types (e.g., strings back to datetime objects)

**Example**:
```python
def set_state(self, state: ComponentState) -> None:
    """Restores the component's state from serialization."""
    with self._lock:
        self._count = state.get('counter', 0)  # Use defaults for missing keys
        self._history = set(state.get('history', []))  # Convert back to set
        if 'last_update' in state:
            self._timestamp = datetime.fromisoformat(state['last_update'])
        config = state.get('config', {})
        self._threshold = config.get('threshold', 10.0)
        self._enabled = config.get('enabled', True)
```

**Important Notes**:
- The component must be initialized with the same constructor parameters before calling `set_state()`
- Static/immutable data (like memory bank references) should be passed in `__init__`, not saved in state
- Always use thread locks when modifying state in multi-threaded contexts

**When to use**:
- Loading checkpoints to resume simulations
- Creating component copies with identical state
- Testing by resetting components to known states

##### `get_entity() -> EntityWithComponents`

**Purpose**: Returns a reference to the entity that owns this component.

**Key Points**:
- The entity is automatically set by the framework when the component is added to an entity
- Use this method to access other components on the same entity
- Raises `RuntimeError` if called before the component is attached to an entity
- Never call this in `__init__` - the entity is set after initialization

**Example**:
```python
def _make_pre_act_value(self) -> str:
    """Access other components through the entity."""
    # Get the memory component
    memory = self.get_entity().get_component(
        '__memory__',
        type_=memory_component.Memory
    )

    # Get another custom component
    goal_component = self.get_entity().get_component('Goal')
    goal_text = goal_component.get_pre_act_value()

    # Access entity properties
    entity_name = self.get_entity().name

    return f"{entity_name} working on: {goal_text}"
```

**Common Use Cases**:
- Accessing memory components to read/write memories
- Reading state from other components
- Getting entity metadata (name, phase, etc.)
- Coordinating behavior between components

**Important**:
- Always check the entity phase when accessing other components (some operations are not allowed during `UPDATE` phase)
- Use type hints when getting components to ensure type safety:
  ```python
  memory = self.get_entity().get_component(
      '__memory__',
      type_=memory_component.Memory  # Ensures correct type
  )
  ```

**Complete Example**:

Here's a complete example showing all three methods working together:

```python
import threading
from datetime import datetime
from concordia.typing.entity_component import BaseComponent, ComponentState
from concordia.components.agent import memory as memory_component

class ActivityTracker(BaseComponent):
    """Tracks entity activity with timestamps."""

    def __init__(self):
        self._activities = []  # List of (timestamp, activity) tuples
        self._lock = threading.Lock()
        self._last_activity_time = None

    def record_activity(self, activity: str):
        """Record a new activity."""
        with self._lock:
            now = datetime.now()
            self._activities.append((now, activity))
            self._last_activity_time = now

    def get_state(self) -> ComponentState:
        """Serialize state for checkpointing."""
        with self._lock:
            # Convert datetime objects to ISO format strings for JSON serialization
            return {
                'activities': [
                    {
                        'timestamp': ts.isoformat(),
                        'activity': activity
                    }
                    for ts, activity in self._activities
                ],
                'last_activity_time': (
                    self._last_activity_time.isoformat()
                    if self._last_activity_time else None
                ),
                'count': len(self._activities)
            }

    def set_state(self, state: ComponentState) -> None:
        """Restore state from checkpoint."""
        with self._lock:
            # Convert ISO strings back to datetime objects
            self._activities = [
                (
                    datetime.fromisoformat(item['timestamp']),
                    item['activity']
                )
                for item in state.get('activities', [])
            ]
            last_time = state.get('last_activity_time')
            self._last_activity_time = (
                datetime.fromisoformat(last_time)
                if last_time else None
            )

    def get_entity(self):
        """Access the owning entity."""
        return super().get_entity()  # Uses BaseComponent's implementation

    def get_recent_activities(self, limit: int = 5) -> list:
        """Get recent activities using entity's memory."""
        # Example of using get_entity() to access other components
        memory = self.get_entity().get_component(
            '__memory__',
            type_=memory_component.Memory
        )
        # Could cross-reference with memory if needed
        with self._lock:
            return self._activities[-limit:]
```

### Component Types

#### 1. ContextComponent

`ContextComponent` provides context during actions and observations. It has these hook methods:

```python
from concordia.typing.entity_component import ContextComponent

class MyContextComponent(ContextComponent):
    def pre_act(self, action_spec) -> str:
        """Called before entity acts. Returns context string."""
        return "Context for action"

    def post_act(self, action_attempt: str) -> str:
        """Called after entity acts. Can process the action."""
        return ""

    def pre_observe(self, observation: str) -> str:
        """Called when entity receives observation."""
        return ""

    def post_observe(self) -> str:
        """Called after observation is processed."""
        return ""

    def update(self) -> None:
        """Called after post_act or post_observe to update state."""
        pass
```

**Example: Simple Constant Component**

```python
from concordia.components.agent import action_spec_ignored
from concordia.typing import entity_component

class MyConstantComponent(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging
):
    """A component that provides a constant value."""

    def __init__(self, value: str, label: str = "My Constant"):
        super().__init__(label)
        self._value = value

    def _make_pre_act_value(self) -> str:
        """Returns the constant value."""
        return self._value

    def get_state(self) -> entity_component.ComponentState:
        return {}

    def set_state(self, state: entity_component.ComponentState) -> None:
        pass
```

**Example: Dynamic Component Using Other Components**

```python
from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.typing import entity_component

class RecentMemoriesSummary(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging
):
    """Summarizes recent memories."""

    def __init__(
        self,
        memory_component_key: str = '__memory__',
        num_memories: int = 5,
        label: str = "Recent Memories"
    ):
        super().__init__(label)
        self._memory_key = memory_component_key
        self._num_memories = num_memories

    def _make_pre_act_value(self) -> str:
        """Retrieves and formats recent memories."""
        memory = self.get_entity().get_component(
            self._memory_key,
            type_=memory_component.Memory
        )
        recent = memory.retrieve_recent(limit=self._num_memories)
        return "\n".join(f"- {mem}" for mem in recent)

    def get_state(self) -> entity_component.ComponentState:
        return {}

    def set_state(self, state: entity_component.ComponentState) -> None:
        pass
```

#### 2. ActingComponent

`ActingComponent` decides what action the entity takes. Only one acting component should exist per entity.

```python
from concordia.typing.entity_component import ActingComponent, ComponentContextMapping
from concordia.typing import entity as entity_lib

class MyActingComponent(ActingComponent):
    """Decides the entity's action."""

    def __init__(self, model):
        self._model = model

    def get_action_attempt(
        self,
        context: ComponentContextMapping,
        action_spec: entity_lib.ActionSpec
    ) -> str:
        """Returns the action the entity should take."""
        # Combine context from all components
        context_str = "\n".join(
            f"{name}:\n{ctx}"
            for name, ctx in context.items()
        )

        # Use language model to generate action
        prompt = f"{context_str}\n\n{action_spec.instruction}"
        return self._model.sample_text(prompt)

    def get_state(self) -> entity_component.ComponentState:
        return {}

    def set_state(self, state: entity_component.ComponentState) -> None:
        pass
```

#### 3. Component with State Management

Components that maintain state should properly implement `get_state` and `set_state`:

```python
import threading
from concordia.typing.entity_component import ContextComponent, ComponentState

class CounterComponent(ContextComponent):
    """A component that counts observations."""

    def __init__(self):
        self._count = 0
        self._lock = threading.Lock()

    def pre_observe(self, observation: str) -> str:
        """Increment counter on each observation."""
        with self._lock:
            self._count += 1
        return ""

    def pre_act(self, action_spec) -> str:
        """Report the count."""
        with self._lock:
            return f"Total observations: {self._count}"

    def get_state(self) -> ComponentState:
        """Save the counter state."""
        with self._lock:
            return {'count': self._count}

    def set_state(self, state: ComponentState) -> None:
        """Restore the counter state."""
        with self._lock:
            self._count = state.get('count', 0)

    def update(self) -> None:
        """Called after post_act/post_observe."""
        pass
```

### Understanding Multiple Inheritance Syntax

Many Concordia components use **multiple inheritance** to combine functionality. Here's how the syntax works:

```python
class MoodComponent(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging
):
    # Class body here
```

**Syntax Breakdown:**

1. **Multiple Parent Classes**: The parentheses contain multiple base classes separated by commas
   - `action_spec_ignored.ActionSpecIgnored` - First parent class
   - `entity_component.ComponentWithLogging` - Second parent class

2. **Inheritance Order Matters**: Python uses Method Resolution Order (MRO) to determine which parent's method to call when there are conflicts. The order is left-to-right:
   - Methods from `ActionSpecIgnored` are checked first
   - Then methods from `ComponentWithLogging`
   - Finally, common ancestors (like `BaseComponent`)

3. **What Each Parent Provides**:
   - **`ActionSpecIgnored`**:
     - Simplifies `pre_act()` implementation
     - Provides `_make_pre_act_value()` pattern
     - Handles action spec caching automatically
     - Inherits from `ContextComponent`

   - **`ComponentWithLogging`**:
     - Adds `_logging_channel` attribute
     - Provides logging infrastructure
     - Inherits from `BaseComponent`

4. **How `super()` Works**: When you call `super().__init__(label)`, Python follows the MRO:
   ```python
   def __init__(self, label: str):
       super().__init__(label)  # Calls ActionSpecIgnored.__init__(label)
       # Then ComponentWithLogging.__init__() is called automatically
   ```

5. **Inheritance Chain**: The full inheritance hierarchy looks like:
   ```
   MoodComponent
   ├── ActionSpecIgnored
   │   └── ContextComponent
   │       └── BaseComponent
   └── ComponentWithLogging
       └── BaseComponent
   ```
   Since both share `BaseComponent`, it's only included once (diamond inheritance is handled by Python's MRO).

**Why Use Multiple Inheritance Here?**

- **Separation of Concerns**: Each parent class provides a specific capability
- **Code Reuse**: Don't duplicate logging or pre_act logic
- **Flexibility**: Mix and match capabilities as needed
- **Standard Pattern**: This is the recommended pattern for most context components in Concordia

**Alternative (Without Multiple Inheritance)**:
```python
# You could inherit from just ContextComponent and implement everything yourself:
class MoodComponent(entity_component.ContextComponent):
    def __init__(self, label: str):
        super().__init__()
        # Would need to implement all the ActionSpecIgnored logic manually
        # Would need to implement logging manually
        # Much more code!
```

### Complete Example: Custom Mood Component

Here's a complete example of a custom component that tracks an entity's mood:

```python
import threading
from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory as memory_component
from concordia.typing import entity_component

class MoodComponent(
    action_spec_ignored.ActionSpecIgnored,
    entity_component.ComponentWithLogging
):
    """Tracks and reports entity mood based on recent memories."""

    def __init__(
        self,
        memory_component_key: str = '__memory__',
        label: str = "Current Mood"
    ):
        super().__init__(label)
        self._memory_key = memory_component_key
        self._lock = threading.Lock()
        self._mood = "neutral"

    def _make_pre_act_value(self) -> str:
        """Analyzes recent memories to determine mood."""
        memory = self.get_entity().get_component(
            self._memory_key,
            type_=memory_component.Memory
        )

        # Get recent memories
        recent = memory.retrieve_recent(limit=5)

        # Simple mood detection (you could use LLM for more sophisticated analysis)
        positive_words = ['happy', 'excited', 'pleased', 'great', 'wonderful']
        negative_words = ['sad', 'angry', 'frustrated', 'disappointed', 'worried']

        text = ' '.join(recent).lower()
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)

        with self._lock:
            if positive_count > negative_count:
                self._mood = "positive"
            elif negative_count > positive_count:
                self._mood = "negative"
            else:
                self._mood = "neutral"

            return f"Current mood: {self._mood}"

    def get_state(self) -> entity_component.ComponentState:
        """Save mood state."""
        with self._lock:
            return {'mood': self._mood}

    def set_state(self, state: entity_component.ComponentState) -> None:
        """Restore mood state."""
        with self._lock:
            self._mood = state.get('mood', 'neutral')
```

### Using Custom Components in Prefabs

To use your custom component in a prefab:

```python
import dataclasses
from concordia.typing import prefab as prefab_lib
from concordia.components.agent import memory as memory_component
from concordia.associative_memory import basic_associative_memory

@dataclasses.dataclass
class EntityWithMood(prefab_lib.Prefab):
    """An entity with a custom mood component."""

    description: str = 'An entity that tracks its mood'

    params: dict = dataclasses.field(default_factory=lambda: {
        'name': 'Alice',
        'goal': 'Be happy and help others',
    })

    def build(
        self,
        model,
        memory_bank: basic_associative_memory.AssociativeMemoryBank,
    ):
        from concordia.prefabs.entity import basic as basic_entity

        # Build a basic entity first
        entity = basic_entity.Entity().build(model, memory_bank)

        # Add your custom component
        mood_component = MoodComponent()
        entity.add_component('mood', mood_component)

        return entity
```

### Best Practices for Custom Components

1. **Thread Safety**: Use locks when accessing shared state:
   ```python
   self._lock = threading.Lock()
   with self._lock:
       # modify state
   ```

2. **Phase Checking**: Check the current phase when accessing other components:
   ```python
   if self.get_entity().get_phase() == entity_component.Phase.UPDATE:
       raise ValueError("Cannot access memory during UPDATE phase")
   ```

3. **State Management**: Always implement `get_state` and `set_state` for checkpointing:
   ```python
   def get_state(self) -> ComponentState:
       return {'key': self._value}

   def set_state(self, state: ComponentState) -> None:
       self._value = state.get('key', default_value)
   ```

4. **Component Access**: Use `get_entity().get_component()` to access other components:
   ```python
   memory = self.get_entity().get_component(
       '__memory__',
       type_=memory_component.Memory
   )
   ```

5. **Logging**: Inherit from `ComponentWithLogging` for built-in logging:
   ```python
   self._logging_channel({'Key': 'MyComponent', 'Value': data})
   ```

6. **Action Spec Ignored**: For simple context components, inherit from `ActionSpecIgnored`:
   ```python
   class MyComponent(
       action_spec_ignored.ActionSpecIgnored,
       entity_component.ComponentWithLogging
   ):
       def _make_pre_act_value(self) -> str:
           return "My context"
   ```

### Component Interaction Patterns

**Pattern 1: Component reads from another component**
```python
def _make_pre_act_value(self) -> str:
    goal_component = self.get_entity().get_component('Goal')
    goal_value = goal_component.get_pre_act_value()
    return f"Working towards: {goal_value}"
```

**Pattern 2: Component writes to memory**
```python
def pre_observe(self, observation: str) -> str:
    memory = self.get_entity().get_component('__memory__')
    memory.add(f"Important: {observation}")
    return ""
```

**Pattern 3: Component processes observations**
```python
def pre_observe(self, observation: str) -> str:
    # Process observation
    if "urgent" in observation.lower():
        self._has_urgent = True
    return ""
```

---

## Best Practices

### 1. Start Simple

Begin with 2-3 entities and a simple scenario. Add complexity gradually.

**Progression:**
- Week 1: 2 entities, basic conversation
- Week 2: 3 entities, add goals and constraints
- Week 3: Custom components or specialized game masters
- Week 4: Complex scenarios with multiple game masters

### 2. Write Clear Goals

Entity goals should be:
- Specific and actionable
- Include relevant constraints
- Match the scenario's tone

### 3. Use Appropriate Premises

Your premise sets the initial context:
```python
premise = """
You are at a coffee shop. It's a quiet Tuesday afternoon.
The barista is friendly and the atmosphere is relaxed.
"""
```

### 4. Set Reasonable Max Steps

- Too few: Simulation ends before resolution
- Too many: Wastes API calls and time
- Start with 5-10 steps, adjust based on your scenario

### 5. Test with DISABLE_LANGUAGE_MODEL

Before spending money on API calls:
```python
model = language_model_utils.language_model_setup(
    api_type='openai',
    model_name='gpt-4',
    api_key='',
    disable_language_model=True,  # Free testing
)
```

### 6. Use Shared Memories Wisely

Shared memories establish common knowledge:
```python
'shared_memories': [
    'All characters know each other from work',
    'They are meeting at a neutral location',
    'The meeting is scheduled for 30 minutes',
]
```

### 7. Choose the Right Engine

- **Sequential**: Most scenarios, conversations, negotiations
- **Simultaneous**: Markets, auctions, voting

### 8. Monitor API Costs

LLM API calls can be expensive. Monitor:
- Number of steps
- Number of entities (more entities = more observations per step)
- Model choice (GPT-4 is more expensive than GPT-3.5)

**Cost Estimation:**
- Each step makes multiple API calls (observations + actions + resolution)
- With 3 entities and 10 steps: ~30-50 API calls
- GPT-4: ~$0.10-0.50 per simulation
- GPT-3.5: ~$0.01-0.05 per simulation
- Test with `DISABLE_LANGUAGE_MODEL=True` to avoid costs during development

### 9. Debugging Tips

**Enable Verbose Output:**
```python
# The engine's run_loop has verbose parameter
# You can modify the engine or check the raw_log for details
raw_log = []
results = sim.play(max_steps=5, raw_log=raw_log)
# Inspect raw_log to see what happened at each step
```

**Check Entity States:**
```python
# After running, check what entities remember
for entity in sim.get_entities():
    if hasattr(entity, 'get_component'):
        memory = entity.get_component('__memory__')
        if memory:
            print(f"{entity.name} has {len(memory.get_all_memories_as_text())} memories")
```

**Validate Configuration:**
```python
# Before running, verify your config
print(f"Entities: {[e.name for e in sim.get_entities()]}")
print(f"Game Masters: {[gm.name for gm in sim.get_game_masters()]}")
print(f"Max steps: {config.default_max_steps}")
print(f"Premise: {config.default_premise}")
```

---

## Troubleshooting

### Problem: "No game masters provided"

**Solution**: Make sure you have at least one game master in your instances:
```python
prefab_lib.InstanceConfig(
    prefab='generic__GameMaster',
    role=prefab_lib.Role.GAME_MASTER,
    params={'name': 'GM'},
)
```

### Problem: Entities don't remember things

**Solution**:
- Check that you're using `EntityAgentWithLogging` (most prefabs do)
- Ensure the memory component is included (it is in `basic__Entity`)
- Verify observations are being made (check verbose output)

### Problem: Simulation ends too quickly

**Solution**:
- Increase `max_steps`
- Make entity goals more complex (require multiple steps)
- Add obstacles or constraints to the scenario

### Problem: API errors or rate limits

**Solution**:
- Add retry logic (some model wrappers have this)
- Use a model with higher rate limits
- Reduce number of entities or steps
- Add delays between API calls

### Problem: Entities act inconsistently

**Solution**:
- Set `randomize_choices=False` for deterministic behavior
- Provide more specific goals
- Use a more capable model (GPT-4 vs GPT-3.5)

### Problem: Game master makes poor decisions

**Solution**:
- Use a better model
- Add more context via shared memories
- Customize event resolution steps
- Provide clearer instructions in the premise

### Problem: Simulation is too slow

**Solution**:
- Use a faster model (GPT-3.5 vs GPT-4)
- Reduce number of entities
- Use simultaneous engine if order doesn't matter
- Reduce `max_steps`

### Problem: Output is hard to read or understand

**Solution**:
- Open the HTML file in a browser for better formatting
- Use `verbose=True` in the engine to see step-by-step progress
- Access `raw_log` for programmatic analysis
- Focus on the "Game Master log" tab in HTML output

### Common Mistakes to Avoid

1. **Forgetting the Initializer**: Always include a `formative_memories_initializer__GameMaster` to set up shared context
2. **Vague Goals**: Entity goals should be specific and actionable
3. **Too Many Entities**: Start with 2-3 entities, add more gradually
4. **Insufficient Max Steps**: Complex scenarios need more steps to resolve
5. **Missing API Key**: Set up your language model API key before running
6. **Wrong Engine Type**: Use sequential for conversations, simultaneous for markets
7. **Not Testing First**: Always test with `DISABLE_LANGUAGE_MODEL=True` first

---

## Next Steps

### Explore Examples

Check out the `examples/` directory:
- `tutorial.ipynb`: Basic tutorial (similar to this guide)
- `marketplace.ipynb`: Economic simulation example
- `dialog.ipynb`: Conversation-focused example
- `selling_cookies.ipynb`: More complex scenario

### Read the Documentation

- **README.md**: Overview and installation
- **Code Documentation**: Read docstrings in the source code
- **Tech Report**: [arXiv paper](https://arxiv.org/abs/2312.03664) for theoretical background

### Experiment

Try building:
1. **A negotiation game**: Two parties trying to reach a deal
2. **A group decision**: Multiple people choosing a restaurant
3. **A market simulation**: Buyers and sellers trading goods
4. **A narrative game**: Characters in a story making choices

### Example Scenarios

**Scenario 1: Roommate Dispute**
```python
instances = [
    prefab_lib.InstanceConfig(
        prefab='basic__Entity',
        role=prefab_lib.Role.ENTITY,
        params={
            'name': 'Alex',
            'goal': 'You want to keep the apartment clean and organized. You prefer quiet study time in the evenings.',
        },
    ),
    prefab_lib.InstanceConfig(
        prefab='basic__Entity',
        role=prefab_lib.Role.ENTITY,
        params={
            'name': 'Jordan',
            'goal': 'You prefer a relaxed environment and like to have friends over. You value flexibility over strict rules.',
        },
    ),
    # ... game master and initializer
]
premise = "Alex and Jordan are roommates who need to discuss house rules and living arrangements."
```

**Scenario 2: Job Interview**
```python
instances = [
    prefab_lib.InstanceConfig(
        prefab='basic__Entity',
        role=prefab_lib.Role.ENTITY,
        params={
            'name': 'Candidate',
            'goal': 'You want to impress the interviewer and get the job. Highlight your relevant experience and enthusiasm.',
        },
    ),
    prefab_lib.InstanceConfig(
        prefab='basic__Entity',
        role=prefab_lib.Role.ENTITY,
        params={
            'name': 'Interviewer',
            'goal': 'You need to assess if the candidate is a good fit. Ask probing questions and evaluate their responses.',
        },
    ),
    # ... game master and initializer
]
premise = "A job interview for a software engineering position at a tech company."
```

**Scenario 3: Family Vacation Planning**
```python
instances = [
    prefab_lib.InstanceConfig(
        prefab='basic__Entity',
        role=prefab_lib.Role.ENTITY,
        params={'name': 'Parent1', 'goal': 'You want a relaxing beach vacation with good food and minimal activities.'},
    ),
    prefab_lib.InstanceConfig(
        prefab='basic__Entity',
        role=prefab_lib.Role.ENTITY,
        params={'name': 'Parent2', 'goal': 'You prefer an active vacation with sightseeing and cultural experiences.'},
    ),
    prefab_lib.InstanceConfig(
        prefab='basic__Entity',
        role=prefab_lib.Role.ENTITY,
        params={'name': 'Teen', 'goal': 'You want somewhere with good nightlife and social activities for young people.'},
    ),
    # ... game master and initializer
]
premise = "A family of three is trying to plan their summer vacation together."
```

### Join the Community

- Check GitHub issues and discussions
- Contribute language model wrappers
- Share your simulations

### Advanced Learning

Once comfortable with basics:
- Create custom components
- Build custom prefabs
- Implement custom engines
- Integrate with external APIs
- Add custom thought chains

---

## Quick Reference

### Minimal Working Example

```python
from concordia.language_model import utils as lm_utils
from concordia.prefabs.simulation import generic as simulation
import concordia.prefabs.entity as entity_prefabs
import concordia.prefabs.game_master as gm_prefabs
from concordia.typing import prefab as prefab_lib
from concordia.utils import helper_functions
import sentence_transformers

# Setup
model = lm_utils.language_model_setup(api_type='openai', model_name='gpt-4', api_key='YOUR_KEY')
embedder = lambda x: sentence_transformers.SentenceTransformer('all-mpnet-base-v2').encode(x)

# Config
prefabs = {**helper_functions.get_package_classes(entity_prefabs),
           **helper_functions.get_package_classes(gm_prefabs)}
instances = [
    prefab_lib.InstanceConfig(prefab='basic__Entity', role=prefab_lib.Role.ENTITY,
                              params={'name': 'Alice', 'goal': 'Your goal here'}),
    prefab_lib.InstanceConfig(prefab='generic__GameMaster', role=prefab_lib.Role.GAME_MASTER,
                              params={'name': 'GM'}),
]
config = prefab_lib.Config(default_premise='Your premise', default_max_steps=10,
                           prefabs=prefabs, instances=instances)

# Run
sim = simulation.Simulation(config=config, model=model, embedder=embedder)
results = sim.play()
```

### Common Patterns

**Two-person negotiation:**
```python
instances = [
    prefab_lib.InstanceConfig(prefab='basic__Entity', role=prefab_lib.Role.ENTITY,
                              params={'name': 'Buyer', 'goal': 'Get the lowest price'}),
    prefab_lib.InstanceConfig(prefab='basic__Entity', role=prefab_lib.Role.ENTITY,
                              params={'name': 'Seller', 'goal': 'Get the highest price'}),
    prefab_lib.InstanceConfig(prefab='generic__GameMaster', role=prefab_lib.Role.GAME_MASTER,
                              params={'name': 'Negotiation GM'}),
]
```

**Group decision:**
```python
# Example: Multiple people choosing a restaurant
instances = [
    prefab_lib.InstanceConfig(
        prefab='basic__Entity',
        role=prefab_lib.Role.ENTITY,
        params={'name': 'Alice', 'goal': 'You prefer Italian food.'},
    ),
    prefab_lib.InstanceConfig(
        prefab='basic__Entity',
        role=prefab_lib.Role.ENTITY,
        params={'name': 'Bob', 'goal': 'You prefer fast food.'},
    ),
    prefab_lib.InstanceConfig(
        prefab='basic__Entity',
        role=prefab_lib.Role.ENTITY,
        params={'name': 'Charlie', 'goal': 'You prefer Asian cuisine.'},
    ),
    prefab_lib.InstanceConfig(
        prefab='generic__GameMaster',
        role=prefab_lib.Role.GAME_MASTER,
        params={
            'name': 'Coordinator',
            'acting_order': 'fixed',  # Round-robin order
        },
    ),
    # ... add initializer ...
]
# Use sequential engine for turn-taking
```

**Market simulation:**
```python
# Example: Simple marketplace
# Note: This requires the marketplace__GameMaster prefab
# Check if it's available in your installation
instances = [
    # Buyers
    prefab_lib.InstanceConfig(
        prefab='basic__Entity',
        role=prefab_lib.Role.ENTITY,
        params={'name': 'Buyer1', 'goal': 'Buy apples at lowest price.'},
    ),
    # Sellers
    prefab_lib.InstanceConfig(
        prefab='basic__Entity',
        role=prefab_lib.Role.ENTITY,
        params={'name': 'Seller1', 'goal': 'Sell apples at highest price.'},
    ),
    # Use marketplace game master (if available)
    prefab_lib.InstanceConfig(
        prefab='marketplace__GameMaster',  # Check if this prefab exists
        role=prefab_lib.Role.GAME_MASTER,
        params={'name': 'Market'},
    ),
]
# Use simultaneous engine for parallel trading
```

---

## Conclusion

Congratulations! You now understand the basics of Concordia. You can:
- ✅ Set up a Concordia environment
- ✅ Create entities and game masters
- ✅ Run simulations
- ✅ Customize behavior
- ✅ Extract and analyze data
- ✅ Save and load checkpoints
- ✅ Troubleshoot common issues

### What You've Learned

1. **Core Concepts**: Entities, Game Masters, Engines, Components, and Prefabs
2. **Setup**: Installing Concordia and configuring language models
3. **Basic Usage**: Creating and running your first simulation
4. **Customization**: Modifying entities, game masters, and engines
5. **Advanced Features**: Custom components, checkpoints, data extraction
6. **Best Practices**: Writing effective goals, managing costs, debugging

### Your Learning Path

**Beginner (You are here!):**
- ✅ Understand core concepts
- ✅ Run basic simulations
- ✅ Customize entities and game masters

**Intermediate (Next steps):**
- Create custom components
- Build specialized game masters
- Integrate with external systems
- Analyze simulation data programmatically

**Advanced:**
- Design custom engines
- Implement complex thought chains
- Build reusable prefab libraries
- Contribute to the framework

### Resources

- **Official Repository**: [GitHub](https://github.com/google-deepmind/concordia)
- **Tech Report**: [arXiv:2312.03664](https://arxiv.org/abs/2312.03664)
- **Examples**: Check the `examples/` directory in the repository
- **Documentation**: Read docstrings in source code for detailed API docs

### Final Tips

1. **Start Small**: Build simple scenarios first, then add complexity
2. **Experiment**: Try different goals, premises, and configurations
3. **Learn from Examples**: Study the provided examples to see patterns
4. **Ask Questions**: Use GitHub issues for help
5. **Share Your Work**: Contribute examples or improvements back to the community

Start with simple scenarios and gradually build more complex simulations. The framework is flexible and powerful - experiment and have fun!

**Happy simulating! 🎮**
