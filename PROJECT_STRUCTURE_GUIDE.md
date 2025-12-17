# Project Structure Guide for Concordia Framework

This guide suggests a recommended way to organize your scripts and code when building projects based on the Concordia framework.

## Where to Place Your Project

**Important**: You're currently in the Concordia framework repository directory. Here are your options:

### Option 1: Sibling Directory (Recommended)
Create your project as a **sibling directory** to the Concordia framework:

```
OneDrive/Documents/
├── concordia/              # Framework (current location)
│   ├── concordia/
│   ├── examples/
│   └── ...
└── my_concordia_project/   # Your project (create here)
    ├── config/
    ├── scripts/
    └── ...
```

**Advantages:**
- Keeps your project code separate from framework code
- Easier to manage dependencies
- Cleaner separation of concerns
- Won't interfere with framework updates

**To use Concordia from a sibling directory:**
- Install Concordia: `pip install -e ../concordia` (if installed in development mode)
- Or install from PyPI: `pip install gdm-concordia`
- Or add to PYTHONPATH: `export PYTHONPATH=../concordia:$PYTHONPATH`

### Option 2: Projects Folder (Alternative)
If you want to keep everything in one place, create a `projects/` folder:

```
concordia/                  # Framework (current location)
├── concordia/
├── examples/
├── projects/               # Your projects folder
│   └── my_project/         # Your project
│       ├── config/
│       ├── scripts/
│       └── ...
└── ...
```

**Advantages:**
- Everything in one location
- Easy to reference framework code
- Good for experimentation

**Disadvantages:**
- Mixes your code with framework code
- Can be confusing when updating the framework

### Option 3: Completely Separate Location
Create your project anywhere on your system and install Concordia as a package:

```bash
pip install gdm-concordia  # or pip install -e /path/to/concordia
```

## Overview

Concordia is a library for building generative social simulations where:
- **Entities** are AI agents/characters
- **Game Masters** narrate and resolve actions
- **Components** provide behaviors (memory, observation, action, etc.)
- **Prefabs** are pre-built templates combining components
- **Simulations** orchestrate the entire game loop

## Recommended Project Structure

```
your_project/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── config/                      # Configuration files
│   ├── __init__.py
│   ├── model_configs.py        # Language model configurations
│   ├── simulation_configs.py   # Simulation parameters
│   └── entity_configs.py       # Entity-specific settings
│
├── components/                  # Custom components (if needed)
│   ├── __init__.py
│   ├── agent/                  # Custom agent components
│   │   ├── __init__.py
│   │   └── custom_component.py
│   └── game_master/            # Custom game master components
│       ├── __init__.py
│       └── custom_gm_component.py
│
├── prefabs/                     # Custom prefabs (if needed)
│   ├── __init__.py
│   ├── entity/                 # Custom entity prefabs
│   │   ├── __init__.py
│   │   └── custom_entity.py
│   └── game_master/            # Custom game master prefabs
│       ├── __init__.py
│       └── custom_gm.py
│
├── scripts/                     # Main execution scripts
│   ├── __init__.py
│   ├── run_simulation.py      # Main simulation runner
│   ├── setup_models.py         # Model initialization utilities
│   └── extract_results.py      # Data extraction utilities
│
├── utils/                       # Project-specific utilities
│   ├── __init__.py
│   ├── logging.py              # Custom logging functions
│   ├── data_processing.py      # Data extraction/processing
│   └── visualization.py        # Plotting/visualization helpers
│
├── data/                        # Data directory
│   ├── inputs/                 # Input data (if any)
│   ├── outputs/                # Simulation outputs
│   │   ├── logs/              # Raw simulation logs
│   │   ├── results/           # Processed results
│   │   └── visualizations/    # Generated plots/figures
│   └── checkpoints/            # Simulation checkpoints (if using)
│
├── notebooks/                   # Jupyter notebooks (for exploration)
│   ├── exploration.ipynb      # Experimentation notebook
│   └── analysis.ipynb         # Results analysis
│
├── tests/                       # Unit tests
│   ├── __init__.py
│   ├── test_components.py
│   └── test_simulation.py
│
└── examples/                    # Example scripts (optional)
    ├── simple_example.py
    └── advanced_example.py
```

## Detailed Structure Explanation

### 1. Configuration (`config/`)

Separate configuration from code for easier experimentation:

```python
# config/model_configs.py
"""Language model configurations."""

MODEL_CONFIGS = {
    'openai_gpt4': {
        'api_type': 'openai',
        'model_name': 'gpt-4',
        'temperature': 0.7,
        'top_p': 0.9,
    },
    'openai_gpt35': {
        'api_type': 'openai',
        'model_name': 'gpt-3.5-turbo',
        'temperature': 0.7,
        'top_p': 0.9,
    },
    # Add more configurations...
}

EMBEDDER_CONFIG = {
    'model_name': 'sentence-transformers/all-mpnet-base-v2',
}
```

```python
# config/simulation_configs.py
"""Simulation parameters."""

from dataclasses import dataclass

@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""
    max_steps: int = 20
    premise: str = "Default simulation premise"
    agent_names: list[str] = None
    goal_name: str = "likability"
    goal_description: str = "Be perceived as likable"
    goal_ideal: float = 1.0
    recent_k: int = 3  # Window size for recent history

    def __post_init__(self):
        if self.agent_names is None:
            self.agent_names = ['Agent A', 'Agent B']
```

### 2. Custom Components (`components/`)

If you need custom behaviors beyond Concordia's built-in components:

```python
# components/agent/custom_memory.py
"""Custom memory component example."""

from concordia.components.agent import memory
from concordia.typing import entity_component

class CustomMemoryComponent(memory.MemoryComponent):
    """Extended memory with custom functionality."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add custom initialization

    # Override or extend methods as needed
```

### 3. Custom Prefabs (`prefabs/`)

Create reusable entity/game master templates:

```python
# prefabs/entity/custom_entity.py
"""Custom entity prefab."""

from concordia.components.agent import memory, observation, plan
from concordia.prefabs.entity import basic
from concordia.typing import entity_component, prefab

class CustomEntity__Entity(basic.BasicEntity__Entity):
    """Custom entity with specific component configuration."""

    def __init__(
        self,
        name: str,
        model,
        embedder,
        goal_name: str = "default_goal",
        **kwargs
    ):
        # Custom component setup
        components = [
            # Add your custom components here
        ]

        super().__init__(
            name=name,
            model=model,
            embedder=embedder,
            components=components,
            **kwargs
        )
```

### 4. Main Scripts (`scripts/`)

Keep main execution scripts clean and focused:

```python
# scripts/run_simulation.py
"""Main simulation runner."""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.model_configs import MODEL_CONFIGS, EMBEDDER_CONFIG
from config.simulation_configs import SimulationConfig
from scripts.setup_models import setup_language_model, setup_embedder
from utils.logging import setup_logging
from utils.data_processing import save_results

def main():
    parser = argparse.ArgumentParser(description='Run Concordia simulation')
    parser.add_argument('--config', type=str, default='default',
                       help='Configuration name')
    parser.add_argument('--output_dir', type=str, default='data/outputs',
                       help='Output directory')
    parser.add_argument('--api_key', type=str, default=None,
                       help='API key (or set OPENAI_API_KEY env var)')
    args = parser.parse_args()

    # Setup
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("API key required")

    model = setup_language_model(MODEL_CONFIGS[args.config], api_key)
    embedder = setup_embedder(EMBEDDER_CONFIG)

    # Load simulation config
    sim_config = SimulationConfig()

    # Create simulation (your implementation here)
    # ... simulation setup code ...

    # Run simulation
    # ... run code ...

    # Save results
    save_results(results, args.output_dir)

    print(f"Simulation complete. Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
```

```python
# scripts/setup_models.py
"""Model initialization utilities."""

import sentence_transformers
from concordia.language_model import utils as language_model_utils

def setup_language_model(config: dict, api_key: str):
    """Initialize language model from configuration."""
    return language_model_utils.language_model_setup(
        api_type=config['api_type'],
        model_name=config['model_name'],
        api_key=api_key,
        disable_language_model=False,
    )

def setup_embedder(config: dict):
    """Initialize embedder from configuration."""
    st_model = sentence_transformers.SentenceTransformer(
        config['model_name']
    )
    return lambda x: st_model.encode(x, show_progress_bar=False)
```

### 5. Utilities (`utils/`)

Project-specific helper functions:

```python
# utils/data_processing.py
"""Data extraction and processing utilities."""

import json
from pathlib import Path
from typing import Any

def extract_turn_data(raw_log: list[dict[str, Any]],
                      agent_names: list[str]) -> list[dict]:
    """Extract structured data from simulation log."""
    # Your extraction logic here
    pass

def save_results(data: Any, output_dir: str, filename: str = 'results.json'):
    """Save results to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Results saved to {filepath}")
```

### 6. Data Organization (`data/`)

Keep data separate from code:

- `data/inputs/`: Any input data files
- `data/outputs/logs/`: Raw simulation logs
- `data/outputs/results/`: Processed/structured results
- `data/outputs/visualizations/`: Generated plots
- `data/checkpoints/`: Save states for long-running simulations

### 7. Notebooks (`notebooks/`)

Use notebooks for exploration and analysis:

- `exploration.ipynb`: Experiment with different configurations
- `analysis.ipynb`: Analyze results, create visualizations

## Best Practices

### 1. **Separation of Concerns**
   - Keep configuration separate from code
   - Separate data processing from simulation logic
   - Use utilities for reusable functions

### 2. **Modularity**
   - Create custom components for reusable behaviors
   - Use prefabs for common entity/game master patterns
   - Break complex simulations into smaller modules

### 3. **Configuration Management**
   - Use config files or dataclasses for parameters
   - Support command-line arguments for flexibility
   - Use environment variables for sensitive data (API keys)

### 4. **Logging and Results**
   - Save raw logs for debugging
   - Extract structured data for analysis
   - Use consistent naming conventions for output files

### 5. **Code Organization**
   - Follow Concordia's patterns (components, prefabs, simulations)
   - Keep main scripts simple and focused
   - Use type hints for better code clarity

### 6. **Testing**
   - Test custom components in isolation
   - Test simulation setup separately from execution
   - Use mock models for faster testing

## Example: Simple Project Structure

For a simpler project, you can use a minimal structure:

```
simple_project/
├── config.py              # All configurations in one file
├── components.py          # Custom components (if any)
├── prefabs.py             # Custom prefabs (if any)
├── run.py                 # Main script
├── utils.py               # Helper functions
├── data/
│   └── outputs/
└── README.md
```

## Example: Complex Project Structure

For larger projects with multiple simulations:

```
complex_project/
├── simulations/           # Different simulation types
│   ├── conversation/
│   ├── negotiation/
│   └── social_dynamics/
├── shared/                # Shared code across simulations
│   ├── components/
│   └── prefabs/
└── ... (rest of structure)
```

## Integration with Concordia Framework

When creating custom components/prefabs, follow Concordia's patterns:

1. **Components** should inherit from appropriate base classes:
   - `ContextComponent` for context processing
   - `ActionSpecIgnored` for components that don't use action specs
   - See `concordia/components/agent/` for examples

2. **Prefabs** should follow naming convention: `name__Type`
   - Example: `custom_entity__Entity`, `my_gm__GameMaster`

3. **Use Concordia's utilities**:
   - `helper_functions.get_package_classes()` to load prefabs
   - `prefab_lib.InstanceConfig` for entity/game master instances
   - `prefab_lib.Config` for simulation configuration

## Next Steps

1. Start with a simple structure and expand as needed
2. Look at `examples/pe_conversation_concordia.py` for a complete example
3. Review `TUTORIAL.md` for framework concepts
4. Check `concordia/prefabs/` for existing prefabs you can use/extend

## Additional Resources

- Framework Tutorial: `TUTORIAL.md`
- Framework README: `README.md`
- Example implementations: `examples/`
- Component documentation: `concordia/components/`
- Prefab examples: `concordia/prefabs/`
