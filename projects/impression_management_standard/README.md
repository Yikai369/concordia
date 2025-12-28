# Impression Management PE - Standard Simulation Loop

A two-agent conversation system with particle filter belief tracking, using the **standard Concordia simulation loop** (`sim.play()`). This project models how an actor (interviewee) tracks and adapts to an audience's (interviewer's) hidden evaluation using a particle filter and prediction error (PE) calculations.

## Overview

This system simulates a conversation between:
- **Actor (John)**: An interviewee trying to make a good impression, using a particle filter to track their belief about the audience's evaluation (`I_hat`)
- **Audience (Jane)**: An interviewer who evaluates the actor and responds based on their true internal evaluation (`I_t`)

The actor uses:
- **Particle Filter (PF)**: Bayesian tracking of the audience's hidden evaluation state
- **Prediction Error (PE)**: Difference between predicted and observed evaluation
- **Reflection**: Adaptive strategy based on current belief
- **Cultural Norms & Personality Traits**: Influence behavior and evaluation

## Installation

### Prerequisites

- Python 3.8+
- Conda environment (recommended)

### Setup

1. **Activate your conda environment**:
   ```bash
   conda activate concordia
   ```

2. **Install dependencies** (if not already installed):
   ```bash
   pip install sentence-transformers python-dotenv
   ```

3. **Set up API key** (for OpenAI models):

   **Option A: Environment variable** (PowerShell):
   ```powershell
   $Env:OPENAI_API_KEY = "sk-your-api-key-here"
   ```

   **Option B: .env file** (recommended):
   Create `projects/impression_management_standard/.env`:
   ```
   OPENAI_API_KEY=sk-your-api-key-here
   ```

## Usage

### Basic Command

From the project root directory:

```bash
python projects/impression_management_standard/main.py --turns 2
```

### Common Options

```bash
# Use local Ollama model instead of OpenAI
python projects/impression_management_standard/main.py --turns 2 --llm_type local

# Specify custom output directory
python projects/impression_management_standard/main.py --turns 4 --save_dir ./my_results

# Use different OpenAI model
python projects/impression_management_standard/main.py --turns 3 --model gpt-4o-mini

# Disable cultural norms or personality traits
python projects/impression_management_standard/main.py --turns 2 --no_audience_norms --no_traits

# Set random seed for reproducibility
python projects/impression_management_standard/main.py --turns 2 --seed 42
```

### All Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--turns` | int | 2 | Total number of conversation turns |
| `--model` | str | `gpt-4o` | OpenAI model name |
| `--temperature` | float | 0.2 | Sampling temperature |
| `--top_p` | float | 0.9 | Top-p nucleus sampling |
| `--window` | int | 3 | Recent K turns to condition on |
| `--outfile` | str | `pe_conversation_log.json` | Output JSON filename |
| `--save_dir` | str | `./temp/<timestamp>` | Output directory (auto-created if None) |
| `--seed` | int | 7 | Random seed for reproducibility |
| `--actor_name` | str | `John` | Actor/interviewee name |
| `--audience_name` | str | `Jane` | Audience/interviewer name |
| `--llm_type` | str | `openai` | LLM type: `openai` or `local` |
| `--local_model` | str | `llama3.1:8b` | Local model name (for Ollama) |
| `--no_audience_norms` | flag | False | Disable cultural norms for audience |
| `--no_traits` | flag | False | Disable personality traits |
| `--no_context` | flag | False | Disable interview context |

## Output Location

### Default Location

By default, conversation data is saved to:
```
./temp/<YYYY-MM-DD_HH-MM-SS>/pe_conversation_log.json
```

For example:
```
./temp/2025-12-17_21-34-41/pe_conversation_log.json
```

The directory is automatically created with a timestamp if `--save_dir` is not specified.

### Custom Location

To specify a custom output directory:
```bash
python projects/impression_management_standard/main.py --turns 2 --save_dir ./my_results
```

The output file will be:
```
./my_results/pe_conversation_log.json
```

### Output Format

The JSON file contains a list of turn logs, each with:
- `time`: ISO timestamp
- `turn`: Turn number
- `speaker`: Actor name
- `listener`: Audience name
- `speaker_text`: Actor's utterance
- `speaker_body`: Actor's body language description
- `audience_I`: Audience's true internal evaluation (0-1)
- `audience_text`: Audience's response text
- `audience_body`: Audience's body language description
- `actor_I_hat`: Actor's belief about evaluation (0-1)
- `actor_pe`: Prediction error (signed: previous I_hat - current I_hat)
- `reflection_text`: Actor's reflection on how to improve
- `ess`: Effective sample size (particle filter quality metric)

Example:
```json
[
  {
    "time": "2025-12-17T21:35:57Z",
    "turn": 2,
    "speaker": "John",
    "listener": "Jane",
    "speaker_text": "I effectively bridge communication...",
    "speaker_body": "Maintain steady eye contact...",
    "audience_I": 0.7,
    "audience_text": "The candidate shows potential...",
    "audience_body": "Fidgeted with hands frequently.",
    "actor_I_hat": 0.86,
    "actor_pe": 0.06,
    "reflection_text": "To improve my goal achievement...",
    "ess": 28.29
  }
]
```

## Project Structure

```
projects/impression_management_standard/
├── main.py                    # Entry point using sim.play()
├── simulation_config.py       # Creates Config object with prefabs/instances
├── simple_audience_prefab.py  # Simple audience prefab for standard loop
├── audience_act_component.py  # Simple act component that returns stored response
├── data_extraction.py         # Extracts data from simulation entities
├── config.py                  # Argument parsing and API key validation
├── setup.py                   # Language model and embedder setup
├── results.py                 # Results saving and display
├── constants.py               # Constants, cultural norms, personality traits
├── utils.py                   # Utility functions
├── models.py                  # Data classes (TurnLog, ConversationConfig)
├── README.md                  # This file
└── docs/
    └── comparison_with_examples.md  # Comparison with manual version
```

## How It Works

### Turn Sequence

**Turn 1:**
1. Actor acts (no PF update, no reflection - PF history is empty)
2. Audience observes actor's utterance → automatically triggers evaluation
3. Audience acts (returns stored evaluation response)

**Turn 2+:**
1. Actor observes audience's previous response → automatically triggers:
   - **PF update** in `post_observe()` (extracts measurement from audience response, computes I_hat and PE)
   - **Reflection** in `pre_act()` (reads updated I_hat from memory, generates reflection)
2. Actor acts (using updated belief and reflection)
3. Audience observes actor's utterance → automatically triggers evaluation
4. Audience acts (returns stored evaluation response)

### Component Lifecycle

The automatic component lifecycle ensures correct ordering:

1. **`pre_observe()`**: All components extract data from observation (runs in parallel)
2. **`post_observe()`**: PF component updates I_hat and PE (runs in parallel with other components)
3. **`pre_act()`**: Reflection component reads I_hat and generates reflection (runs **after** `post_observe()` completes)
4. **`act()`**: Act component uses I_hat and reflection to generate utterance

**Key Point**: The sequential phase ordering (`post_observe()` → `pre_act()`) ensures PF update completes before reflection reads I_hat, avoiding race conditions.

### Components

**Actor Components:**
- `IMPEActComponent`: Generates utterances based on current belief and reflection
- `IMPEActorParticleFilterComponent`: Updates particle filter in `post_observe()` if audience response available
- `IMPEReflectionComponent`: Generates reflection in `pre_act()` if PF history exists
- `IMPEMemoryComponent`: Stores conversation history, PF state, and reflections
- `CulturalNormsComponent`: Applies cultural norms to actor behavior
- `PersonalityTraitsComponent`: Applies personality traits to actor behavior

**Audience Components:**
- `IMPEAudienceEvaluationComponent`: Triggers automatically on `observe()` → generates I_t and response
- `SimpleAudienceActComponent`: Returns stored evaluation response

## Differences from Manual Version

### Original (`impression_management/`)
- Manual conversation loop in `conversation.py`
- Direct component method calls
- Full control over turn sequence
- Game master created but not used

### Standard Version (`impression_management_standard/`)
- Uses `sim.play()` with automatic loop
- Config-based setup with prefabs/instances
- Automatic component lifecycle
- Game master orchestrates turns
- Automatic logging

## Troubleshooting

### API Key Issues

If you see:
```
ERROR: OPENAI_API_KEY environment variable required for OpenAI.
```

**Solutions:**
1. Set environment variable: `$Env:OPENAI_API_KEY = "sk-..."`
2. Create `.env` file: `OPENAI_API_KEY=sk-...`
3. Use local model: `--llm_type local`

### Import Errors

If you see `ModuleNotFoundError`:
- Make sure you're running from the project root directory
- Ensure the conda environment is activated
- Check that all dependencies are installed

### No Output File

If no output file is created:
- Check that the script completed successfully
- Verify write permissions in the output directory
- Check console output for error messages

## Examples

### Example 1: Basic Run
```bash
python projects/impression_management_standard/main.py --turns 3
```

### Example 2: Local Model
```bash
python projects/impression_management_standard/main.py --turns 2 --llm_type local --local_model llama3.1:8b
```

### Example 3: Custom Output
```bash
python projects/impression_management_standard/main.py --turns 4 --save_dir ./results/run1 --seed 42
```

### Example 4: Minimal Features
```bash
python projects/impression_management_standard/main.py --turns 2 --no_audience_norms --no_traits --no_context
```

## Further Reading

- **Comparison with Examples**: See `docs/comparison_with_examples.md` for detailed comparison between manual and standard approaches
- **Particle Filter Algorithm**: See `projects/impression_management/docs/particle_filter.md` for technical details
- **Component Architecture**: See `concordia/components/agent/impression_management_pe.py` for component implementations

## License

Part of the Concordia framework project.
