# Impression Management Game Master: Neurotype-Based Norm Adherence Experiment

A Concordia-based experiment simulating structured interviews between agents with different neurotypes operating under shared Riffer communication norms.

## Overview

This experiment studies how agents with different **neurotypes** (Riffer vs. Caden) adhere to **Riffer communication norms** in a simulated job interview. The goal is to measure:

- **Satisfaction**: How satisfied each agent is with the interaction
- **Surprise**: How unexpected each agent finds the other's behavior
- **Competence**: How well the candidate demonstrates job-relevant competence

## Architecture

### Core Components

#### `constants.py`
Defines all world-building constants:
- **Riffer communication norms** (shared environment)
- **Job description** (what both agents know)
- **Role contexts** (candidate and interviewer perspectives)
- **Agent names** and parameter defaults
- **Particle filter parameters** (for potential belief tracking)
- **Behavioral instructions** (per-agent guidance)
- **Memory generation prompts** (for backstory creation)
- **Interview question bank** (for interviewer guidance)

#### `formative_memories_initializer.py`
Generates personalized backstories for agents:
- Uses LLM to create 20-50 memories per agent
- Tailors memories to match neurotype traits
- Candidate memories: communication, social experiences, job-relevant stories
- Interviewer memories: interviewing experience, hiring philosophy, candidate assessment

#### `game_master.py`
Orchestrates the interview and metrics evaluation:
- **Interview flow**: Generates greeting, manages turn-by-turn interaction
- **Agent responses**: Prompts for candidate and interviewer to generate natural speech
- **Metrics evaluation**: Assesses satisfaction, surprise, competence
- **Logging**: Records all interactions with full metrics per turn

#### `main.py`
Experiment entry point:
- Command-line interface for selecting conditions
- Runs single or all experimental conditions
- Generates memories, executes interview, evaluates metrics
- Saves results (logs, transcripts, reports, memories)
- Produces aggregate statistics

## Experimental Design

### Conditions
Four 2Ã—2 combinations:

1. **Riffer candidate Ã— Riffer interviewer**
2. **Caden candidate Ã— Caden interviewer**
3. **Riffer candidate Ã— Caden interviewer**
4. **Caden candidate Ã— Riffer interviewer**

### Interview Flow

```
Turn 0: Greeting
  - Interviewer generates greeting

Turn t (1 to max_turns):
  - Candidate responds to interviewer's message
  - System evaluates candidate's norm adherence, satisfaction, competence
  - Interviewer responds to candidate
  - System evaluates interviewer's norm adherence, satisfaction
  - Both agents' surprise scores recorded
```

### Metrics Per Turn

For each interaction turn, the system logs:

**Satisfaction (0-10)**
- How comfortable does each agent feel?
- Based on clarity, mutual understanding, pace, communication style fit

**Surprise (0-10, 0=expected, 10=very unexpected)**
- How aligned is the other agent with expectations?
- Based on neurotype match and norm adherence

**Competence (Candidate only, 0-10)**
- Does the candidate demonstrate job-relevant skills?
- Based on problem-solving, communication clarity, professionalism

## Usage

### Basic Run (Default: Questionnaire for Riffer_x_Caden)

Builds the candidate and interviewer, initializes formative memories, and then runs both Convergent Validity 1 and Convergent Validity 2 questionnaires for the `Riffer_x_Caden` condition.

```bash
python main.py \
  --model gemini-flash-latest \
  --save_dir ./results
```

### Interview Run

Runs the interview loop instead of the questionnaire.

```bash
python main.py \
  --model gemini-flash-latest \
  --save_dir ./results \
  --interview \
  --turns 10
```

### Run Single Condition

```bash
python main.py \
  --model gemini-flash-latest \
  --condition Riffer_x_Caden \
  --save_dir ./results \
  --turns 5
```

### Command-Line Arguments

- `GEMINI_API_KEY` or `GOOGLE_API_KEY`: Environment variable for Gemini API access
- `--save_dir`: Directory for saving results (default: timestamped folder under `./temp/`)
- `--turns`: Number of interview turns (default: 6)
- `--condition`: Choose condition or "all" (default: all)
- `--model`: Gemini model name (default: `gemini-flash-latest`)

## Output Structure

```
results/
â”œâ”€â”€ aggregate_results.json          # Summary across all conditions
â”œâ”€â”€ Riffer_x_Riffer/
â”‚   â”œâ”€â”€ interaction_log.json        # Detailed metrics per turn
â”‚   â”œâ”€â”€ transcript.txt              # Full conversation transcript
â”‚   â”œâ”€â”€ report.txt                  # Summary statistics and insights
â”‚   â””â”€â”€ memories.txt                # Generated agent memories
â”œâ”€â”€ Caden_x_Caden/
â”‚   â””â”€â”€ [same structure]
â”œâ”€â”€ Riffer_x_Caden/
â”‚   â””â”€â”€ [same structure]
â””â”€â”€ Caden_x_Riffer/
    â””â”€â”€ [same structure]
```

### Files

- **interaction_log.json**: Machine-readable metrics (turn, satisfaction, surprise, competence)
- **transcript.txt**: Full conversation for manual review
- **report.txt**: Human-readable summary with averages and insights
- **memories.txt**: Agent backstories used in the conversation
- **aggregate_results.json**: High-level statistics across conditions

## Key Design Decisions

### 1. Norm Clarity
Both agents have **explicit knowledge** of the Riffer communication norms via:
- Shared `RIFFER_COMMUNICATION_NORMS` constant
- Role context mentioning norm awareness
- Behavioral instructions referencing norms

This allows studying adherence despite knowledge, not adherence through ignorance.

### 2. Imperfect Adherence
Behavioral instructions build in tension:
- Agents know the norms
- Agents want to succeed (candidate) or evaluate fairly (interviewer)
- But their natural neurotype may conflict with norms
- Real-time decision pressure

This models the psychological reality that knowing â‰  doing.

### 3. Conversational Naturalness
- No rigid scripts; agents generate natural speech
- Interviewer draws from question bank but adapts
- Candidate references memories and responds organically
- Conversation flows naturally while being measurable

### 4. Role Separation
- **Candidate**: Minority role, trying to adapt to norms, wants to be hired
- **Interviewer**: Majority role, enforces norms, evaluates competence
- Game Master: Neutral orchestrator, evaluates metrics, doesn't participate

## Expected Findings

### Hypotheses

1. **Norm adherence varies by neurotype mismatch**
   - Riffer agents may struggle under Caden norms
   - Caden agents may struggle under Riffer norms

2. **Satisfaction correlates with neurotype match**
   - Matching neurotypes â†’ higher mutual satisfaction
   - Mismatched neurotypes â†’ potential friction

3. **Surprise inversely correlates with neurotype match**
   - Matching neurotypes â†’ behavior is expected
   - Mismatched neurotypes â†’ more surprising behavior

4. **Competence is separable from adherence**
   - An agent can be highly competent but poorly adherent
   - Or adherent but struggling with tasks

## Customization

### Modify Norms
Edit `RIFFER_COMMUNICATION_NORMS` in `constants.py` to change the experimental norms.

### Add More Neurotypes
Extend `NEUROTYPE_CHOICES` and update `_get_neurotype_description()` in `formative_memories_initializer.py`.

### Change Job Role
Replace `JOB_DESCRIPTION` and interview context in `constants.py`.

### Adjust Particle Filter Parameters
Modify `DEFAULT_*` constants for belief tracking (if extending with belief modeling).

## Technical Notes

### Dependencies
- Concordia framework
- Gemini API (Google AI Studio)
- Python 3.9+

### LLM Calls
- **Memory generation**: ~40 calls (20 candidate + 20 interviewer) per condition
- **Interview per turn**: ~6 calls (1 candidate, 1 interviewer, 4 evaluations)
- **Total per condition**: ~100 LLM calls
- **All conditions**: ~400 calls

### Token Usage
Approximate for 4 conditions Ã— 10 turns:
- Input tokens: ~50,000
- Output tokens: ~30,000

## References

The experiment is based on the specification in `im_gm.md`, which outlines:
- Neurotype-based norm adherence research
- Impression management dynamics
- Behavioral instruction theory
- Metrics for measuring adherence, satisfaction, and surprise

## Future Extensions

1. **Belief Tracking**: Integrate particle filters to model each agent's beliefs about the other
2. **Multi-round**: Multiple interview sessions to study learning
3. **Vignette Injection**: Add specific scenarios or dilemmas
4. **Behavioral Variants**: Test different levels of norm knowledge or motivation
5. **Real Agent Integration**: Connect to existing Concordia agents with richer behavior models
