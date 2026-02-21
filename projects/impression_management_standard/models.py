"""Data models for Impression Management PE Conversation."""

from dataclasses import dataclass


@dataclass
class TurnLog:
    """Log entry for a single turn."""
    time: str
    turn: int
    speaker: str
    listener: str
    speaker_text: str
    speaker_body: str
    audience_I: float  # I_t (true hidden state)
    audience_text: str
    audience_body: str
    actor_I_hat: float  # Actor's belief
    actor_pe: float  # Signed prediction error (positive = underestimating, negative = overestimating)
    reflection_text: str
    ess: float  # Effective sample size


@dataclass
class ConversationConfig:
    """Configuration for the conversation."""
    turns: int
    model: str
    temperature: float
    top_p: float
    window: int
    outfile: str
    no_audience_norms: bool
    no_traits: bool
    no_context: bool
    seed: int
    save_dir: str
    actor_name: str
    audience_name: str
    llm_type: str
    local_model: str
    print_trace: bool = False  # Whether to print pretty trace
    no_plots: bool = False  # Whether to disable plotting
    enable_info_flow_logging: bool = False  # Whether to enable information flow history logging
    enable_simplified_log: bool = False  # Whether to generate simplified log (requires enable_info_flow_logging)
    simplified_log_format: str = 'compact'  # Format: 'compact', 'markdown', or 'text'
    save_component_logs: bool = False  # Whether to save Concordia component-level logs
    enable_self_assessment: bool = False  # Whether to enable self-assessment component
    consistency_threshold: float = 0.7  # Minimum consistency score (0-1) to accept response without revision
    disable_revision: bool = False  # Whether to disable revision of inconsistent responses (only log assessments)
    no_instructions: bool = False  # Whether to disable Instructions component
    no_self_perception: bool = False  # Whether to disable SelfPerception component
    enable_situation_perception: bool = False  # Whether to enable SituationPerception
    enable_person_by_situation: bool = False  # Whether to enable PersonBySituation
    no_world_building: bool = False  # Whether to disable world-building context
    no_interview_context: bool = False  # Whether to disable interview context
