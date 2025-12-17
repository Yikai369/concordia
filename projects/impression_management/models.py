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
    actor_pe: float  # Absolute prediction error
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
