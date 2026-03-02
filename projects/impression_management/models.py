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
    actor_options: list[dict[str, str]]
    actor_chosen_index: int | None
    actor_chosen: str
    audience_options: list[dict[str, str]]
    audience_chosen_index: int | None
    audience_chosen: str
    actor_interaction_summary: str
    audience_interaction_summary: str


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
    trait_mode: str
    no_context: bool
    seed: int
    save_dir: str
    actor_name: str
    audience_name: str
    audience_traits_spreadsheet: str | None
    llm_type: str
    local_model: str
