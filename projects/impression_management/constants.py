"""Constants for Impression Management PE Conversation."""

from concordia.components.agent.impression_management_pe import CulturalNorm

# All 17 cultural norms
ALL_CULTURAL_NORMS: list[CulturalNorm] = [
    CulturalNorm(
        "Stated purpose first",
        "Every interaction begins with a shared statement of its goal (e.g., solving a problem, sharing news)."
    ),
    CulturalNorm(
        "Announced topics",
        "Participants clearly outline discussion topics or goals ahead of time and ask before switching subjects."
    ),
    CulturalNorm(
        "Direct, literal language",
        "Plain, literal wording is preferred; transparency outweighs courtesy or euphemism. Sarcasm and other kinds of non-literal knowledge are judged negatively."
    ),
    CulturalNorm(
        "Hidden agendas",
        "Intentions are declared openly; social maneuvering and diplomacy using non-literal or implicit language is considered deceptive and judged negatively."
    ),
    CulturalNorm(
        "Optional small talk",
        "Chit-chat without clear practical purpose (e.g., small talk about weather or personal topics) are generally frowned upon; skipping it is socially acceptable."
    ),
    CulturalNorm(
        "Respect for passions",
        "Lengthy monologues about special interests are generally acceptable and listened to attentively."
    ),
    CulturalNorm(
        "Generous common ground",
        "Speakers assume shared understanding and do not apologize for minor mismatches."
    ),
    CulturalNorm(
        "Low coordination pressure",
        "Momentary overlaps, pauses, or conversational \"misfires\" are shrugged off without embarrassment."
    ),
    CulturalNorm(
        "Slow conversational pacing",
        "Long pauses are normal; no one is pressed for rapid replies, and brief interruptions are tolerated."
    ),
    CulturalNorm(
        "Open clarification",
        "Asking follow-up questions, interrupting, and restating points for accuracy is encouraged, not seen as impolite."
    ),
    CulturalNorm(
        "Eye contact",
        "Looking away or avoiding eye contact is normal; engagement is signaled by words rather than gaze."
    ),
    CulturalNorm(
        "Comfortable silence & parallel play",
        "Quiet co-presence (e.g., reading or scrolling side-by-side) counts as meaningful social time and perceived as comforting and not awkward."
    ),
    CulturalNorm(
        "Negotiated personal space",
        "Physical distance and touch are explicitly discussed; default is preference for greater personal space."
    ),
    CulturalNorm(
        "Integrity over politeness",
        "Even \"white lies\" are discouraged; straightforward feedback is valued and not taken as rudeness. Deception is judged very negatively regardless of intent."
    ),
    CulturalNorm(
        "Minimal figurative speech",
        "Sarcasm, innuendo, and indirect hints are uncommon and usually clarified explicitly."
    ),
    CulturalNorm(
        "Preference of traits in others",
        "Intelligence, authenticity, and focused interests are admired more than overt sociability and extraversion."
    ),
    CulturalNorm(
        "Balanced reciprocity",
        "Each person contributes effort commensurate with capacity; performative enthusiasm is unnecessary."
    ),
    CulturalNorm(
        "Brief by default",
        "Interactions respect \"social battery\" limits; shorter, purpose-driven exchanges are typical and end without offence."
    ),
]

# Trait mode options
TRAIT_MODE_NONE = 'none'
TRAIT_MODE_ACTOR_ONLY = 'actor_only'
TRAIT_MODE_CHOICES = [
    TRAIT_MODE_NONE,
    TRAIT_MODE_ACTOR_ONLY,
]
DEFAULT_TRAIT_MODE = TRAIT_MODE_ACTOR_ONLY

# Default interview role
DEFAULT_INTERVIEW_ROLE = """
    Role: Customer Service Agent

    Responsibilities:
    - Understanding customer issues and needs
    - Resolving problems efficiently and accurately
    - Communicating solutions clearly and empathetically

    The interview evaluates:
    - Problem-solving ability
    - Judgment and decision-making under constraints
    - Clarity and effectiveness of communication
"""

# Default agent names
DEFAULT_ACTOR_NAME = "Riffer"
DEFAULT_AUDIENCE_NAME = "Caden"

# Default parameters
DEFAULT_NUM_PARTICLES = 200
DEFAULT_PROCESS_SIGMA = 0.03
DEFAULT_OBS_SIGMA = 0.08
DEFAULT_RECENT_K = 3
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9
DEFAULT_TURNS = 2
DEFAULT_SEED = 7
