"""Constants for the IM game-master experiment.

This module preserves world building constants, communication norms,
interview context, names, and particle filter defaults used by IMPE components.
"""

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

CADEN_COMMUNICATION_NORMS = """
Caden social norms:

- Direct and explicit communication is preferred
- Minimal reliance on non-verbal cues
- Honesty is valued over politeness
- Pauses in conversation are acceptable
- Open clarification is encouraged
"""

# =============================================================================
# Customer Service Job Description (Shared Knowledge)
# =============================================================================

JOB_DESCRIPTION = """
Customer Service Representative Role

Responsibilities:
- Understanding customer issues and needs
- Resolving problems efficiently and accurately
- Communicating solutions clearly and empathetically

Evaluation Criteria:
- Clarity of communication
- Problem-solving ability
- Responsiveness and adaptability
- Ability to follow communication norms
"""

# =============================================================================
# Interview Context for Candidate and Interviewer
# =============================================================================

CANDIDATE_ROLE_CONTEXT = """
You are applying for a customer service role.

You understand the job requirements and will be evaluated on:
- communication clarity
- problem-solving ability
- professionalism

The interviewer operates under Caden communication norms:
- direct
- explicit
- minimal reliance on non-verbal cues

You are aware that:
- you will be judged based on these norms
- deviations may negatively affect evaluation

Your goal is to perform well in the interview while managing your natural tendencies.
"""

INTERVIEWER_ROLE_CONTEXT = """
You are conducting an interview for a customer service role.

You evaluate candidates based on:
- clarity
- directness
- problem-solving ability
- adherence to communication norms

You follow Caden communication norms and expect candidates to do the same.

You will ask questions to assess competence while maintaining a natural conversational flow.
"""

# =============================================================================
# Agent Names
# =============================================================================

DEFAULT_CANDIDATE_NAME = "Candidate"
DEFAULT_INTERVIEWER_NAME = "Interviewer"
AGENT_NAME_POOL = [
    "Alex",
    "Jordan",
    "Taylor",
    "Casey",
    "Morgan",
]

# =============================================================================
# Particle Filter Parameters
# =============================================================================

DEFAULT_NUM_PARTICLES = 200
DEFAULT_PROCESS_SIGMA = 0.03
DEFAULT_OBS_SIGMA = 0.08
DEFAULT_RECENT_K = 3

# =============================================================================
# Interaction Parameters
# =============================================================================

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TURNS = 10
DEFAULT_SEED = 42

# =============================================================================
# Neurotype Definitions
# =============================================================================

NEUROTYPE_RIFFER = "Riffer"
NEUROTYPE_CADEN = "Caden"

NEUROTYPE_CHOICES = [
    NEUROTYPE_RIFFER,
    NEUROTYPE_CADEN,
]

GROUP_BEHAVIOR_CONTEXT = {
    NEUROTYPE_RIFFER: (
        "GROUP IDENTITY: You belong to the Riffer group.\n"
        "GROUP-LEVEL COMMUNICATION TENDENCIES FOR ACTING: "
        "favor direct literal wording, explicit assumptions, predictable structure, "
        "and clear clarification requests; under pressure, keep message content concrete "
        "and body language controlled rather than performative."
    ),
    NEUROTYPE_CADEN: (
        "GROUP IDENTITY: You belong to the Caden group.\n"
        "GROUP-LEVEL COMMUNICATION TENDENCIES FOR ACTING: "
        "favor adaptive social pacing, context-sensitive phrasing, smooth turn-taking, "
        "and rapport-preserving responses; under pressure, keep messages cooperative "
        "and body language socially reassuring."
    ),
}

# =============================================================================
# Experiment Conditions
# =============================================================================

EXPERIMENT_CONDITIONS = [
    ("Riffer", "Riffer"),  # (candidate, interviewer)
    ("Caden", "Caden"),
    ("Riffer", "Caden"),
    ("Caden", "Riffer"),
]

# =============================================================================
# Candidate Behavioral Instructions
# =============================================================================

CANDIDATE_BEHAVIORAL_INSTRUCTIONS = """
When responding:
- You are being evaluated for a job
- You are trying to follow Riffer communication norms:
  - direct
  - explicit
  - clear
- You want to appear competent and aligned with expectations
- You actively monitor:
  - whether your response is clear
  - whether it follows norms

However:
- your natural tendencies may still influence your response
- under pressure, you may revert to default communication patterns

Balance:
- performing well in the interview
- adhering to norms
- your natural tendencies
"""

# =============================================================================
# Interviewer Behavioral Instructions
# =============================================================================

INTERVIEWER_BEHAVIORAL_INSTRUCTIONS = """
When responding:
- You are assessing the candidate for a customer service role
- You ask questions to evaluate competence

You should:
- draw from relevant interview questions
- adapt based on previous responses
- maintain a natural conversational flow

You evaluate based on:
- clarity
- directness
- problem-solving ability
- adherence to norms

If the candidate deviates:
- you may notice
- it may influence your evaluation
"""

# =============================================================================
# Formative Memory Prompt Templates
# =============================================================================

MEMORY_PROMPTS = [
    "Describe a time you helped someone who was confused or frustrated.",
    "Tell a story about resolving a misunderstanding.",
    "Describe a situation where you had to explain something clearly.",
    "Recall a time you didn’t know the answer but had to respond anyway.",
    "Describe how you handled a difficult or demanding person.",
    "Describe a time you had to solve a problem without clear instructions.",
    "Tell a story where you had to think quickly to handle a situation.",
    "Describe a time you made a mistake and how you handled it.",
    "Describe a situation where communication broke down.",
    "Recall a time you had to stay calm while someone else was upset.",
    "Describe a time you worked with someone very different from you.",
    "Describe a situation where you had to be especially clear to avoid confusion.",

    "Describe a time you misunderstood what someone expected from you.",
    "Recall a situation where your response surprised someone.",
    "Describe a time you weren’t sure how to respond in a conversation.",
    "Describe a time someone reacted differently than you expected.",
    "Recall a moment when being honest caused tension.",

    "Describe a time you felt out of place in a group.",
    "Recall a situation where you had to adjust to fit in.",
    "Describe how you behave when entering a new group of people.",
    "Describe a group experience that went well or poorly.",

    "Describe a time you felt overwhelmed in a situation involving others.",
    "Recall a situation where you didn’t know what was expected of you.",
    "Describe a time you had to make a decision quickly.",

    "Describe your first experience with responsibility.",
    "Recall a time you worked with others toward a goal.",
    "Describe how you handled unclear instructions in a task.",

    "Describe a time you had a disagreement with someone you knew well.",

    "Describe a memorable experience from school involving other people.",

    "Describe a time you had to choose between being honest and being polite."
]

# =============================================================================
# Diversity of Interview Questions (for interviewer inspiration)
# =============================================================================

INTERVIEW_QUESTION_BANK = [
    "Tell me about yourself and your experience in customer service.",
    "Describe a challenging customer interaction you've had.",
    "How do you handle a customer who is frustrated or upset?",
    "What does good communication mean to you?",
    "Give an example of how you've resolved a difficult problem.",
    "How do you prioritize tasks when you have multiple issues to handle?",
    "Describe your approach to learning new systems or procedures.",
    "Tell me about a time you had to clarify something you didn't understand.",
    "How do you stay calm under pressure?",
    "What are your strengths in this type of work?",
    "How do you approach explaining complex information to others?",
    "Describe a time you had to make a decision without clear guidance.",
    "How do you ensure accuracy in your work?",
    "Tell me about a time you received critical feedback.",
    "What would you do if you gave a customer incorrect information?",
]


NEUROTYPE_TRAIT_PARAGRAPHS = {
    NEUROTYPE_CADEN: """
      This individual exhibits a preference for routines and structured environments, finding comfort in predictability and familiar patterns. Social interactions are complex terrains for them; they tend to focus on minute details rather than the broader context, which can make conversations challenging. Often, they struggle with non-verbal cues and facial expressions, leading to misinterpretations and social friction. They are introspective and analytical, occasionally relying on logic systems like flowcharts to decipher social dynamics. This approach can result in conversational misunderstandings, as sarcasm and jokes may fly past them due to a literal interpretation of language. Sensory experiences can be intense, with bright lights, loud noises, and certain textures overwhelming them. Despite societal perceptions of emotional distance, they deeply value fairness and often feel others' suffering more acutely than expressed. When stressed, repetitive behaviors provide solace, and they are passionate about specific interests, sometimes bordering on obsession. While they may be seen as distant or odd, their world is rich with patterns and logical intricacies that offer a distinct way of understanding the universe around them.
    """,
    NEUROTYPE_RIFFER: """
      This individual is comfortable with ambiguity and can easily navigate unstructured environments. Social cues, including non-verbal signals and facial expressions, are intuitive to them, allowing for smooth conversations and connections. They are adaptable in social settings, often picking up on subtle hints and adjusting their communication style accordingly. They may rely on shared cultural references and humor to build rapport, and they typically find it easy to engage in small talk. Sensory experiences are generally manageable for them, and they may not be as affected by stimuli that others find overwhelming. They value social harmony and are often skilled at reading the emotional states of others, responding with empathy. While they may not have the intense focus on specific interests that some neurodivergent individuals do, they can still be passionate about hobbies or topics. Their social world is rich with connections and shared understanding, allowing them to navigate complex social landscapes with relative ease.This person navigates the social world with ease and adaptability, comfortably engaging with others and adjusting their behavior across a wide range of social situations. Rather than relying heavily on routines, they thrive in dynamic environments where expectations may shift and interactions are spontaneous. They naturally focus on the broader context of situations, easily grasping overarching concepts while still recalling relevant details when needed. Social interactions tend to feel intuitive and energizing. Reading non-verbal cues, understanding implied meanings, and participating in spontaneous conversations happen with little conscious effort. They often enjoy social activities and feel comfortable in environments with varying levels of stimulation, adapting smoothly to busy gatherings or unfamiliar settings. Instead of relying primarily on logic or rehearsed scripts, they communicate fluidly and express themselves naturally in conversation. Their communication style is generally flexible and nuanced, allowing them to interpret subtleties in language and avoid common misunderstandings. They are able to express empathy and fairness openly, and these feelings are usually conveyed clearly to others. Their interests tend to be diverse and socially integrated rather than intensely focused on narrow topics, helping them connect easily with different groups of people. Because their behavior aligns closely with common social expectations, forming relationships and integrating into social environments usually happens naturally. While they still value personal interests and individuality, they are comfortable balancing these with the social rhythms and spontaneity of everyday interactions.
    """,
}
