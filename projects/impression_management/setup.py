"""Setup functions for LLM, embedder, and memory."""

from concordia.associative_memory import basic_associative_memory
from concordia.language_model import gpt_model
import sentence_transformers

from projects.impression_management.config import ConversationConfig


def setup_language_model(config: ConversationConfig, api_key: str):
    """Setup and return OpenAI language model."""
    if config.llm_type != 'openai':
        raise ValueError('Only llm_type=openai is supported in this setup.')

    return gpt_model.GptLanguageModel(
        model_name=config.model,
        api_key=api_key,
    )


def setup_embedder_and_memory():
    """Setup sentence embedder and memory bank."""
    st_model = sentence_transformers.SentenceTransformer(
        'sentence-transformers/all-mpnet-base-v2'
    )
    embedder = lambda x: st_model.encode(x, show_progress_bar=False)
    memory_bank = basic_associative_memory.AssociativeMemoryBank(
        sentence_embedder=embedder
    )
    return embedder, memory_bank
