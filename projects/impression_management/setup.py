"""Setup functions for LLM, embedder, and memory."""

import sentence_transformers

from concordia.associative_memory import basic_associative_memory
from concordia.language_model import utils as language_model_utils

from projects.impression_management.config import ConversationConfig


def setup_language_model(config: ConversationConfig, api_key: str):
    """Setup and return language model."""
    if config.llm_type == 'openai':
        return language_model_utils.language_model_setup(
            api_type='openai',
            model_name=config.model,
            api_key=api_key,
            disable_language_model=False,
        )
    else:
        # Local Ollama model
        from concordia.language_model import ollama_language_model
        return ollama_language_model.OllamaLanguageModel(
            model_name=config.local_model,
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
