"""
API Aggregators / API中转站

提供统一的接口访问多个AI模型提供商。
"""

from .openrouter import chat as openrouter_chat, create_client as openrouter_client
from .siliconflow import chat as siliconflow_chat, create_client as siliconflow_client
from .poe import chat as poe_chat, create_client as poe_client
from .ai302 import chat as ai302_chat, create_client as ai302_client
from .fireworks import chat as fireworks_chat, create_client as fireworks_client
from .novita import chat as novita_chat, create_client as novita_client
from .groq import chat as groq_chat, create_client as groq_client
from .together import chat as together_chat, create_client as together_client
from .anyscale import chat as anyscale_chat, create_client as anyscale_client
from .perplexity import chat as perplexity_chat, create_client as perplexity_client
from .mistral import chat as mistral_chat, create_client as mistral_client
from .cohere import chat as cohere_chat, create_client as cohere_client
from .ai21 import chat as ai21_chat, create_client as ai21_client

__all__ = [
    "openrouter_chat", "openrouter_client",
    "siliconflow_chat", "siliconflow_client",
    "poe_chat", "poe_client",
    "ai302_chat", "ai302_client",
    "fireworks_chat", "fireworks_client",
    "novita_chat", "novita_client",
    "groq_chat", "groq_client",
    "together_chat", "together_client",
    "anyscale_chat", "anyscale_client",
    "perplexity_chat", "perplexity_client",
    "mistral_chat", "mistral_client",
    "cohere_chat", "cohere_client",
    "ai21_chat", "ai21_client",
]
