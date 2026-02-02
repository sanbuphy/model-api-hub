"""
LLM (Large Language Model) API providers.

Available providers:
- openai_llm - OpenAI GPT models
- gemini_llm - Google Gemini models
- deepseek_llm - DeepSeek models
- dashscope_llm - Alibaba DashScope (Tongyi Qianwen)
- xunfei_llm - Xunfei Spark
- groq_llm - Groq inference
- together_llm - Together AI
- cohere_llm - Cohere models
- mistral_llm - Mistral AI models
- modelscope_llm - ModelScope models
"""

from .openai_llm import chat as openai_chat, create_client as openai_client
from .gemini_llm import chat as gemini_chat, create_client as gemini_client
from .deepseek_llm import chat as deepseek_chat, create_client as deepseek_client
from .dashscope_llm import chat as dashscope_chat, create_client as dashscope_client
from .xunfei_llm import chat as xunfei_chat
from .groq_llm import chat as groq_chat, create_client as groq_client
from .together_llm import chat as together_chat, create_client as together_client
from .cohere_llm import chat as cohere_chat, create_client as cohere_client
from .mistral_llm import chat as mistral_chat, create_client as mistral_client
from .modelscope_llm import chat as modelscope_chat, create_client as modelscope_client

__all__ = [
    "openai_chat",
    "openai_client",
    "gemini_chat",
    "gemini_client",
    "deepseek_chat",
    "deepseek_client",
    "dashscope_chat",
    "dashscope_client",
    "xunfei_chat",
    "groq_chat",
    "groq_client",
    "together_chat",
    "together_client",
    "cohere_chat",
    "cohere_client",
    "mistral_chat",
    "mistral_client",
    "modelscope_chat",
    "modelscope_client",
]
