"""
LLM (Large Language Model) API providers.

Available providers:
- openai_llm - OpenAI GPT models
- anthropic_llm - Anthropic Claude models
- gemini_llm - Google Gemini models
- deepseek_llm - DeepSeek models
- kimi_llm - Moonshot Kimi models
- zhipuai_llm - ZhipuAI GLM models
- siliconflow_llm - SiliconFlow models
- minimax_llm - MiniMax models
- yiyan_llm - Baidu Yiyan models
- dashscope_llm - Alibaba DashScope (Tongyi Qianwen)
- xunfei_llm - Xunfei Spark
- groq_llm - Groq inference
- together_llm - Together AI
- cohere_llm - Cohere models
- mistral_llm - Mistral AI models
- modelscope_llm - ModelScope models
- perplexity_llm - Perplexity models
- azure_openai_llm - Azure OpenAI models

Usage:
    >>> from model_api_hub.api.llm import deepseek_chat, deepseek_chat_stream
    >>> response = deepseek_chat("Hello!")
    >>> for chunk in deepseek_chat_stream("Hello!"):
    ...     print(chunk, end="")
"""

# OpenAI
from .openai_llm import (
    chat as openai_chat,
    chat_stream as openai_chat_stream,
    create_client as openai_client
)

# Anthropic
from .anthropic_llm import (
    chat as anthropic_chat,
    create_client as anthropic_client
)

# Google Gemini
from .gemini_llm import (
    chat as gemini_chat,
    create_client as gemini_client
)

# DeepSeek
from .deepseek_llm import (
    chat as deepseek_chat,
    chat_stream as deepseek_chat_stream,
    create_client as deepseek_client
)

# Kimi (Moonshot)
from .kimi_llm import (
    chat as kimi_chat,
    create_client as kimi_client
)

# ZhipuAI
from .zhipuai_llm import (
    chat as zhipuai_chat,
    create_client as zhipuai_client
)

# SiliconFlow
from .siliconflow_llm import (
    chat as siliconflow_chat,
    create_client as siliconflow_client
)

# MiniMax
from .minimax_llm import (
    chat as minimax_chat,
    create_client as minimax_client
)

# Baidu Yiyan
from .yiyan_llm import (
    chat as yiyan_chat
)

# Alibaba DashScope
from .dashscope_llm import (
    chat as dashscope_chat,
    create_client as dashscope_client
)

# Xunfei Spark
from .xunfei_llm import (
    chat as xunfei_chat
)

# Groq
from .groq_llm import (
    chat as groq_chat,
    create_client as groq_client
)

# Together AI
from .together_llm import (
    chat as together_chat,
    create_client as together_client
)

# Cohere
from .cohere_llm import (
    chat as cohere_chat,
    create_client as cohere_client
)

# Mistral
from .mistral_llm import (
    chat as mistral_chat,
    create_client as mistral_client
)

# ModelScope
from .modelscope_llm import (
    chat as modelscope_chat,
    create_client as modelscope_client
)

# Perplexity
from .perplexity_llm import (
    chat as perplexity_chat
)

# Azure OpenAI
from .azure_openai_llm import (
    chat as azure_chat
)

__all__ = [
    # OpenAI
    "openai_chat",
    "openai_chat_stream",
    "openai_client",
    # Anthropic
    "anthropic_chat",
    "anthropic_client",
    # Gemini
    "gemini_chat",
    "gemini_client",
    # DeepSeek
    "deepseek_chat",
    "deepseek_chat_stream",
    "deepseek_client",
    # Kimi
    "kimi_chat",
    "kimi_client",
    # ZhipuAI
    "zhipuai_chat",
    "zhipuai_client",
    # SiliconFlow
    "siliconflow_chat",
    "siliconflow_client",
    # MiniMax
    "minimax_chat",
    "minimax_client",
    # Yiyan
    "yiyan_chat",
    # DashScope
    "dashscope_chat",
    "dashscope_client",
    # Xunfei
    "xunfei_chat",
    # Groq
    "groq_chat",
    "groq_client",
    # Together
    "together_chat",
    "together_client",
    # Cohere
    "cohere_chat",
    "cohere_client",
    # Mistral
    "mistral_chat",
    "mistral_client",
    # ModelScope
    "modelscope_chat",
    "modelscope_client",
    # Perplexity
    "perplexity_chat",
    # Azure
    "azure_chat",
]
