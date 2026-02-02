"""
并行智算云 API client - 并行科技 AI 大模型平台

Website: https://ai.paratera.com/
Docs: https://ai.paratera.com/document/llm/quickStart/useApi
Features: 模型广场、API 调用、智算云服务
Pricing: 按量付费
"""

import os
from typing import Optional, List, Dict, Any
from openai import OpenAI

# API Configuration
API_KEY: str = os.getenv("PARATERA_AI_API_KEY", "")
BASE_URL: str = os.getenv("PARATERA_AI_BASE_URL", "https://ai.paratera.com/api/v1")
DEFAULT_MODEL: str = "deepseek-chat"

# Available models
AVAILABLE_MODELS: List[str] = [
    "deepseek-chat",
    "deepseek-reasoner",
    "deepseek-coder",
    "glm-4",
    "qwen-max",
    "qwen-plus",
    "llama-3.1-70b",
    "llama-3.1-8b",
    "abab6.5s-chat",
]


def create_client(api_key: Optional[str] = None) -> OpenAI:
    """Create 并行智算云 client."""
    key = api_key or API_KEY
    if not key:
        raise ValueError("并行智算云 API key is required. Set PARATERA_AI_API_KEY environment variable.")
    
    return OpenAI(
        base_url=BASE_URL,
        api_key=key,
    )


def chat(
    message: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs
) -> str:
    """
    Chat using 并行智算云.
    
    Args:
        message: User message
        model: Model identifier
        api_key: API key (optional)
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        **kwargs: Additional parameters
    
    Returns:
        Response text
    """
    client = create_client(api_key)
    
    messages = [{"role": "user", "content": message}]
    
    request_params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    
    if max_tokens:
        request_params["max_tokens"] = max_tokens
    
    request_params.update(kwargs)
    
    response = client.chat.completions.create(**request_params)
    return response.choices[0].message.content


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test 并行智算云")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--message", default="你好！请介绍一下自己。", help="Message")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model")
    args = parser.parse_args()
    
    if args.api_key:
        print(f"Using model: {args.model}")
        response = chat(args.message, api_key=args.api_key, model=args.model)
        print(f"Response: {response}")
    else:
        print("Please provide --api-key")
