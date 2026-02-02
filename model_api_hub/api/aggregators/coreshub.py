"""
基石智算 CoresHub API client - 青云科技 AI 算力云

Website: https://www.qingcloud.com/products/ai
Features: AI 算力云服务，支持 Kimi、MiniMax、GLM、Qwen 等
Pricing: 按量付费
"""

import os
from typing import Optional, List, Dict, Any
from openai import OpenAI

# API Configuration
API_KEY: str = os.getenv("CORESHUB_API_KEY", "")
BASE_URL: str = os.getenv("CORESHUB_BASE_URL", "https://api.coreshub.com/v1")
DEFAULT_MODEL: str = "kimi-k2.5"

# Available models
AVAILABLE_MODELS: List[str] = [
    # Kimi
    "kimi-k2.5",
    "kimi-k2",
    "kimi-v1",
    # MiniMax
    "minimax-m2.1",
    "minimax-m2",
    "abab6.5s-chat",
    # GLM
    "glm-4.7",
    "glm-4",
    "glm-4-plus",
    # Qwen
    "qwen3-30b-a3b-instruct",
    "qwen2.5-72b-instruct",
    "qwen2-72b-instruct",
    # DeepSeek
    "deepseek-chat",
    "deepseek-reasoner",
]


def create_client(api_key: Optional[str] = None) -> OpenAI:
    """Create 基石智算 client."""
    key = api_key or API_KEY
    if not key:
        raise ValueError("基石智算 API key is required. Set CORESHUB_API_KEY environment variable.")
    
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
    Chat using 基石智算.
    
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


def stream_chat(
    message: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs
):
    """Stream chat response."""
    client = create_client(api_key)
    
    messages = [{"role": "user", "content": message}]
    
    request_params = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
    }
    
    if max_tokens:
        request_params["max_tokens"] = max_tokens
    
    request_params.update(kwargs)
    
    response = client.chat.completions.create(**request_params)
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test 基石智算")
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
