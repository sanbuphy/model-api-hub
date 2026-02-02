"""
蓝耘元生代 MaaS API client - 高性能大模型服务平台

Website: https://www.lanyun.net/
Features: DeepSeek、GLM、Kimi 等高性能推理
Pricing: 按量付费，千万 Token 福利
"""

import os
from typing import Optional, List, Dict, Any
from openai import OpenAI

# API Configuration
API_KEY: str = os.getenv("LANYUN_MAAS_API_KEY", "")
BASE_URL: str = os.getenv("LANYUN_MAAS_BASE_URL", "https://api.lanyun.net/v1")
DEFAULT_MODEL: str = "deepseek-v3.2"

# Available models
AVAILABLE_MODELS: List[str] = [
    "deepseek-v3.2",
    "deepseek-r1",
    "deepseek-chat",
    "glm-4.7",
    "glm-4",
    "kimi-k2",
    "kimi-v1",
    "qwen2.5-72b",
    "llama-3.1-405b",
]


def create_client(api_key: Optional[str] = None) -> OpenAI:
    """Create 蓝耘元生代 client."""
    key = api_key or API_KEY
    if not key:
        raise ValueError("蓝耘元生代 API key is required. Set LANYUN_MAAS_API_KEY environment variable.")
    
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
    Chat using 蓝耘元生代.
    
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
    
    parser = argparse.ArgumentParser(description="Test 蓝耘元生代")
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
