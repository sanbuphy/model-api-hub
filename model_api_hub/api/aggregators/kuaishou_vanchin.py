"""
快手万擎 Vanchin API client - 企业级大模型服务平台

Website: https://vanchin.kuaishou.com/
Features: KAT-Coder 编程模型、CodeFlicker 智能开发工具
Pricing: 企业级服务
"""

import os
from typing import Optional, List, Dict, Any
from openai import OpenAI

# API Configuration
API_KEY: str = os.getenv("KUAISHOU_VANCHIN_API_KEY", "")
BASE_URL: str = os.getenv("KUAISHOU_VANCHIN_BASE_URL", "https://api.vanchin.kuaishou.com/v1")
DEFAULT_MODEL: str = "kat-coder-air-v1"

# Available models
AVAILABLE_MODELS: List[str] = [
    "kat-coder-air-v1",
    "kat-coder-pro-v1",
    "kat-coder-max-v1",
]


def create_client(api_key: Optional[str] = None) -> OpenAI:
    """Create 快手万擎 client."""
    key = api_key or API_KEY
    if not key:
        raise ValueError("快手万擎 API key is required. Set KUAISHOU_VANCHIN_API_KEY environment variable.")
    
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
    Chat using 快手万擎.
    
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
    
    parser = argparse.ArgumentParser(description="Test 快手万擎")
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
