"""
SophNet API client - 算能科技 AI 云平台

Website: https://sophnet.com/
Features: DeepSeek 极速版、RISC-V 芯片优化
Pricing: 按量付费
"""

import os
from typing import Optional, List, Dict, Any
from openai import OpenAI

# API Configuration
API_KEY: str = os.getenv("SOPHNET_API_KEY", "")
BASE_URL: str = os.getenv("SOPHNET_BASE_URL", "https://api.sophnet.com/v1")
DEFAULT_MODEL: str = "deepseek-v3"

# Available models
AVAILABLE_MODELS: List[str] = [
    "deepseek-v3",
    "deepseek-r1",
    "deepseek-r1-full",
    "deepseek-coder",
    "qwen-2-72b",
    "qwen-2-7b",
    "llama-3.1-70b",
    "llama-3.1-8b",
]


def create_client(api_key: Optional[str] = None) -> OpenAI:
    """Create SophNet client."""
    key = api_key or API_KEY
    if not key:
        raise ValueError("SophNet API key is required. Set SOPHNET_API_KEY environment variable.")
    
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
    Chat using SophNet.
    
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
    
    parser = argparse.ArgumentParser(description="Test SophNet")
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
