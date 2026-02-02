"""
无问芯穹 Infinigence API client - 企业级 AI 大模型服务平台

Website: https://www.infinigence.com/
Features: 异构算力集群、推理加速、Qwen3 加速版
Pricing: 免费额度 + 按量付费
"""

import os
from typing import Optional, List, Dict, Any
from openai import OpenAI

# API Configuration
API_KEY: str = os.getenv("INFINIGENCE_API_KEY", "")
BASE_URL: str = os.getenv("INFINIGENCE_BASE_URL", "https://api.infinigence.com/v1")
DEFAULT_MODEL: str = "qwen3-32b"

# Available models
AVAILABLE_MODELS: List[str] = [
    "qwen3-32b",
    "qwen3-14b",
    "qwen3-8b",
    "qwen3-4b",
    "qwen3-1.5b",
    "qwen3-0.5b",
    "qwen2.5-72b-instruct",
    "qwen2.5-32b-instruct",
    "qwen2.5-14b-instruct",
    "qwen2.5-7b-instruct",
    "deepseek-chat",
    "deepseek-coder",
    "llama-3.1-70b",
    "llama-3.1-8b",
]


def create_client(api_key: Optional[str] = None) -> OpenAI:
    """Create 无问芯穹 client."""
    key = api_key or API_KEY
    if not key:
        raise ValueError("无问芯穹 API key is required. Set INFINIGENCE_API_KEY environment variable.")
    
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
    Chat using 无问芯穹.
    
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
    
    parser = argparse.ArgumentParser(description="Test 无问芯穹")
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
