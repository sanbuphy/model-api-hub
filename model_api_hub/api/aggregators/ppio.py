"""
PPIO 派欧云 API client - 一站式 AI 云服务平台

Website: https://ppio.cn/
API Docs: https://ppio.cn/docs/model/llm
Features: 大模型 API 服务、GPU 容器实例、Serverless
Pricing: 按量付费，支持 DeepSeek、Llama、Qwen 等
"""

import os
from typing import Optional, List, Dict, Any
from openai import OpenAI

# API Configuration
API_KEY: str = os.getenv("PPIO_API_KEY", "")
BASE_URL: str = os.getenv("PPIO_BASE_URL", "https://api.ppinfra.com/v3/openai")
DEFAULT_MODEL: str = "deepseek/deepseek-chat"

# Available models
AVAILABLE_MODELS: List[str] = [
    # DeepSeek
    "deepseek/deepseek-chat",
    "deepseek/deepseek-coder",
    "deepseek/deepseek-reasoner",
    # Llama
    "meta-llama/llama-3.1-405b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3-70b-instruct",
    "meta-llama/llama-3-8b-instruct",
    # Qwen
    "qwen/qwen-2-72b-instruct",
    "qwen/qwen-2-7b-instruct",
    "qwen/qwen1.5-110b-chat",
    # Mistral
    "mistralai/mixtral-8x7b-instruct",
    "mistralai/mistral-7b-instruct",
    # Other
    "google/gemma-2-27b-it",
    "google/gemma-2-9b-it",
]


def create_client(api_key: Optional[str] = None) -> OpenAI:
    """Create PPIO client."""
    key = api_key or API_KEY
    if not key:
        raise ValueError("PPIO API key is required. Set PPIO_API_KEY environment variable.")
    
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
    Chat using PPIO.
    
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
    
    parser = argparse.ArgumentParser(description="Test PPIO")
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
