"""
七牛云 AI 大模型广场 API client - 中国版 OpenRouter

Website: https://www.qiniu.com/products/ai
Features: 统一API架构、多模型调度、Agent+MCP服务
Pricing: 按量付费
"""

import os
from typing import Optional, List, Dict, Any
from openai import OpenAI

# API Configuration
API_KEY: str = os.getenv("QINIU_AI_API_KEY", "")
BASE_URL: str = os.getenv("QINIU_AI_BASE_URL", "https://ai.qiniuapi.com/v1")
DEFAULT_MODEL: str = "deepseek-chat"

# Available models
AVAILABLE_MODELS: List[str] = [
    # DeepSeek
    "deepseek-chat",
    "deepseek-reasoner",
    "deepseek-coder",
    # Qwen
    "qwen-max",
    "qwen-plus",
    "qwen-turbo",
    "qwen2.5-72b-instruct",
    # GLM
    "glm-4",
    "glm-4-plus",
    "glm-4-flash",
    # Llama
    "llama-3.1-405b",
    "llama-3.1-70b",
    "llama-3.1-8b",
    # Other
    "moonshot-v1-128k",
    "abab6.5s-chat",
    "hunyuan-standard",
    "ernie-bot-4",
]


def create_client(api_key: Optional[str] = None) -> OpenAI:
    """Create 七牛云 AI client."""
    key = api_key or API_KEY
    if not key:
        raise ValueError("七牛云 AI API key is required. Set QINIU_AI_API_KEY environment variable.")
    
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
    Chat using 七牛云 AI.
    
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
    
    parser = argparse.ArgumentParser(description="Test 七牛云 AI")
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
