"""
金山云星流 API client - 一站式 AI 训推全流程平台

Website: https://www.ksyun.com/product/ai
Features: 模型API服务、AI训推全流程、具身智能支持
Pricing: 按量付费
"""

import os
from typing import Optional, List, Dict, Any
from openai import OpenAI

# API Configuration
API_KEY: str = os.getenv("KSYUN_STARFLOW_API_KEY", "")
BASE_URL: str = os.getenv("KSYUN_STARFLOW_BASE_URL", "https://api.starflow.ksyun.com/v1")
DEFAULT_MODEL: str = "xiaomimimo-v2-flash"

# Available models
AVAILABLE_MODELS: List[str] = [
    "xiaomimimo-v2-flash",
    "deepseek-chat",
    "deepseek-reasoner",
    "glm-4",
    "qwen-max",
    "llama-3.1-70b",
]


def create_client(api_key: Optional[str] = None) -> OpenAI:
    """Create 金山云星流 client."""
    key = api_key or API_KEY
    if not key:
        raise ValueError("金山云星流 API key is required. Set KSYUN_STARFLOW_API_KEY environment variable.")
    
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
    Chat using 金山云星流.
    
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
    
    parser = argparse.ArgumentParser(description="Test 金山云星流")
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
