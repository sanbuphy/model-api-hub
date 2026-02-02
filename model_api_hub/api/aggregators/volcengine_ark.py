"""
火山方舟 Volcengine Ark API client - 字节跳动大模型服务平台

Website: https://www.volcengine.com/product/ark
Features: 模型精调、评测、推理、MaaS 平台
Pricing: 按量付费
"""

import os
from typing import Optional, List, Dict, Any
from openai import OpenAI

# API Configuration
API_KEY: str = os.getenv("VOLCENGINE_ARK_API_KEY", "")
BASE_URL: str = os.getenv("VOLCENGINE_ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
DEFAULT_MODEL: str = "doubao-pro-32k"

# Available models
AVAILABLE_MODELS: List[str] = [
    # 豆包系列
    "doubao-pro-32k",
    "doubao-pro-128k",
    "doubao-lite-32k",
    "doubao-lite-128k",
    "doubao-vision-32k",
    # DeepSeek
    "deepseek-r1",
    "deepseek-v3",
    "deepseek-coder",
    # 其他
    "glm-4",
    "qwen-max",
    "llama-3.1-70b",
    "abab6.5s-chat",
]


def create_client(api_key: Optional[str] = None) -> OpenAI:
    """Create 火山方舟 client."""
    key = api_key or API_KEY
    if not key:
        raise ValueError("火山方舟 API key is required. Set VOLCENGINE_ARK_API_KEY environment variable.")
    
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
    Chat using 火山方舟.
    
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
    
    parser = argparse.ArgumentParser(description="Test 火山方舟")
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
