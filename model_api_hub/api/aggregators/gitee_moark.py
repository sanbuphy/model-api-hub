"""
模力方舟 Gitee MoArk API client - Gitee AI 模型广场

Website: https://ai.gitee.com/
Docs: https://ai.gitee.com/docs/getting-started
Features: Serverless API、免费大模型、文心大模型4.5
Pricing: 免费额度 + 按量付费
"""

import os
from typing import Optional, List, Dict, Any
from openai import OpenAI

# API Configuration
API_KEY: str = os.getenv("GITEE_MOARK_API_KEY", "")
BASE_URL: str = os.getenv("GITEE_MOARK_BASE_URL", "https://ai.gitee.com/v1")
DEFAULT_MODEL: str = "Qwen2.5-72B-Instruct"

# Available models
AVAILABLE_MODELS: List[str] = [
    # Qwen
    "Qwen2.5-72B-Instruct",
    "Qwen2.5-32B-Instruct",
    "Qwen2.5-14B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen2-72B-Instruct",
    # Llama
    "Meta-Llama-3.1-405B-Instruct",
    "Meta-Llama-3.1-70B-Instruct",
    "Meta-Llama-3.1-8B-Instruct",
    # DeepSeek
    "DeepSeek-V2.5",
    "DeepSeek-Coder-V2",
    # GLM
    "glm-4-9b-chat",
    "chatglm3-6b",
    # Baidu
    "ernie-4.5",
    "ernie-bot-4",
    "ernie-bot",
    # Other
    "Yi-1.5-34B-Chat",
    "gemma-2-27b-it",
    "Mixtral-8x7B-Instruct",
]


def create_client(api_key: Optional[str] = None) -> OpenAI:
    """Create 模力方舟 client."""
    key = api_key or API_KEY
    if not key:
        raise ValueError("模力方舟 API key is required. Set GITEE_MOARK_API_KEY environment variable.")
    
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
    Chat using 模力方舟.
    
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
    
    parser = argparse.ArgumentParser(description="Test 模力方舟")
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
