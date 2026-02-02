"""
Fireworks AI API client - Fast inference for open-source models

Website: https://fireworks.ai/
Features: Optimized inference for Llama, Mixtral, Qwen, etc.
Pricing: Pay-as-you-go with competitive rates
"""

import os
from typing import Optional, List, Dict, Any
from openai import OpenAI

# API Configuration
API_KEY: str = os.getenv("FIREWORKS_API_KEY", "")
BASE_URL: str = "https://api.fireworks.ai/inference/v1"
DEFAULT_MODEL: str = "accounts/fireworks/models/llama-v3p1-405b-instruct"

# Available models
AVAILABLE_MODELS: List[str] = [
    # Llama 3.1
    "accounts/fireworks/models/llama-v3p1-405b-instruct",
    "accounts/fireworks/models/llama-v3p1-70b-instruct",
    "accounts/fireworks/models/llama-v3p1-8b-instruct",
    # Llama 3
    "accounts/fireworks/models/llama-v3-70b-instruct",
    "accounts/fireworks/models/llama-v3-8b-instruct",
    # Mixtral
    "accounts/fireworks/models/mixtral-8x22b-instruct",
    "accounts/fireworks/models/mixtral-8x7b-instruct",
    # Qwen
    "accounts/fireworks/models/qwen2p5-72b-instruct",
    "accounts/fireworks/models/qwen2-72b-instruct",
    # Other
    "accounts/fireworks/models/yi-large",
    "accounts/fireworks/models/deepseek-v3",
    "accounts/fireworks/models/deepseek-coder-v2",
    "accounts/fireworks/models/starcoder-16b",
]


def create_client(api_key: Optional[str] = None) -> OpenAI:
    """Create Fireworks AI client."""
    key = api_key or API_KEY
    if not key:
        raise ValueError("Fireworks API key is required. Set FIREWORKS_API_KEY environment variable.")
    
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
    Chat using Fireworks AI.
    
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
    
    parser = argparse.ArgumentParser(description="Test Fireworks AI")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--message", default="Hello! What can you do?", help="Message")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model")
    args = parser.parse_args()
    
    if args.api_key:
        print(f"Using model: {args.model}")
        response = chat(args.message, api_key=args.api_key, model=args.model)
        print(f"Response: {response}")
    else:
        print("Please provide --api-key")
