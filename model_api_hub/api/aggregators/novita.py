"""
Novita AI API client - 国内AI模型聚合平台

Website: https://novita.ai/
Features: 支持多种开源模型，价格优惠
"""

import os
from typing import Optional, List, Dict, Any
from openai import OpenAI

# API Configuration
API_KEY: str = os.getenv("NOVITA_API_KEY", "")
BASE_URL: str = "https://api.novita.ai/v3/openai"
DEFAULT_MODEL: str = "meta-llama/llama-3.1-405b-instruct"

# Available models
AVAILABLE_MODELS: List[str] = [
    "meta-llama/llama-3.1-405b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3-70b-instruct",
    "meta-llama/llama-3-8b-instruct",
    "mistralai/mixtral-8x7b-instruct",
    "mistralai/mistral-7b-instruct",
    "google/gemma-2-27b-it",
    "google/gemma-2-9b-it",
    "qwen/qwen-2-72b-instruct",
    "qwen/qwen-2-7b-instruct",
    "deepseek/deepseek-chat",
    "deepseek/deepseek-coder",
]


def create_client(api_key: Optional[str] = None) -> OpenAI:
    """Create Novita AI client."""
    key = api_key or API_KEY
    if not key:
        raise ValueError("Novita API key is required. Set NOVITA_API_KEY environment variable.")
    
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
    Chat using Novita AI.
    
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
    
    parser = argparse.ArgumentParser(description="Test Novita AI")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--message", default="Hello! Who are you?", help="Message")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model")
    args = parser.parse_args()
    
    if args.api_key:
        print(f"Using model: {args.model}")
        response = chat(args.message, api_key=args.api_key, model=args.model)
        print(f"Response: {response}")
    else:
        print("Please provide --api-key")
