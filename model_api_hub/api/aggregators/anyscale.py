"""
Anyscale API client - Endpoints for open-source LLMs

Website: https://www.anyscale.com/
Features: Production-ready inference for open-source models
Pricing: Pay-as-you-go
"""

import os
from typing import Optional, List, Dict, Any
from openai import OpenAI

# API Configuration
API_KEY: str = os.getenv("ANYSCALE_API_KEY", "")
BASE_URL: str = "https://api.endpoints.anyscale.com/v1"
DEFAULT_MODEL: str = "meta-llama/Meta-Llama-3-70B-Instruct"

# Available models
AVAILABLE_MODELS: List[str] = [
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "HuggingFaceH4/zephyr-7b-beta",
    "codellama/CodeLlama-70b-Instruct-hf",
]


def create_client(api_key: Optional[str] = None) -> OpenAI:
    """Create Anyscale client."""
    key = api_key or API_KEY
    if not key:
        raise ValueError("Anyscale API key is required. Set ANYSCALE_API_KEY environment variable.")
    
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
    Chat using Anyscale.
    
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
    
    parser = argparse.ArgumentParser(description="Test Anyscale")
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
