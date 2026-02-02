"""
OpenRouter API client - Universal API for LLMs

Website: https://openrouter.ai/
Features: Access 100+ models from different providers with unified API
Pricing: Pay-as-you-go, competitive rates
"""

import os
from typing import Optional, List, Dict, Any
from openai import OpenAI

# API Configuration
API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
BASE_URL: str = "https://openrouter.ai/api/v1"
DEFAULT_MODEL: str = "anthropic/claude-3.5-sonnet"

# Available models (partial list - check https://openrouter.ai/docs#models for full list)
AVAILABLE_MODELS: List[str] = [
    # Anthropic
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-opus",
    "anthropic/claude-3-haiku",
    # OpenAI
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-4-turbo",
    "openai/gpt-3.5-turbo",
    # Google
    "google/gemini-pro-1.5",
    "google/gemini-flash-1.5",
    # Meta
    "meta-llama/llama-3.1-405b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    # Mistral
    "mistralai/mistral-large",
    "mistralai/mistral-medium",
    # Other popular models
    "microsoft/wizardlm-2-8x22b",
    "nousresearch/hermes-3-llama-3.1-405b",
    "perplexity/llama-3.1-sonar-large-128k-online",
    "deepseek/deepseek-chat",
]


def create_client(api_key: Optional[str] = None) -> OpenAI:
    """Create OpenRouter client."""
    key = api_key or API_KEY
    if not key:
        raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable.")
    
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
    Chat using OpenRouter.
    
    Args:
        message: User message
        model: Model identifier (e.g., 'anthropic/claude-3.5-sonnet')
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


def list_models(api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """List available models from OpenRouter."""
    import requests
    
    key = api_key or API_KEY
    headers = {"Authorization": f"Bearer {key}"}
    
    response = requests.get(f"{BASE_URL}/models", headers=headers)
    response.raise_for_status()
    
    return response.json().get("data", [])


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test OpenRouter")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--message", default="Hello! What can you do?", help="Message")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    args = parser.parse_args()
    
    if args.list_models:
        models = list_models(args.api_key)
        print(f"Available models: {len(models)}")
        for model in models[:10]:
            print(f"  - {model.get('id')}: {model.get('name')}")
    elif args.api_key:
        print(f"Using model: {args.model}")
        response = chat(args.message, api_key=args.api_key, model=args.model)
        print(f"Response: {response}")
    else:
        print("Please provide --api-key or use --list-models")
