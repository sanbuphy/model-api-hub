"""
Anthropic Claude API client.

Official API: https://docs.anthropic.com/claude/reference/getting-started-with-the-api
Supported models: claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-3-5-sonnet
"""

import os
from typing import Optional, Dict, Any, List
from anthropic import Anthropic

# API Configuration
API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
BASE_URL: str = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
DEFAULT_MODEL: str = "claude-3-5-sonnet-20241022"

# Available models
AVAILABLE_MODELS: List[str] = [
    "claude-3-opus-20240229",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "claude-3-haiku-20240307",
]


def create_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> Anthropic:
    """Create Anthropic client."""
    key = api_key or API_KEY
    if not key:
        raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")
    
    return Anthropic(
        api_key=key,
        base_url=base_url or BASE_URL,
    )


def chat(
    message: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    system: Optional[str] = None,
    **kwargs
) -> str:
    """
    Chat with Claude.
    
    Args:
        message: User message
        model: Model name (default: claude-3-5-sonnet)
        api_key: API key (optional, uses ANTHROPIC_API_KEY env var if not provided)
        base_url: Base URL (optional)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature (0-1)
        system: System prompt
        **kwargs: Additional parameters
    
    Returns:
        Response text
    """
    client = create_client(api_key, base_url)
    
    messages = [{"role": "user", "content": message}]
    
    request_params = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    
    if system:
        request_params["system"] = system
    
    request_params.update(kwargs)
    
    response = client.messages.create(**request_params)
    return response.content[0].text


def stream_chat(
    message: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    system: Optional[str] = None,
    **kwargs
):
    """
    Stream chat with Claude.
    
    Yields:
        Text chunks as they arrive
    """
    client = create_client(api_key, base_url)
    
    messages = [{"role": "user", "content": message}]
    
    request_params = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
        "stream": True,
    }
    
    if system:
        request_params["system"] = system
    
    request_params.update(kwargs)
    
    with client.messages.stream(**request_params) as stream:
        for text in stream.text_stream:
            yield text


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Anthropic Claude")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--message", default="Hello! Who are you?", help="Message to send")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--stream", action="store_true", help="Stream response")
    args = parser.parse_args()
    
    if args.api_key:
        if args.stream:
            print(f"Streaming response from {args.model}:")
            for chunk in stream_chat(args.message, api_key=args.api_key, model=args.model):
                print(chunk, end="", flush=True)
            print()
        else:
            print(f"Response from {args.model}:")
            response = chat(args.message, api_key=args.api_key, model=args.model)
            print(response)
    else:
        print("Please provide --api-key")
