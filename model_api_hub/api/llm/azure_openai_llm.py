"""
Azure OpenAI API client.

Official API: https://learn.microsoft.com/en-us/azure/ai-services/openai/
Note: Requires Azure OpenAI resource deployment
"""

import os
from typing import Optional, Dict, Any, List
from openai import AzureOpenAI

# API Configuration
API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
DEFAULT_MODEL: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")

# Available models (deployment names may vary)
AVAILABLE_MODELS: List[str] = [
    "gpt-4",
    "gpt-4-32k",
    "gpt-4-turbo",
    "gpt-35-turbo",
    "gpt-35-turbo-16k",
]


def create_client(
    api_key: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    api_version: Optional[str] = None,
) -> AzureOpenAI:
    """Create Azure OpenAI client."""
    key = api_key or API_KEY
    endpoint = azure_endpoint or AZURE_ENDPOINT
    
    if not key:
        raise ValueError("Azure OpenAI API key is required. Set AZURE_OPENAI_API_KEY environment variable or pass api_key parameter.")
    if not endpoint:
        raise ValueError("Azure OpenAI endpoint is required. Set AZURE_OPENAI_ENDPOINT environment variable or pass azure_endpoint parameter.")
    
    return AzureOpenAI(
        api_key=key,
        azure_endpoint=endpoint,
        api_version=api_version or API_VERSION,
    )


def chat(
    message: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    azure_endpoint: Optional[str] = None,
    api_version: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs
) -> str:
    """
    Chat with Azure OpenAI.
    
    Args:
        message: User message
        model: Deployment name (default: from AZURE_OPENAI_DEPLOYMENT env var)
        api_key: API key (optional)
        azure_endpoint: Azure endpoint (optional)
        api_version: API version (optional)
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens in response
        **kwargs: Additional parameters
    
    Returns:
        Response text
    """
    client = create_client(api_key, azure_endpoint, api_version)
    
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
    azure_endpoint: Optional[str] = None,
    api_version: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs
):
    """
    Stream chat with Azure OpenAI.
    
    Yields:
        Text chunks as they arrive
    """
    client = create_client(api_key, azure_endpoint, api_version)
    
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
    
    parser = argparse.ArgumentParser(description="Test Azure OpenAI")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--endpoint", help="Azure endpoint")
    parser.add_argument("--message", default="Hello! Who are you?", help="Message to send")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Deployment name")
    parser.add_argument("--stream", action="store_true", help="Stream response")
    args = parser.parse_args()
    
    if args.api_key and args.endpoint:
        if args.stream:
            print(f"Streaming response from {args.model}:")
            for chunk in stream_chat(args.message, api_key=args.api_key, azure_endpoint=args.endpoint, model=args.model):
                print(chunk, end="", flush=True)
            print()
        else:
            print(f"Response from {args.model}:")
            response = chat(args.message, api_key=args.api_key, azure_endpoint=args.endpoint, model=args.model)
            print(response)
    else:
        print("Please provide --api-key and --endpoint")
