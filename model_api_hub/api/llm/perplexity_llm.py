"""
Perplexity API client.

Official API: https://docs.perplexity.ai/
Supported models: llama-3.1-sonar-small, llama-3.1-sonar-large, etc.
Note: Perplexity provides search-augmented responses
"""

import os
from typing import Optional, Dict, Any, List
from openai import OpenAI

# API Configuration
API_KEY: str = os.getenv("PERPLEXITY_API_KEY", "")
BASE_URL: str = os.getenv("PERPLEXITY_BASE_URL", "https://api.perplexity.ai")
DEFAULT_MODEL: str = "llama-3.1-sonar-large-128k-online"

# Available models
AVAILABLE_MODELS: List[str] = [
    "llama-3.1-sonar-small-128k-online",
    "llama-3.1-sonar-large-128k-online",
    "llama-3.1-sonar-huge-128k-online",
    "llama-3.1-sonar-small-128k-chat",
    "llama-3.1-sonar-large-128k-chat",
    "llama-3.1-8b-instruct",
    "llama-3.1-70b-instruct",
]


def create_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> OpenAI:
    """Create Perplexity client (OpenAI-compatible)."""
    key = api_key or API_KEY
    if not key:
        raise ValueError("Perplexity API key is required. Set PERPLEXITY_API_KEY environment variable or pass api_key parameter.")
    
    return OpenAI(
        api_key=key,
        base_url=base_url or BASE_URL,
    )


def chat(
    message: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    return_citations: bool = False,
    **kwargs
) -> str:
    """
    Chat with Perplexity (search-augmented).
    
    Args:
        message: User message
        model: Model name (default: llama-3.1-sonar-large-128k-online)
        api_key: API key (optional)
        base_url: Base URL (optional)
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens in response
        return_citations: If True, returns citations in response
        **kwargs: Additional parameters
    
    Returns:
        Response text (with citations if return_citations=True)
    """
    client = create_client(api_key, base_url)
    
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
    
    content = response.choices[0].message.content
    
    if return_citations and hasattr(response, 'citations'):
        citations = response.citations if hasattr(response, 'citations') else []
        return f"{content}\n\nCitations:\n" + "\n".join([f"- {c}" for c in citations])
    
    return content


def search(
    query: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "llama-3.1-sonar-large-128k-online",
) -> Dict[str, Any]:
    """
    Search with Perplexity and return structured results.
    
    Returns:
        Dict with 'answer', 'citations', and 'usage'
    """
    client = create_client(api_key, base_url)
    
    messages = [{"role": "user", "content": query}]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )
    
    return {
        "answer": response.choices[0].message.content,
        "citations": getattr(response, 'citations', []),
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
            "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            "total_tokens": response.usage.total_tokens if response.usage else 0,
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Perplexity")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--message", default="What are the latest developments in AI?", help="Message to send")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--citations", action="store_true", help="Include citations")
    args = parser.parse_args()
    
    if args.api_key:
        print(f"Response from {args.model}:")
        response = chat(args.message, api_key=args.api_key, model=args.model, return_citations=args.citations)
        print(response)
    else:
        print("Please provide --api-key")
