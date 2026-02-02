"""
Perplexity API client - Search-augmented LLM

Website: https://www.perplexity.ai/
Features: Real-time search + LLM for accurate answers
Pricing: Pay-as-you-go
"""

import os
from typing import Optional, List, Dict, Any
from openai import OpenAI

# API Configuration
API_KEY: str = os.getenv("PERPLEXITY_API_KEY", "")
BASE_URL: str = "https://api.perplexity.ai"
DEFAULT_MODEL: str = "llama-3.1-sonar-large-128k-online"

# Available models
AVAILABLE_MODELS: List[str] = [
    # Online models (with search)
    "llama-3.1-sonar-large-128k-online",
    "llama-3.1-sonar-small-128k-online",
    "llama-3.1-sonar-huge-128k-online",
    # Chat models (no search)
    "llama-3.1-sonar-large-128k-chat",
    "llama-3.1-sonar-small-128k-chat",
    # Open-source models
    "llama-3.1-70b-instruct",
    "llama-3.1-8b-instruct",
    "mixtral-8x7b-instruct",
]


def create_client(api_key: Optional[str] = None) -> OpenAI:
    """Create Perplexity client."""
    key = api_key or API_KEY
    if not key:
        raise ValueError("Perplexity API key is required. Set PERPLEXITY_API_KEY environment variable.")
    
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
    return_citations: bool = False,
    **kwargs
) -> str:
    """
    Chat using Perplexity (search-augmented).
    
    Args:
        message: User message
        model: Model identifier
        api_key: API key (optional)
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        return_citations: Include citations in response
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
    content = response.choices[0].message.content
    
    if return_citations and hasattr(response, 'citations'):
        citations = response.citations if hasattr(response, 'citations') else []
        return f"{content}\n\nSources:\n" + "\n".join([f"- {c}" for c in citations])
    
    return content


def search(
    query: str,
    api_key: Optional[str] = None,
    model: str = "llama-3.1-sonar-large-128k-online",
) -> Dict[str, Any]:
    """
    Search with Perplexity.
    
    Returns:
        Dict with 'answer', 'citations', and 'usage'
    """
    client = create_client(api_key)
    
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
    parser.add_argument("--message", default="What are the latest AI developments?", help="Message")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model")
    parser.add_argument("--citations", action="store_true", help="Include citations")
    args = parser.parse_args()
    
    if args.api_key:
        print(f"Using model: {args.model}")
        response = chat(args.message, api_key=args.api_key, model=args.model, return_citations=args.citations)
        print(f"Response: {response}")
    else:
        print("Please provide --api-key")
