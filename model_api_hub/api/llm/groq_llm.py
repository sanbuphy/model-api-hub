"""
Groq LLM API wrapper.

Provides interface for Groq's high-speed inference:
- llama-3.1-405b-reasoning - Llama 3.1 405B
- llama-3.1-70b-versatile - Llama 3.1 70B
- llama-3.1-8b-instant - Llama 3.1 8B
- mixtral-8x7b-32768 - Mixtral 8x7B
- gemma-7b-it - Gemma 7B
"""

from openai import OpenAI
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

DEFAULT_BASE_URL: str = "https://api.groq.com/openai/v1"
DEFAULT_MODEL: str = "llama-3.1-70b-versatile"

API_KEY: Optional[str] = None


def create_client(api_key: Optional[str] = None, base_url: str = DEFAULT_BASE_URL) -> OpenAI:
    """Create Groq client (OpenAI-compatible)."""
    if api_key is None:
        api_key = get_api_key("groq")
    return OpenAI(api_key=api_key, base_url=base_url)


def get_completion(
    client: OpenAI,
    messages: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    **kwargs
) -> str:
    """Request completion from Groq LLM."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs
    )
    return response.choices[0].message.content


def chat(
    prompt: str,
    system_prompt: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    **kwargs
) -> str:
    """Quick chat function."""
    client = create_client(api_key=api_key)
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return get_completion(client, messages, model=model, **kwargs)


def main() -> None:
    """Demo usage."""
    config_mgr = ConfigManager()
    llm_config = config_mgr.get_llm_config("groq")
    model = llm_config.get("model", DEFAULT_MODEL)
    
    client = create_client()
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Hello! What can you do?"}
    ]
    
    print(f"Sending request to Groq model: {model}")
    response = get_completion(client, messages, model=model)
    print("Response:")
    print(response)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Groq LLM API Demo")
    parser.add_argument("--api-key", type=str, help="Groq API key")
    args = parser.parse_args()
    
    if args.api_key:
        API_KEY = args.api_key
    main()
