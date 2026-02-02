"""
OpenAI LLM API wrapper.

Provides interface for OpenAI's language models including:
- gpt-4o - Latest GPT-4 Omni model
- gpt-4o-mini - Lightweight GPT-4 Omni
- gpt-4-turbo - GPT-4 Turbo
- gpt-3.5-turbo - GPT-3.5 Turbo
"""

from openai import OpenAI
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

DEFAULT_BASE_URL: str = "https://api.openai.com/v1"
DEFAULT_MODEL: str = "gpt-4o"

API_KEY: Optional[str] = None


def create_client(api_key: Optional[str] = None, base_url: str = DEFAULT_BASE_URL) -> OpenAI:
    """Create OpenAI client."""
    if api_key is None:
        api_key = get_api_key("openai")
    return OpenAI(api_key=api_key, base_url=base_url)


def get_completion(
    client: OpenAI,
    messages: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 1.0,
    stream: bool = False,
    **kwargs
) -> str:
    """Request completion from OpenAI LLM."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=stream,
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
    llm_config = config_mgr.get_llm_config("openai")
    model = llm_config.get("model", DEFAULT_MODEL)
    
    client = create_client()
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Hello! What can you do?"}
    ]
    
    print(f"Sending request to OpenAI model: {model}")
    response = get_completion(client, messages, model=model)
    print("Response:")
    print(response)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OpenAI LLM API Demo")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    args = parser.parse_args()
    
    if args.api_key:
        API_KEY = args.api_key
    main()
