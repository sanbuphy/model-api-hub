"""
Mistral AI LLM API wrapper.

Provides interface for Mistral AI's language models:
- mistral-large-latest - Mistral Large
- mistral-medium-latest - Mistral Medium
- mistral-small-latest - Mistral Small
- codestral-latest - Codestral (code generation)
"""

from mistralai import Mistral
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

DEFAULT_MODEL: str = "mistral-large-latest"

API_KEY: Optional[str] = None


def create_client(api_key: Optional[str] = None) -> Mistral:
    """Create Mistral client."""
    if api_key is None:
        api_key = get_api_key("mistral")
    return Mistral(api_key=api_key)


def get_completion(
    client: Mistral,
    messages: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    **kwargs
) -> str:
    """Request completion from Mistral LLM."""
    response = client.chat.complete(
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
    llm_config = config_mgr.get_llm_config("mistral")
    model = llm_config.get("model", DEFAULT_MODEL)
    
    client = create_client()
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Hello! What can you do?"}
    ]
    
    print(f"Sending request to Mistral model: {model}")
    response = get_completion(client, messages, model=model)
    print("Response:")
    print(response)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Mistral LLM API Demo")
    parser.add_argument("--api-key", type=str, help="Mistral API key")
    args = parser.parse_args()
    
    if args.api_key:
        API_KEY = args.api_key
    main()
