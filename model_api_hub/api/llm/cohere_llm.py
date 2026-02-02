"""
Cohere LLM API wrapper.

Provides interface for Cohere's language models:
- command-r-plus - Command R+
- command-r - Command R
- command - Command
- command-light - Command Light
"""

import cohere
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

DEFAULT_MODEL: str = "command-r-plus"

API_KEY: Optional[str] = None


def create_client(api_key: Optional[str] = None) -> cohere.Client:
    """Create Cohere client."""
    if api_key is None:
        api_key = get_api_key("cohere")
    return cohere.Client(api_key)


def get_completion(
    client: cohere.Client,
    messages: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    **kwargs
) -> str:
    """Request completion from Cohere LLM."""
    # Convert messages to Cohere format
    chat_history = []
    for msg in messages[:-1]:
        role = "USER" if msg["role"] == "user" else "CHATBOT"
        chat_history.append({"role": role, "message": msg["content"]})
    
    message = messages[-1]["content"] if messages else ""
    
    response = client.chat(
        model=model,
        message=message,
        chat_history=chat_history,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs
    )
    return response.text


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
    llm_config = config_mgr.get_llm_config("cohere")
    model = llm_config.get("model", DEFAULT_MODEL)
    
    client = create_client()
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Hello! What can you do?"}
    ]
    
    print(f"Sending request to Cohere model: {model}")
    response = get_completion(client, messages, model=model)
    print("Response:")
    print(response)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cohere LLM API Demo")
    parser.add_argument("--api-key", type=str, help="Cohere API key")
    args = parser.parse_args()
    
    if args.api_key:
        API_KEY = args.api_key
    main()
