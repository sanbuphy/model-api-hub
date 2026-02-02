"""
DeepSeek LLM API wrapper.

Provides interface for DeepSeek's language models including:
- deepseek-chat - General-purpose chat model
- deepseek-reasoner - Reasoning-specialized model
"""

from openai import OpenAI
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

# Default configuration
DEFAULT_BASE_URL: str = "https://api.deepseek.com"
DEFAULT_MODEL: str = "deepseek-chat"


def create_client(api_key: Optional[str] = None, base_url: str = DEFAULT_BASE_URL) -> OpenAI:
    """
    Create OpenAI-compatible client for DeepSeek API.

    Args:
        api_key: DeepSeek API key. If None, loads from environment.
        base_url: Base URL for DeepSeek API.

    Returns:
        Configured OpenAI client instance.
    """
    if api_key is None:
        api_key = get_api_key("deepseek")
    return OpenAI(api_key=api_key, base_url=base_url)


def get_completion(
    client: OpenAI,
    messages: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.7,
    stream: bool = False,
    **kwargs
) -> str:
    """
    Request a completion from DeepSeek LLM.

    Args:
        client: OpenAI client instance.
        messages: List of message dictionaries with 'role' and 'content'.
        model: Model name to use.
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature (0-2).
        top_p: Nucleus sampling threshold (0-1).
        stream: Whether to stream the response.
        **kwargs: Additional parameters to pass to the API.

    Returns:
        The content of the model's response.
    """
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
    """
    Quick chat function for single prompt completion.

    Args:
        prompt: User prompt text.
        system_prompt: Optional system prompt.
        api_key: Optional API key. If None, loads from environment.
        model: Model name to use.
        **kwargs: Additional parameters for get_completion.

    Returns:
        Model response text.
    """
    client = create_client(api_key=api_key)

    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    return get_completion(client, messages, model=model, **kwargs)


def chat_stream(
    prompt: str,
    system_prompt: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.7,
    **kwargs
):
    """
    Streaming chat function that yields response chunks as they arrive.

    Args:
        prompt: User prompt text.
        system_prompt: Optional system prompt.
        api_key: Optional API key. If None, loads from environment.
        model: Model name to use.
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature (0-2).
        top_p: Nucleus sampling threshold (0-1).
        **kwargs: Additional parameters.

    Yields:
        str: Response text chunks.

    Example:
        >>> for chunk in chat_stream("Tell me a story"):
        ...     print(chunk, end="", flush=True)
    """
    client = create_client(api_key=api_key)

    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
        **kwargs
    )

    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def main() -> None:
    """
    Main function demonstrating usage with config file support.
    """
    # Option 1: Direct API key
    api_key = None  # Set your API key directly or load from .env

    # Option 2: Use config manager
    config_mgr = ConfigManager()
    llm_config = config_mgr.get_llm_config("deepseek")
    model = llm_config.get("model", DEFAULT_MODEL)
    max_tokens = llm_config.get("max_tokens", 4096)
    temperature = llm_config.get("temperature", 0.7)

    # Create client and make request
    client = create_client(api_key=api_key)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Hello! Can you explain what you're capable of?"}
    ]

    print(f"Sending request to DeepSeek model: {model}")
    print("-" * 50)

    response = get_completion(
        client,
        messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature
    )

    print("Response:")
    print(response)


if __name__ == "__main__":
    # You can also pass API key directly when running:
    # python deepseek_llm.py --api-key your_key_here
    import argparse
    parser = argparse.ArgumentParser(description="DeepSeek LLM API Demo")
    parser.add_argument("--api-key", type=str, help="DeepSeek API key")
    args = parser.parse_args()

    if args.api_key:
        main()  # Will use the passed API key
    else:
        main()  # Will load from environment
