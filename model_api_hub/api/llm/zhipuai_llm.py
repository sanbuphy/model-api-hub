"""
ZhipuAI (GLM) LLM API wrapper.

Provides interface for ZhipuAI's language models including:
- glm-4-plus - Latest general-purpose model
- glm-4-0520 - Previous version
- glm-4-air - Lightweight version
- glm-4-flash - Fast inference version
"""

from openai import OpenAI
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

# Default configuration
DEFAULT_BASE_URL: str = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_MODEL: str = "glm-4-plus"


def create_client(api_key: Optional[str] = None, base_url: str = DEFAULT_BASE_URL) -> OpenAI:
    """
    Create OpenAI-compatible client for ZhipuAI API.

    Args:
        api_key: ZhipuAI API key. If None, loads from environment.
        base_url: Base URL for ZhipuAI API.

    Returns:
        Configured OpenAI client instance.
    """
    if api_key is None:
        api_key = get_api_key("zhipuai")
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
    Request a completion from ZhipuAI LLM.

    Args:
        client: OpenAI client instance.
        messages: List of message dictionaries with 'role' and 'content'.
        model: Model name to use.
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature (0-1).
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


def main() -> None:
    """
    Main function demonstrating usage with config file support.
    """
    # Option 1: Direct API key
    api_key = None  # Set your API key directly or load from .env

    # Option 2: Use config manager
    config_mgr = ConfigManager()
    llm_config = config_mgr.get_llm_config("zhipuai")
    model = llm_config.get("model", DEFAULT_MODEL)
    max_tokens = llm_config.get("max_tokens", 4096)
    temperature = llm_config.get("temperature", 0.7)

    # Create client and make request
    client = create_client(api_key=api_key)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": "你好！请介绍一下你自己的主要能力。"}
    ]

    print(f"Sending request to ZhipuAI model: {model}")
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
    import argparse
    parser = argparse.ArgumentParser(description="ZhipuAI LLM API Demo")
    parser.add_argument("--api-key", type=str, help="ZhipuAI API key")
    args = parser.parse_args()

    if args.api_key:
        # Override with command-line provided key
        client = create_client(api_key=args.api_key)
    else:
        main()
