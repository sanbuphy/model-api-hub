"""
SiliconFlow LLM API wrapper.

SiliconFlow provides access to multiple AI models including:
- deepseek-ai/DeepSeek-V3 - MoE architecture with 671B parameters
- deepseek-ai/DeepSeek-R1 - Reasoning model with 671B parameters
- zai-org/GLM-4.5 - MoE architecture with 335B parameters
- Qwen/Qwen3-Coder-480B-A35B-Instruct - Code model
- moonshotai/Kimi-K2-Instruct - MoE architecture with 1T parameters

Website: https://cloud.siliconflow.cn/
Models: https://cloud.siliconflow.cn/me/models
"""

from openai import OpenAI
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

# Default configuration
DEFAULT_BASE_URL: str = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL: str = "deepseek-ai/DeepSeek-V3"


def create_client(api_key: Optional[str] = None, base_url: str = DEFAULT_BASE_URL) -> OpenAI:
    """
    Create OpenAI-compatible client for SiliconFlow API.

    Args:
        api_key: SiliconFlow API key. If None, loads from environment.
        base_url: Base URL for SiliconFlow API.

    Returns:
        Configured OpenAI client instance.
    """
    if api_key is None:
        api_key = get_api_key("siliconflow")
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
    Request a completion from SiliconFlow LLM.

    Args:
        client: OpenAI client instance.
        messages: List of message dictionaries with 'role' and 'content'.
        model: Model identifier.
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
        model: Model identifier to use.
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
    llm_config = config_mgr.get_llm_config("siliconflow")
    model = llm_config.get("model", DEFAULT_MODEL)
    max_tokens = llm_config.get("max_tokens", 4096)
    temperature = llm_config.get("temperature", 0.7)

    # Create client and make request
    client = create_client(api_key=api_key)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Hello! Can you explain what SiliconFlow is?"}
    ]

    print(f"Sending request via SiliconFlow to model: {model}")
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
    parser = argparse.ArgumentParser(description="SiliconFlow LLM API Demo")
    parser.add_argument("--api-key", type=str, help="SiliconFlow API key")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Model to use")
    args = parser.parse_args()

    if args.api_key:
        client = create_client(api_key=args.api_key)
    else:
        main()
