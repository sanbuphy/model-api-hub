"""
StepFun LLM API wrapper.

Provides interface for StepFun's language models including:
- step-3.5-flash - General-purpose chat model
- step-2-16k - Large context chat model
"""

from openai import OpenAI
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

DEFAULT_BASE_URL: str = "https://api.stepfun.com/v1"
DEFAULT_MODEL: str = "step-3.5-flash"


def create_client(api_key: Optional[str] = None, base_url: str = DEFAULT_BASE_URL) -> OpenAI:
    """
    Create OpenAI-compatible client for StepFun API.

    Args:
        api_key: StepFun API key. If None, loads from environment.
        base_url: Base URL for StepFun API.

    Returns:
        Configured OpenAI client instance.
    """
    if api_key is None:
        api_key = get_api_key("stepfun")
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
    """
    Request a completion from StepFun LLM.

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
    top_p: float = 1.0,
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

    try:
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    finally:
        client.close()


def main() -> None:
    """
    Main function demonstrating usage with config file support.
    """
    config_mgr = ConfigManager()
    llm_config = config_mgr.get_llm_config("stepfun")
    model = llm_config.get("model", DEFAULT_MODEL)
    max_tokens = llm_config.get("max_tokens", 4096)
    temperature = llm_config.get("temperature", 0.7)

    print(f"模型: {model}")
    print("=" * 50)

    print("\n【同步调用测试】")
    print("-" * 50)
    response = chat(
        prompt="你好！请用一句话介绍你自己？",
        system_prompt="你是一个有用的AI助手。",
        model=model,
        max_tokens=max_tokens,
        temperature=temperature
    )
    print("回复:")
    print(response)

    print("\n\n【流式调用测试】")
    print("-" * 50)
    print("回复:")
    for chunk in chat_stream(
        prompt="请详细介绍一下你的能力和特点？",
        system_prompt="你是一个有用的AI助手。",
        model=model,
        max_tokens=max_tokens,
        temperature=temperature
    ):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
