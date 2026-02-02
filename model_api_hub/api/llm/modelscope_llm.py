"""
ModelScope (魔搭社区) LLM API wrapper - 阿里开源模型平台

Website: https://www.modelscope.cn/
API Docs: https://www.modelscope.cn/docs/model-service/API-Inference/intro
API Key: https://www.modelscope.cn/my/mynotebook/authorization
Pricing: 每日免费 2000 次调用（单个模型每日上限 500 次）

Available Models:
- Qwen series: qwen-max, qwen-plus, qwen-turbo, qwen2.5-72b, qwen2.5-32b, qwen2.5-14b, qwen2.5-7b
- Llama series: llama3.1-70b, llama3.1-8b, llama3-70b, llama3-8b
- DeepSeek: deepseek-v3, deepseek-r1
- GLM: glm-4, glm-4-plus
- Other: yi-large, baichuan2-13b, chatglm3-6b

Quick Start:
1. 访问 https://www.modelscope.cn/my/mynotebook/authorization
2. 获取 SDK Token 作为 API Key
3. 参考文档: https://www.modelscope.cn/docs/model-service/API-Inference/intro
"""

from openai import OpenAI
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

# Default configuration
DEFAULT_BASE_URL: str = "https://api-inference.modelscope.cn/v1"
DEFAULT_MODEL: str = "qwen-max"


def create_client(api_key: Optional[str] = None, base_url: str = DEFAULT_BASE_URL) -> OpenAI:
    """
    Create OpenAI-compatible client for ModelScope API.

    Args:
        api_key: ModelScope API key. If None, loads from environment.
        base_url: Base URL for ModelScope API.

    Returns:
        Configured OpenAI client instance.
    """
    if api_key is None:
        api_key = get_api_key("modelscope")
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
    Request a completion from ModelScope LLM.

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


def main() -> None:
    """
    Main function demonstrating usage with config file support.
    """
    # Option 1: Direct API key
    api_key = None  # Set your API key directly or load from .env

    # Option 2: Use config manager
    config_mgr = ConfigManager()
    llm_config = config_mgr.get_llm_config("modelscope")
    model = llm_config.get("model", DEFAULT_MODEL)
    max_tokens = llm_config.get("max_tokens", 4096)
    temperature = llm_config.get("temperature", 0.7)

    # Create client and make request
    client = create_client(api_key=api_key)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Hello! Can you explain what you're capable of?"}
    ]

    print(f"Sending request to ModelScope model: {model}")
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
    # python modelscope_llm.py --api-key your_key_here
    import argparse
    parser = argparse.ArgumentParser(description="ModelScope LLM API Demo")
    parser.add_argument("--api-key", type=str, help="ModelScope API key")
    args = parser.parse_args()

    if args.api_key:
        main()  # Will use the passed API key
    else:
        main()  # Will load from environment
