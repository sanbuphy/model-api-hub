"""
Alibaba DashScope (Tongyi Qianwen) LLM API wrapper.

Provides interface for Alibaba's Tongyi Qianwen models:
- qwen-max - Qwen Max
- qwen-plus - Qwen Plus
- qwen-turbo - Qwen Turbo
- qwen-long - Qwen Long (long context)
"""

import dashscope
from dashscope import Generation
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

DEFAULT_MODEL: str = "qwen-max"

API_KEY: Optional[str] = None


def create_client(api_key: Optional[str] = None):
    """Create DashScope client."""
    if api_key is None:
        api_key = get_api_key("dashscope")
    dashscope.api_key = api_key
    return dashscope


def get_completion(
    messages: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    **kwargs
) -> str:
    """Request completion from DashScope LLM."""
    if api_key is None:
        api_key = get_api_key("dashscope")
    
    response = Generation.call(
        model=model,
        messages=messages,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs
    )
    
    if response.status_code == 200:
        return response.output.text
    else:
        raise Exception(f"Error: {response.message}")


def chat(
    prompt: str,
    system_prompt: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    **kwargs
) -> str:
    """Quick chat function."""
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return get_completion(messages, model=model, api_key=api_key, **kwargs)


def main() -> None:
    """Demo usage."""
    config_mgr = ConfigManager()
    llm_config = config_mgr.get_llm_config("dashscope")
    model = llm_config.get("model", DEFAULT_MODEL)
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "你好！你能做什么？"}
    ]
    
    print(f"Sending request to DashScope model: {model}")
    response = get_completion(messages, model=model)
    print("Response:")
    print(response)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DashScope LLM API Demo")
    parser.add_argument("--api-key", type=str, help="DashScope API key")
    args = parser.parse_args()
    
    if args.api_key:
        API_KEY = args.api_key
    main()
