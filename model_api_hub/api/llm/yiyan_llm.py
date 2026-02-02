"""
Yiyan (Baidu Ernie) LLM API wrapper.

Provides interface for Baidu's Ernie Bot models including:
- ernie-4.0-8k - Ernie 4.0 with 8K context
- ernie-3.5-8k - Ernie 3.5 with 8K context
- ernie-speed-8k - Fast inference model
"""

import requests
import json
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

# Default configuration
DEFAULT_BASE_URL: str = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat"
DEFAULT_MODEL: str = "ernie-4.0-8k"


def get_access_token(api_key: str) -> str:
    """
    Get access token from API key (format: {client_id}_{client_secret}).

    Args:
        api_key: Baidu API key in format "client_id_client_secret".

    Returns:
        Access token string.
    """
    try:
        client_id, client_secret = api_key.split("_")
    except ValueError:
        raise ValueError("Invalid API key format. Expected: client_id_client_secret")

    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret
    }

    response = requests.post(url, params=params)
    response.raise_for_status()
    return response.json()["access_token"]


def create_client(api_key: Optional[str] = None) -> str:
    """
    Get access token for Yiyan API.

    Args:
        api_key: Yiyan API key. If None, loads from environment.

    Returns:
        Access token string.
    """
    if api_key is None:
        api_key = get_api_key("yiyan")
    return get_access_token(api_key)


def get_completion(
    access_token: str,
    messages: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.8,
    penalty_score: float = 1.0,
    stream: bool = False,
    **kwargs
) -> str:
    """
    Request a completion from Yiyan LLM.

    Args:
        access_token: Baidu API access token.
        messages: List of message dictionaries with 'role' and 'content'.
        model: Model name to use (endpoint name).
        temperature: Sampling temperature (0-1).
        top_p: Nucleus sampling threshold (0-1).
        penalty_score: Penalty for repetition (1-2).
        stream: Whether to stream the response.
        **kwargs: Additional parameters to pass to the API.

    Returns:
        The content of the model's response.
    """
    url = f"{DEFAULT_BASE_URL}/{model}"

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "penalty_score": penalty_score,
        "stream": stream,
        **kwargs
    }

    params = {"access_token": access_token}

    response = requests.post(url, headers=headers, params=params, json=payload)
    response.raise_for_status()
    result = response.json()

    return result["result"]


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
    access_token = create_client(api_key=api_key)

    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "user", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    return get_completion(access_token, messages, model=model, **kwargs)


def main() -> None:
    """
    Main function demonstrating usage with config file support.
    """
    # Option 1: Direct API key
    api_key = None  # Set your API key directly or load from .env

    # Option 2: Use config manager
    config_mgr = ConfigManager()
    llm_config = config_mgr.get_llm_config("yiyan")
    model = llm_config.get("model", DEFAULT_MODEL)
    temperature = llm_config.get("temperature", 0.7)

    # Create client and make request
    access_token = create_client(api_key=api_key)

    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": "你好！请介绍一下你自己。"}
    ]

    print(f"Sending request to Yiyan model: {model}")
    print("-" * 50)

    response = get_completion(
        access_token,
        messages,
        model=model,
        temperature=temperature
    )

    print("Response:")
    print(response)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Yiyan LLM API Demo")
    parser.add_argument("--api-key", type=str, help="Yiyan API key (format: client_id_client_secret)")
    args = parser.parse_args()

    if args.api_key:
        access_token = create_client(api_key=args.api_key)
        messages = [{"role": "user", "content": "你好！请介绍一下你自己。"}]
        response = get_completion(access_token, messages)
        print("Response:")
        print(response)
    else:
        main()
