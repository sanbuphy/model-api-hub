"""
Poe API client - 聚合多个AI模型

Website: https://poe.com/
Features: 访问Claude、GPT、Gemini等多个模型
Note: 需要Poe订阅
"""

import os
from typing import Optional, List, Dict, Any
import requests

# API Configuration
API_KEY: str = os.getenv("POE_API_KEY", "")
BASE_URL: str = "https://api.poe.com/bot"
DEFAULT_BOT: str = "Claude-3.5-Sonnet"

# Available bots
AVAILABLE_BOTS: List[str] = [
    "Claude-3.5-Sonnet",
    "Claude-3.5-Sonnet-200k",
    "Claude-3-Opus",
    "Claude-3-Sonnet",
    "Claude-3-Haiku",
    "GPT-4o",
    "GPT-4o-Mini",
    "GPT-4-Turbo",
    "GPT-4",
    "GPT-3.5-Turbo",
    "Gemini-1.5-Pro",
    "Gemini-1.5-Flash",
    "Llama-3.1-405B-T",
    "Llama-3.1-70B-T",
    "Llama-3.1-8B-T",
    "DALL-E-3",
    "StableDiffusionXL",
]


def create_client(api_key: Optional[str] = None) -> requests.Session:
    """Create Poe client session."""
    key = api_key or API_KEY
    if not key:
        raise ValueError("Poe API key is required. Set POE_API_KEY environment variable.")
    
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    })
    return session


def chat(
    message: str,
    bot: str = DEFAULT_BOT,
    api_key: Optional[str] = None,
    **kwargs
) -> str:
    """
    Chat using Poe.
    
    Args:
        message: User message
        bot: Bot name (e.g., 'Claude-3.5-Sonnet')
        api_key: API key (optional)
        **kwargs: Additional parameters
    
    Returns:
        Response text
    """
    session = create_client(api_key)
    
    url = f"{BASE_URL}/{bot}"
    
    payload = {
        "query": message,
    }
    payload.update(kwargs)
    
    response = session.post(url, json=payload)
    response.raise_for_status()
    
    return response.json().get("text", "")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Poe")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--message", default="Hello! Who are you?", help="Message")
    parser.add_argument("--bot", default=DEFAULT_BOT, help="Bot name")
    args = parser.parse_args()
    
    if args.api_key:
        print(f"Using bot: {args.bot}")
        response = chat(args.message, api_key=args.api_key, bot=args.bot)
        print(f"Response: {response}")
    else:
        print("Please provide --api-key")
