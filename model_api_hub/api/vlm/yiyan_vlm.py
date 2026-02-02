"""
Yiyan (Baidu Ernie Bot) VLM API wrapper.

Provides interface for Baidu's vision capabilities in Ernie Bot models.
"""

import requests
import json
from typing import Dict, Any, List, Optional
import base64
import os
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

# Default configuration
DEFAULT_BASE_URL: str = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat"


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


def encode_image(image_path: str) -> str:
    """
    Encode an image file to base64 string.

    Args:
        image_path: Path to the image file.

    Returns:
        Base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_completion(
    access_token: str,
    messages: List[Dict[str, Any]],
    model: str = "ernie-vision-4.0",
    temperature: float = 0.7,
    top_p: float = 0.8,
    stream: bool = False,
    **kwargs
) -> str:
    """
    Request a completion from Yiyan VLM.

    Args:
        access_token: Baidu API access token.
        messages: List of message dictionaries with 'role' and 'content'.
        model: Model name to use.
        temperature: Sampling temperature (0-1).
        top_p: Nucleus sampling threshold (0-1).
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
        "stream": stream,
        **kwargs
    }

    params = {"access_token": access_token}

    response = requests.post(url, headers=headers, params=params, json=payload)
    response.raise_for_status()
    result = response.json()

    return result["result"]


def analyze_image(
    image_path: str,
    prompt: str = "请详细描述这张图片。",
    api_key: Optional[str] = None,
    model: str = "ernie-vision-4.0",
    **kwargs
) -> str:
    """
    Analyze a single image with text prompt.

    Args:
        image_path: Path to the image file.
        prompt: Text prompt describing what to do with the image.
        api_key: Optional API key. If None, loads from environment.
        model: Model name to use.
        **kwargs: Additional parameters for get_completion.

    Returns:
        Model response text.
    """
    base64_image = encode_image(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": f"{base64_image}"
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    access_token = create_client(api_key=api_key)
    return get_completion(access_token, messages, model=model, **kwargs)


def main() -> None:
    """
    Main function demonstrating usage with config file support.
    """
    # Option 1: Direct API key
    api_key = None  # Set your API key directly or load from .env

    # Test with single image
    image_path = "test_image.jpg"

    if os.path.exists(image_path):
        print(f"Analyzing image: {image_path}")
        print("-" * 50)

        response = analyze_image(
            image_path,
            prompt="请详细描述这张图片。",
            api_key=api_key
        )

        print("Response:")
        print(response)
    else:
        print(f"Image file not found: {image_path}")
        print("Please provide a valid image path to test the VLM API.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Yiyan VLM API Demo")
    parser.add_argument("--api-key", type=str, help="Yiyan API key")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--prompt", type=str, default="请详细描述这张图片。",
                        help="Text prompt for the image")
    args = parser.parse_args()

    if args.image:
        response = analyze_image(args.image, args.prompt, api_key=args.api_key)
        print("Response:")
        print(response)
    else:
        main()
