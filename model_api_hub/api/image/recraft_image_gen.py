"""
Recraft Image Generation API wrapper.

Provides interface for Recraft's image generation models.
"""

import requests
import os
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

# Default configuration
DEFAULT_BASE_URL: str = "https://api.recraft.ai/v1/images/generations"
DEFAULT_MODEL: str = "recraft-v3"
DEFAULT_SIZE: str = "1024x1024"


def generate_image(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    size: str = DEFAULT_SIZE,
    n: int = 1,
    style: Optional[str] = None,
    negative_prompt: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Generate an image using Recraft API.

    Args:
        prompt: Text description for image generation.
        api_key: Recraft API key. If None, loads from environment.
        model: Model identifier to use.
        size: Image dimensions (e.g., "1024x1024", "1536x640", "640x1536").
        n: Number of images to generate.
        style: Optional style preset (e.g., "realistic", "illustration", "3d-render").
        negative_prompt: Optional text describing what should NOT appear.

    Returns:
        API response dictionary with generated image URLs, or None if failed.
    """
    if api_key is None:
        api_key = get_api_key("recraft")

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "n": n
    }

    if style is not None:
        payload["style"] = style
    if negative_prompt is not None:
        payload["negative_prompt"] = negative_prompt

    headers: Dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(DEFAULT_BASE_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error generating image: {e}")
        return None


def save_image_from_url(image_url: str, output_path: str = "generated_image.png") -> bool:
    """
    Download and save an image from URL to local file.

    Args:
        image_url: URL of the image to download.
        output_path: Local path where to save the image.

    Returns:
        True if successful, False otherwise.
    """
    try:
        response = requests.get(image_url)
        response.raise_for_status()

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Image saved successfully to: {output_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return False
    except Exception as e:
        print(f"Error saving image: {e}")
        return False


def text_to_image(
    prompt: str,
    output_path: str = "generated_image.png",
    api_key: Optional[str] = None,
    **kwargs
) -> bool:
    """
    Generate an image from text prompt and save it.

    Args:
        prompt: Text description for image generation.
        output_path: Local path to save the generated image.
        api_key: Optional API key. If None, loads from environment.
        **kwargs: Additional parameters for generate_image.

    Returns:
        True if successful, False otherwise.
    """
    result = generate_image(prompt, api_key=api_key, **kwargs)

    if result and "data" in result:
        images = result["data"]
        if images and len(images) > 0:
            image_url = images[0]["url"]
            return save_image_from_url(image_url, output_path)
    return False


def main() -> None:
    """
    Main function demonstrating usage with config file support.
    """
    # Option 1: Direct API key
    api_key = None  # Set your API key directly or load from .env

    # Option 2: Use config manager
    config_mgr = ConfigManager()
    img_config = config_mgr.get_image_config("recraft")
    model = img_config.get("model", DEFAULT_MODEL)
    size = img_config.get("size", DEFAULT_SIZE)

    # Test text to image
    prompt = "a futuristic cityscape at sunset, cyberpunk style, neon lights, highly detailed"
    output_path = "generated_image.png"

    print(f"Generating image with prompt: {prompt}")
    print(f"Model: {model}")
    print(f"Size: {size}")
    print("-" * 50)

    success = text_to_image(
        prompt=prompt,
        output_path=output_path,
        api_key=api_key,
        model=model,
        size=size
    )

    if success:
        print(f"Image generation completed successfully!")
    else:
        print("Image generation failed")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Recraft Image Generation API Demo")
    parser.add_argument("--api-key", type=str, help="Recraft API key")
    parser.add_argument("--prompt", type=str,
                        default="a beautiful landscape",
                        help="Text prompt for image generation")
    parser.add_argument("--output", type=str, default="generated_image.png",
                        help="Output file path")
    args = parser.parse_args()

    if args.prompt:
        text_to_image(args.prompt, args.output, api_key=args.api_key)
    else:
        main()
