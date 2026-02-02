"""
OpenAI DALL-E Image Generation API wrapper.

Provides interface for OpenAI's image generation models:
- dall-e-3 - DALL-E 3 (high quality)
- dall-e-2 - DALL-E 2 (faster, lower cost)
"""

from openai import OpenAI
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

DEFAULT_MODEL: str = "dall-e-3"
DEFAULT_SIZE: str = "1024x1024"

API_KEY: Optional[str] = None


def create_client(api_key: Optional[str] = None) -> OpenAI:
    """Create OpenAI client."""
    if api_key is None:
        api_key = get_api_key("openai")
    return OpenAI(api_key=api_key)


def text_to_image(
    prompt: str,
    output_path: str = "output.png",
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    size: str = DEFAULT_SIZE,
    quality: str = "standard",
    n: int = 1,
    **kwargs
) -> bool:
    """Generate image from text using DALL-E."""
    try:
        client = create_client(api_key=api_key)
        
        response = client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            n=n,
            **kwargs
        )
        
        # Download image
        import requests
        image_url = response.data[0].url
        image_response = requests.get(image_url)
        
        with open(output_path, "wb") as f:
            f.write(image_response.content)
        
        return True
    except Exception as e:
        print(f"Error generating image: {e}")
        return False


def main():
    """Demo usage."""
    import argparse
    parser = argparse.ArgumentParser(description="OpenAI DALL-E Image Generation Demo")
    parser.add_argument("--prompt", required=True, help="Image generation prompt")
    parser.add_argument("--output", default="output.png", help="Output file path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to use (dall-e-2 or dall-e-3)")
    parser.add_argument("--size", default=DEFAULT_SIZE, help="Image size (e.g., 1024x1024)")
    parser.add_argument("--api-key", help="OpenAI API key")
    args = parser.parse_args()
    
    if args.api_key:
        global API_KEY
        API_KEY = args.api_key
    
    success = text_to_image(
        args.prompt,
        args.output,
        api_key=args.api_key,
        model=args.model,
        size=args.size
    )
    
    if success:
        print(f"Image saved to: {args.output}")
    else:
        print("Image generation failed")


if __name__ == "__main__":
    main()
