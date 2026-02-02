"""
Alibaba DashScope (Tongyi Wanxiang) Image Generation API wrapper.

Provides interface for Tongyi Wanxiang image generation:
- wanx-v1 - Tongyi Wanxiang V1
"""

import dashscope
from dashscope import ImageSynthesis
from typing import Optional
import requests
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

DEFAULT_MODEL: str = "wanx-v1"

API_KEY: Optional[str] = None


def text_to_image(
    prompt: str,
    output_path: str = "output.png",
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    size: str = "1024x1024",
    **kwargs
) -> bool:
    """Generate image from text using Tongyi Wanxiang."""
    try:
        if api_key is None:
            api_key = get_api_key("dashscope")
        
        response = ImageSynthesis.call(
            model=model,
            prompt=prompt,
            size=size,
            api_key=api_key,
            **kwargs
        )
        
        if response.status_code == 200:
            # Download image from URL
            image_url = response.output.results[0].url
            img_response = requests.get(image_url)
            
            with open(output_path, "wb") as f:
                f.write(img_response.content)
            return True
        else:
            print(f"Error: {response.message}")
            return False
            
    except Exception as e:
        print(f"Error generating image: {e}")
        return False


def main():
    """Demo usage."""
    import argparse
    parser = argparse.ArgumentParser(description="DashScope Tongyi Wanxiang Image Generation Demo")
    parser.add_argument("--prompt", required=True, help="Image generation prompt")
    parser.add_argument("--output", default="output.png", help="Output file path")
    parser.add_argument("--size", default="1024x1024", help="Image size")
    parser.add_argument("--api-key", help="DashScope API key")
    args = parser.parse_args()
    
    if args.api_key:
        global API_KEY
        API_KEY = args.api_key
    
    success = text_to_image(
        args.prompt,
        args.output,
        api_key=args.api_key,
        size=args.size
    )
    
    if success:
        print(f"Image saved to: {args.output}")
    else:
        print("Image generation failed")


if __name__ == "__main__":
    main()
