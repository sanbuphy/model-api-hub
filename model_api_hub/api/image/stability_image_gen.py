"""
Stability AI Image Generation API wrapper.

Provides interface for Stability AI's image generation models:
- stable-diffusion-xl-1024-v1-0 - SDXL
- stable-diffusion-v1-6 - SD 1.6
- stable-diffusion-3-medium - SD3 Medium
"""

import requests
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

DEFAULT_ENGINE: str = "stable-diffusion-xl-1024-v1-0"
API_BASE_URL: str = "https://api.stability.ai/v2beta/stable-image/generate/sd3"

API_KEY: Optional[str] = None


def text_to_image(
    prompt: str,
    output_path: str = "output.png",
    api_key: Optional[str] = None,
    engine: str = DEFAULT_ENGINE,
    width: int = 1024,
    height: int = 1024,
    **kwargs
) -> bool:
    """Generate image from text using Stability AI."""
    try:
        if api_key is None:
            api_key = get_api_key("stability_ai")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "image/*"
        }
        
        files = {
            "none": ("", ""),
        }
        
        data = {
            "prompt": prompt,
            "width": width,
            "height": height,
            **kwargs
        }
        
        response = requests.post(
            API_BASE_URL,
            headers=headers,
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"Error generating image: {e}")
        return False


def main():
    """Demo usage."""
    import argparse
    parser = argparse.ArgumentParser(description="Stability AI Image Generation Demo")
    parser.add_argument("--prompt", required=True, help="Image generation prompt")
    parser.add_argument("--output", default="output.png", help="Output file path")
    parser.add_argument("--engine", default=DEFAULT_ENGINE, help="Engine to use")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--api-key", help="Stability AI API key")
    args = parser.parse_args()
    
    if args.api_key:
        global API_KEY
        API_KEY = args.api_key
    
    success = text_to_image(
        args.prompt,
        args.output,
        api_key=args.api_key,
        engine=args.engine,
        width=args.width,
        height=args.height
    )
    
    if success:
        print(f"Image saved to: {args.output}")
    else:
        print("Image generation failed")


if __name__ == "__main__":
    main()
