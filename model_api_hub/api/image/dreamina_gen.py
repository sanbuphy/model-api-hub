"""
Dreamina (即梦) Image Generation API - 字节跳动 AI 绘画平台

Website: https://jimeng.jianying.com/
Features: AI 绘画、智能画布、风格迁移
Pricing: 免费额度 + 付费订阅
"""

import os
from typing import Optional, List, Dict, Any
import requests
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key

# API Configuration
DEFAULT_IMAGE_MODEL: str = "jimeng-2.0"
BASE_URL: str = "https://jimeng.jianying.com/api/v1"

# Available models
AVAILABLE_IMAGE_MODELS: List[str] = [
    "jimeng-2.0",
    "jimeng-1.5",
    "jimeng-1.0",
]


def create_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """Create request headers."""
    key = api_key or get_api_key("dreamina")
    if not key:
        raise ValueError("Dreamina API key is required. Set DREAMINA_API_KEY environment variable.")
    
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def text_to_image(
    prompt: str,
    output_path: str = "output.png",
    api_key: Optional[str] = None,
    model: str = DEFAULT_IMAGE_MODEL,
    width: int = 1024,
    height: int = 1024,
    **kwargs
) -> bool:
    """
    Generate image from text using Dreamina.
    
    Args:
        prompt: Image generation prompt
        output_path: Output file path
        api_key: API key (optional)
        model: Model identifier
        width: Image width
        height: Image height
        **kwargs: Additional parameters
    
    Returns:
        True if successful, False otherwise
    """
    try:
        headers = create_headers(api_key)
        
        payload = {
            "model": model,
            "prompt": prompt,
            "width": width,
            "height": height,
            **kwargs
        }
        
        response = requests.post(
            f"{BASE_URL}/images/generations",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        
        if "data" in result and len(result["data"]) > 0:
            image_url = result["data"][0]["url"]
            
            # Download image
            img_response = requests.get(image_url, timeout=30)
            img_response.raise_for_status()
            
            with open(output_path, "wb") as f:
                f.write(img_response.content)
            return True
        else:
            print(f"Error: No image data in response - {result}")
            return False
            
    except Exception as e:
        print(f"Error generating image: {e}")
        return False


def main():
    """Demo usage."""
    import argparse
    parser = argparse.ArgumentParser(description="Dreamina Image Generation Demo")
    parser.add_argument("--prompt", required=True, help="Image generation prompt")
    parser.add_argument("--output", default="dreamina_output.png", help="Output file path")
    parser.add_argument("--api-key", help="Dreamina API key")
    parser.add_argument("--model", default=DEFAULT_IMAGE_MODEL, help="Model to use")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    args = parser.parse_args()
    
    print(f"Generating image with prompt: {args.prompt}")
    success = text_to_image(
        prompt=args.prompt,
        output_path=args.output,
        api_key=args.api_key,
        model=args.model,
        width=args.width,
        height=args.height
    )
    
    if success:
        print(f"Image saved to: {args.output}")
    else:
        print("Failed to generate image")


if __name__ == "__main__":
    main()
