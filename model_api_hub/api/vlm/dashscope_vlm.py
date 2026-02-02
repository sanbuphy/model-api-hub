"""
Alibaba DashScope (Qwen-VL) Vision API wrapper.

Provides interface for Qwen's vision-language models:
- qwen-vl-plus - Qwen VL Plus
- qwen-vl-max - Qwen VL Max
"""

import dashscope
from dashscope import MultiModalConversation
from typing import Optional
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

DEFAULT_MODEL: str = "qwen-vl-plus"

API_KEY: Optional[str] = None


def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_image(
    image_path: str,
    prompt: str = "这是什么?",
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    **kwargs
) -> str:
    """Analyze image using Qwen-VL."""
    if api_key is None:
        api_key = get_api_key("dashscope")
    
    # Encode image
    base64_image = encode_image(image_path)
    
    messages = [{
        "role": "user",
        "content": [
            {"image": f"data:image/jpeg;base64,{base64_image}"},
            {"text": prompt}
        ]
    }]
    
    response = MultiModalConversation.call(
        model=model,
        messages=messages,
        api_key=api_key,
        **kwargs
    )
    
    if response.status_code == 200:
        return response.output.choices[0].message.content[0]["text"]
    else:
        raise Exception(f"Error: {response.message}")


def main():
    """Demo usage."""
    import argparse
    parser = argparse.ArgumentParser(description="DashScope Qwen-VL Demo")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--prompt", default="这是什么?", help="Prompt for image analysis")
    parser.add_argument("--api-key", help="DashScope API key")
    args = parser.parse_args()
    
    if args.api_key:
        global API_KEY
        API_KEY = args.api_key
    
    response = analyze_image(args.image, args.prompt, api_key=args.api_key)
    print("Response:")
    print(response)


if __name__ == "__main__":
    main()
