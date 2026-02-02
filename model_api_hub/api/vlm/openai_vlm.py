"""
OpenAI GPT-4 Vision API wrapper.

Provides interface for OpenAI's vision-language models:
- gpt-4o - GPT-4o (multimodal)
- gpt-4o-mini - GPT-4o Mini (multimodal)
- gpt-4-turbo - GPT-4 Turbo with vision
"""

from openai import OpenAI
from typing import Dict, Any, List, Optional
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

DEFAULT_MODEL: str = "gpt-4o"

API_KEY: Optional[str] = None


def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def create_client(api_key: Optional[str] = None) -> OpenAI:
    """Create OpenAI client."""
    if api_key is None:
        api_key = get_api_key("openai")
    return OpenAI(api_key=api_key)


def analyze_image(
    image_path: str,
    prompt: str = "What's in this image?",
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    **kwargs
) -> str:
    """Analyze image using OpenAI VLM."""
    client = create_client(api_key=api_key)
    base64_image = encode_image(image_path)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        **kwargs
    )
    return response.choices[0].message.content


def main():
    """Demo usage."""
    import argparse
    parser = argparse.ArgumentParser(description="OpenAI VLM Demo")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--prompt", default="What's in this image?", help="Prompt for image analysis")
    parser.add_argument("--api-key", help="OpenAI API key")
    args = parser.parse_args()
    
    if args.api_key:
        global API_KEY
        API_KEY = args.api_key
    
    response = analyze_image(args.image, args.prompt, api_key=args.api_key)
    print("Response:")
    print(response)


if __name__ == "__main__":
    main()
