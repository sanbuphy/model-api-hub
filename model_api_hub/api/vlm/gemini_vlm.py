"""
Google Gemini Vision API wrapper.

Provides interface for Gemini's vision-language models:
- gemini-1.5-pro - Gemini 1.5 Pro (multimodal)
- gemini-1.5-flash - Gemini 1.5 Flash (multimodal)
"""

import google.generativeai as genai
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

DEFAULT_MODEL: str = "gemini-1.5-pro"

API_KEY: Optional[str] = None


def create_client(api_key: Optional[str] = None):
    """Create Gemini client."""
    if api_key is None:
        api_key = get_api_key("gemini")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(DEFAULT_MODEL)


def analyze_image(
    image_path: str,
    prompt: str = "What's in this image?",
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    **kwargs
) -> str:
    """Analyze image using Gemini VLM."""
    if api_key is None:
        api_key = get_api_key("gemini")
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel(model_name)
    
    # Load image
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    response = model.generate_content(
        [prompt, {"mime_type": "image/jpeg", "data": image_data}],
        **kwargs
    )
    return response.text


def main():
    """Demo usage."""
    import argparse
    parser = argparse.ArgumentParser(description="Gemini VLM Demo")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--prompt", default="What's in this image?", help="Prompt for image analysis")
    parser.add_argument("--api-key", help="Gemini API key")
    args = parser.parse_args()
    
    if args.api_key:
        global API_KEY
        API_KEY = args.api_key
    
    response = analyze_image(args.image, args.prompt, api_key=args.api_key)
    print("Response:")
    print(response)


if __name__ == "__main__":
    main()
