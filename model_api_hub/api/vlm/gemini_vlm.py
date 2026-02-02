"""
Google Gemini Vision API wrapper.

Provides interface for Gemini's vision-language models:
- gemini-1.5-pro - Gemini 1.5 Pro (multimodal)
- gemini-1.5-flash - Gemini 1.5 Flash (multimodal)

Migration: Updated to use google.genai (new SDK) instead of deprecated google.generativeai
"""

from google import genai
from google.genai import types
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

DEFAULT_MODEL: str = "gemini-1.5-pro"

API_KEY: Optional[str] = None


def create_client(api_key: Optional[str] = None) -> genai.Client:
    """Create Gemini client."""
    if api_key is None:
        api_key = get_api_key("gemini")
    return genai.Client(api_key=api_key)


def analyze_image(
    image_path: str,
    prompt: str = "What's in this image?",
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    **kwargs
) -> str:
    """Analyze image using Gemini VLM."""
    client = create_client(api_key)
    
    # Upload image file
    image_file = client.files.upload(file=image_path)
    
    response = client.models.generate_content(
        model=model,
        contents=[prompt, image_file],
        config=types.GenerateContentConfig(**kwargs)
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
    
    response = analyze_image(args.image, args.prompt, api_key=args.api_key)
    print("Response:")
    print(response)


if __name__ == "__main__":
    main()
