"""
OpenAI Text-to-Speech API wrapper.

Provides interface for OpenAI's TTS capabilities including:
- tts-1 - Standard model
- tts-1-hd - High quality model

Voices: alloy, echo, fable, onyx, nova, shimmer
"""

from openai import OpenAI
import os
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

# Default configuration
DEFAULT_MODEL: str = "tts-1"
DEFAULT_VOICE: str = "alloy"


def text_to_speech(
    text: str,
    output_path: str = "output.mp3",
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    voice: str = DEFAULT_VOICE
) -> bool:
    """
    Convert text to speech using OpenAI API.

    Args:
        text: Text to convert to speech (max 4096 characters).
        output_path: Path to save the audio file.
        api_key: OpenAI API key. If None, loads from environment.
        model: Model to use (tts-1 or tts-1-hd).
        voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer).

    Returns:
        True if successful, False otherwise.
    """
    if api_key is None:
        api_key = get_api_key("openai")

    client = OpenAI(api_key=api_key)

    try:
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text
        )

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        response.stream_to_file(output_path)
        print(f"Audio saved successfully to: {output_path}")
        return True

    except Exception as e:
        print(f"Error generating speech: {e}")
        return False


def main() -> None:
    """
    Main function demonstrating usage with config file support.
    """
    # Option 1: Direct API key
    api_key = None  # Set your API key directly or load from .env

    # Test text to speech
    text = "Hello! This is a test of the OpenAI text to speech API."
    output_path = "output.mp3"

    print(f"Converting text to speech: {text}")
    print(f"Model: {DEFAULT_MODEL}")
    print(f"Voice: {DEFAULT_VOICE}")
    print("-" * 50)

    success = text_to_speech(
        text=text,
        output_path=output_path,
        api_key=api_key
    )

    if success:
        print(f"Text-to-speech conversion completed successfully!")
    else:
        print("Text-to-speech conversion failed")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OpenAI TTS API Demo")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--text", type=str, default="Hello! This is a test.",
                        help="Text to convert to speech")
    parser.add_argument("--output", type=str, default="output.mp3",
                        help="Output audio file path")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE,
                        choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                        help="Voice to use")
    args = parser.parse_args()

    text_to_speech(args.text, args.output, api_key=args.api_key, voice=args.voice)
