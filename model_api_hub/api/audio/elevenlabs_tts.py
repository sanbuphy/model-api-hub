"""
ElevenLabs Text-to-Speech API wrapper.

Provides interface for ElevenLabs' TTS capabilities including:
- eleven_multilingual_v2 - Multilingual model
- eleven_monolingual_v1 - English only
- eleven_turbo_v2 - Faster, lower latency
"""

import requests
import os
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

# Default configuration
DEFAULT_BASE_URL: str = "https://api.elevenlabs.io/v1/text-to-speech"
DEFAULT_MODEL: str = "eleven_multilingual_v2"
DEFAULT_VOICE_ID: str = "21m00Tcm4TlvDq8ikWAM"  # "Rachel" voice


def text_to_speech(
    text: str,
    output_path: str = "output.mp3",
    api_key: Optional[str] = None,
    model_id: str = DEFAULT_MODEL,
    voice_id: str = DEFAULT_VOICE_ID,
    stability: float = 0.5,
    similarity_boost: float = 0.75
) -> bool:
    """
    Convert text to speech using ElevenLabs API.

    Args:
        text: Text to convert to speech.
        output_path: Path to save the audio file.
        api_key: ElevenLabs API key. If None, loads from environment.
        model_id: Model to use for TTS.
        voice_id: Voice ID to use.
        stability: Voice stability (0-1, higher = more stable).
        similarity_boost: Voice similarity enhancement (0-1).

    Returns:
        True if successful, False otherwise.
    """
    if api_key is None:
        api_key = get_api_key("elevenlabs")

    url = f"{DEFAULT_BASE_URL}/{voice_id}"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }

    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost
        }
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Audio saved successfully to: {output_path}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Error generating speech: {e}")
        return False


def get_available_voices(api_key: Optional[str] = None) -> Optional[dict]:
    """
    Get list of available voices from ElevenLabs.

    Args:
        api_key: ElevenLabs API key. If None, loads from environment.

    Returns:
        Dictionary with voice information, or None if failed.
    """
    if api_key is None:
        api_key = get_api_key("elevenlabs")

    url = "https://api.elevenlabs.io/v1/voices"
    headers = {"xi-api-key": api_key}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting voices: {e}")
        return None


def main() -> None:
    """
    Main function demonstrating usage with config file support.
    """
    # Option 1: Direct API key
    api_key = None  # Set your API key directly or load from .env

    # Option 2: Use config manager
    config_mgr = ConfigManager()
    audio_config = config_mgr.get_audio_config("elevenlabs")
    model = audio_config.get("model", DEFAULT_MODEL)
    voice_id = audio_config.get("voice_id", DEFAULT_VOICE_ID)

    # Test text to speech
    text = "Hello! This is a test of the ElevenLabs text to speech API."
    output_path = "output.mp3"

    print(f"Converting text to speech: {text}")
    print(f"Model: {model}")
    print(f"Voice ID: {voice_id}")
    print("-" * 50)

    success = text_to_speech(
        text=text,
        output_path=output_path,
        api_key=api_key,
        model_id=model,
        voice_id=voice_id
    )

    if success:
        print(f"Text-to-speech conversion completed successfully!")
    else:
        print("Text-to-speech conversion failed")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ElevenLabs TTS API Demo")
    parser.add_argument("--api-key", type=str, help="ElevenLabs API key")
    parser.add_argument("--text", type=str, default="Hello! This is a test.",
                        help="Text to convert to speech")
    parser.add_argument("--output", type=str, default="output.mp3",
                        help="Output audio file path")
    args = parser.parse_args()

    text_to_speech(args.text, args.output, api_key=args.api_key)
