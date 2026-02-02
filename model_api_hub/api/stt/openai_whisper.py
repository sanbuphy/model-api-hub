"""
OpenAI Whisper Speech-to-Text API wrapper.

Provides interface for OpenAI's Whisper model for speech recognition.
"""

from openai import OpenAI
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

DEFAULT_MODEL: str = "whisper-1"

API_KEY: Optional[str] = None


def create_client(api_key: Optional[str] = None) -> OpenAI:
    """Create OpenAI client."""
    if api_key is None:
        api_key = get_api_key("openai")
    return OpenAI(api_key=api_key)


def speech_to_text(
    audio_path: str,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    **kwargs
) -> str:
    """Transcribe audio to text using Whisper."""
    client = create_client(api_key=api_key)
    
    with open(audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            language=language,
            prompt=prompt,
            **kwargs
        )
    
    return transcript.text


def translate_to_english(
    audio_path: str,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    **kwargs
) -> str:
    """Translate audio to English text."""
    client = create_client(api_key=api_key)
    
    with open(audio_path, "rb") as audio_file:
        translation = client.audio.translations.create(
            model=model,
            file=audio_file,
            **kwargs
        )
    
    return translation.text


def main():
    """Demo usage."""
    import argparse
    parser = argparse.ArgumentParser(description="OpenAI Whisper STT Demo")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--language", help="Language code (e.g., zh, en, ja)")
    parser.add_argument("--translate", action="store_true", help="Translate to English")
    parser.add_argument("--api-key", help="OpenAI API key")
    args = parser.parse_args()
    
    if args.api_key:
        global API_KEY
        API_KEY = args.api_key
    
    if args.translate:
        result = translate_to_english(args.audio, api_key=args.api_key)
        print("Translation (English):")
    else:
        result = speech_to_text(args.audio, api_key=args.api_key, language=args.language)
        print("Transcription:")
    
    print(result)


if __name__ == "__main__":
    main()
