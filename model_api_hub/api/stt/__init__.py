"""
STT (Speech-to-Text) API providers.

Available providers:
- openai_whisper - OpenAI Whisper
"""

from .openai_whisper import speech_to_text, translate_to_english

__all__ = [
    "speech_to_text",
    "translate_to_english",
]
