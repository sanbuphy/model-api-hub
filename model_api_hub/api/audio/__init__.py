"""
Audio (TTS) API providers.

Available providers:
- azure_tts - Azure Speech Services
- baidu_tts - Baidu Speech
"""

from .azure_tts import text_to_speech as azure_text_to_speech
from .baidu_tts import text_to_speech as baidu_text_to_speech

__all__ = [
    "azure_text_to_speech",
    "baidu_text_to_speech",
]
