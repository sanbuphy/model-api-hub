"""
Audio (TTS) API providers.

Available providers:
- azure_tts - Azure Speech Services
- baidu_tts - Baidu Speech
- minimax_tts - MiniMax TTS & Voice Clone
"""

from .azure_tts import text_to_speech as azure_text_to_speech
from .baidu_tts import text_to_speech as baidu_text_to_speech
from .minimax_tts import text_to_speech as minimax_text_to_speech, clone_voice as minimax_clone_voice, speech_to_text as minimax_stt

__all__ = [
    "azure_text_to_speech",
    "baidu_text_to_speech",
    "minimax_text_to_speech",
    "minimax_clone_voice",
    "minimax_stt",
]
