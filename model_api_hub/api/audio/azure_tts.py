"""
Azure Speech Services TTS API wrapper.

Provides interface for Azure Text-to-Speech:
- en-US-JennyNeural - Jenny (US English)
- zh-CN-XiaoxiaoNeural - Xiaoxiao (Chinese)
- ja-JP-NanamiNeural - Nanami (Japanese)
"""

import azure.cognitiveservices.speech as speechsdk
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

DEFAULT_VOICE: str = "en-US-JennyNeural"

API_KEY: Optional[str] = None
REGION: Optional[str] = None


def create_speech_config(
    api_key: Optional[str] = None,
    region: Optional[str] = None
) -> speechsdk.SpeechConfig:
    """Create Azure Speech config."""
    if api_key is None:
        api_key = get_api_key("azure_speech")
    if region is None:
        region = get_api_key("azure_speech_region") or "eastus"
    
    speech_config = speechsdk.SpeechConfig(subscription=api_key, region=region)
    return speech_config


def text_to_speech(
    text: str,
    output_path: str = "output.wav",
    api_key: Optional[str] = None,
    region: Optional[str] = None,
    voice: str = DEFAULT_VOICE,
    **kwargs
) -> bool:
    """Convert text to speech using Azure TTS."""
    try:
        speech_config = create_speech_config(api_key, region)
        speech_config.speech_synthesis_voice_name = voice
        
        # Set output to file
        audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        
        result = synthesizer.speak_text_async(text).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return True
        else:
            print(f"Error: {result.reason}")
            return False
            
    except Exception as e:
        print(f"Error generating speech: {e}")
        return False


def main():
    """Demo usage."""
    import argparse
    parser = argparse.ArgumentParser(description="Azure TTS Demo")
    parser.add_argument("--text", required=True, help="Text to convert to speech")
    parser.add_argument("--output", default="output.wav", help="Output audio file")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="Voice to use")
    parser.add_argument("--api-key", help="Azure Speech API key")
    parser.add_argument("--region", default="eastus", help="Azure region")
    args = parser.parse_args()
    
    if args.api_key:
        global API_KEY
        API_KEY = args.api_key
    if args.region:
        global REGION
        REGION = args.region
    
    success = text_to_speech(
        args.text,
        args.output,
        api_key=args.api_key,
        region=args.region,
        voice=args.voice
    )
    
    if success:
        print(f"Audio saved to: {args.output}")
    else:
        print("TTS failed")


if __name__ == "__main__":
    main()
