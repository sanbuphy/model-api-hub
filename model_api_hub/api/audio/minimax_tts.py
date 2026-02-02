"""
MiniMax Audio API client - 语音合成、声音克隆、语音识别

Website: https://www.minimaxi.com/
Features: TTS、声音克隆、语音识别、语音对话
Pricing: 免费额度 + 按量付费
"""

import os
from typing import Optional, List, Dict, Any
import requests

# API Configuration
API_KEY: str = os.getenv("MINIMAX_AUDIO_API_KEY", "")
GROUP_ID: str = os.getenv("MINIMAX_GROUP_ID", "")
BASE_URL: str = os.getenv("MINIMAX_AUDIO_BASE_URL", "https://api.minimaxi.chat/v1")
DEFAULT_TTS_MODEL: str = "speech-01"

# Available models
AVAILABLE_TTS_MODELS: List[str] = [
    "speech-01",
    "speech-01-turbo",
]

AVAILABLE_VOICE_IDS: List[str] = [
    "male-qn-qingse",
    "male-qn-jingying",
    "female-shaonv",
    "female-yujie",
]


def create_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """Create request headers."""
    key = api_key or API_KEY
    if not key:
        raise ValueError("MiniMax API key is required. Set MINIMAX_AUDIO_API_KEY environment variable.")
    
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def text_to_speech(
    text: str,
    output_path: str = "output.mp3",
    api_key: Optional[str] = None,
    model: str = DEFAULT_TTS_MODEL,
    voice_id: str = "male-qn-qingse",
    speed: float = 1.0,
    **kwargs
) -> bool:
    """
    Convert text to speech using MiniMax.
    
    Args:
        text: Text to convert
        output_path: Output audio file path
        api_key: API key (optional)
        model: Model identifier
        voice_id: Voice identifier
        speed: Speech speed (0.5-2.0)
        **kwargs: Additional parameters
    
    Returns:
        True if successful, False otherwise
    """
    try:
        headers = create_headers(api_key)
        
        payload = {
            "model": model,
            "text": text,
            "voice_id": voice_id,
            "speed": speed,
            **kwargs
        }
        
        response = requests.post(
            f"{BASE_URL}/text_to_speech",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        
        if "audio_hex" in result:
            import binascii
            audio_data = binascii.unhexlify(result["audio_hex"])
            
            with open(output_path, "wb") as f:
                f.write(audio_data)
            return True
        elif "audio_url" in result:
            audio_response = requests.get(result["audio_url"], timeout=30)
            audio_response.raise_for_status()
            
            with open(output_path, "wb") as f:
                f.write(audio_response.content)
            return True
        else:
            print(f"Error: No audio data in response - {result}")
            return False
            
    except Exception as e:
        print(f"Error generating speech: {e}")
        return False


def clone_voice(
    audio_path: str,
    text: str,
    output_path: str = "output_cloned.mp3",
    api_key: Optional[str] = None,
    model: str = DEFAULT_TTS_MODEL,
    **kwargs
) -> bool:
    """
    Clone voice from audio sample and generate speech.
    
    Args:
        audio_path: Path to voice sample audio file (3-10 seconds recommended)
        text: Text to synthesize
        output_path: Output audio file path
        api_key: API key (optional)
        model: Model identifier
        **kwargs: Additional parameters
    
    Returns:
        True if successful, False otherwise
    """
    try:
        headers = create_headers(api_key)
        
        # Upload voice sample
        with open(audio_path, "rb") as f:
            files = {"voice_file": f}
            upload_response = requests.post(
                f"{BASE_URL}/voice_clone",
                headers={"Authorization": headers["Authorization"]},
                files=files,
                timeout=60
            )
            upload_response.raise_for_status()
            upload_result = upload_response.json()
            voice_id = upload_result.get("voice_id")
        
        if not voice_id:
            print("Error: Failed to upload voice sample")
            return False
        
        # Generate speech with cloned voice
        payload = {
            "model": model,
            "text": text,
            "voice_id": voice_id,
            **kwargs
        }
        
        response = requests.post(
            f"{BASE_URL}/text_to_speech",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        
        if "audio_hex" in result:
            import binascii
            audio_data = binascii.unhexlify(result["audio_hex"])
            
            with open(output_path, "wb") as f:
                f.write(audio_data)
            return True
        elif "audio_url" in result:
            audio_response = requests.get(result["audio_url"], timeout=30)
            audio_response.raise_for_status()
            
            with open(output_path, "wb") as f:
                f.write(audio_response.content)
            return True
        else:
            print(f"Error: No audio data in response - {result}")
            return False
            
    except Exception as e:
        print(f"Error cloning voice: {e}")
        return False


def speech_to_text(
    audio_path: str,
    api_key: Optional[str] = None,
    model: str = "speech_recognition",
    **kwargs
) -> str:
    """
    Convert speech to text using MiniMax.
    
    Args:
        audio_path: Path to audio file
        api_key: API key (optional)
        model: Model identifier
        **kwargs: Additional parameters
    
    Returns:
        Transcribed text
    """
    try:
        headers = create_headers(api_key)
        
        with open(audio_path, "rb") as f:
            files = {"audio": f}
            response = requests.post(
                f"{BASE_URL}/speech_to_text",
                headers={"Authorization": headers["Authorization"]},
                files=files,
                data={"model": model, **kwargs},
                timeout=60
            )
        
        response.raise_for_status()
        result = response.json()
        
        return result.get("text", "")
        
    except Exception as e:
        print(f"Error transcribing speech: {e}")
        return ""


def main():
    """Demo usage."""
    import argparse
    parser = argparse.ArgumentParser(description="MiniMax Audio Demo")
    parser.add_argument("--mode", choices=["tts", "clone", "stt"], default="tts", help="API mode")
    parser.add_argument("--input", required=True, help="Input text or audio path")
    parser.add_argument("--text", help="Text for voice cloning mode")
    parser.add_argument("--output", default="output.mp3", help="Output file path")
    parser.add_argument("--api-key", help="MiniMax API key")
    parser.add_argument("--model", help="Model to use")
    parser.add_argument("--voice-id", default="male-qn-qingse", help="Voice ID for TTS")
    args = parser.parse_args()
    
    if args.mode == "tts":
        success = text_to_speech(
            args.input,
            args.output,
            api_key=args.api_key,
            model=args.model or DEFAULT_TTS_MODEL,
            voice_id=args.voice_id
        )
        if success:
            print(f"Audio saved to: {args.output}")
        else:
            print("TTS failed")
    elif args.mode == "clone":
        if not args.text:
            print("Error: --text is required for clone mode")
            return
        success = clone_voice(
            args.input,
            args.text,
            args.output,
            api_key=args.api_key,
            model=args.model or DEFAULT_TTS_MODEL
        )
        if success:
            print(f"Cloned audio saved to: {args.output}")
        else:
            print("Voice cloning failed")
    else:  # stt
        text = speech_to_text(
            args.input,
            api_key=args.api_key,
            model=args.model or "speech_recognition"
        )
        print("Transcribed text:")
        print(text)


if __name__ == "__main__":
    main()
