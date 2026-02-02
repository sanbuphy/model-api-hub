"""
Baidu Speech TTS API wrapper.

Provides interface for Baidu Text-to-Speech.
"""

import requests
import json
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

DEFAULT_VOICE: int = 0  # 0: Female, 1: Male
DEFAULT_SPEED: int = 5  # 0-15
DEFAULT_PITCH: int = 5  # 0-15
DEFAULT_VOLUME: int = 5  # 0-15

API_KEY: Optional[str] = None
SECRET_KEY: Optional[str] = None
APP_ID: Optional[str] = None

TOKEN_URL: str = "https://aip.baidubce.com/oauth/2.0/token"
TTS_URL: str = "https://tsn.baidu.com/text2audio"


def get_access_token(api_key: str, secret_key: str) -> str:
    """Get Baidu access token."""
    params = {
        "grant_type": "client_credentials",
        "client_id": api_key,
        "client_secret": secret_key
    }
    response = requests.post(TOKEN_URL, params=params)
    return response.json().get("access_token")


def text_to_speech(
    text: str,
    output_path: str = "output.mp3",
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    voice: int = DEFAULT_VOICE,
    speed: int = DEFAULT_SPEED,
    pitch: int = DEFAULT_PITCH,
    volume: int = DEFAULT_VOLUME,
    **kwargs
) -> bool:
    """Convert text to speech using Baidu TTS."""
    try:
        if api_key is None:
            api_key = get_api_key("baidu_tts")
        if secret_key is None:
            secret_key = get_api_key("baidu_tts_secret_key")
        
        access_token = get_access_token(api_key, secret_key)
        
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        payload = {
            "tex": text,
            "tok": access_token,
            "cuid": "model_api_hub",
            "ctp": 1,
            "lan": "zh",
            "spd": speed,
            "pit": pitch,
            "vol": volume,
            "per": voice,
            **kwargs
        }
        
        response = requests.post(TTS_URL, headers=headers, data=payload)
        
        if response.headers.get("Content-Type") == "audio/mp3":
            with open(output_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error generating speech: {e}")
        return False


def main():
    """Demo usage."""
    import argparse
    parser = argparse.ArgumentParser(description="Baidu TTS Demo")
    parser.add_argument("--text", required=True, help="Text to convert to speech")
    parser.add_argument("--output", default="output.mp3", help="Output audio file")
    parser.add_argument("--voice", type=int, default=DEFAULT_VOICE, help="Voice (0=female, 1=male)")
    parser.add_argument("--api-key", help="Baidu API key")
    parser.add_argument("--secret-key", help="Baidu Secret key")
    args = parser.parse_args()
    
    if args.api_key:
        global API_KEY
        API_KEY = args.api_key
    if args.secret_key:
        global SECRET_KEY
        SECRET_KEY = args.secret_key
    
    success = text_to_speech(
        args.text,
        args.output,
        api_key=args.api_key,
        secret_key=args.secret_key,
        voice=args.voice
    )
    
    if success:
        print(f"Audio saved to: {args.output}")
    else:
        print("TTS failed")


if __name__ == "__main__":
    main()
