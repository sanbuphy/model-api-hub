"""
Hailuo AI (海螺AI) API client - MiniMax 视频生成平台

Website: https://hailuoai.video/
Features: AI 视频生成、数字人、语音合成
Pricing: 免费额度 + 按量付费
"""

import os
from typing import Optional, List, Dict, Any
import requests
import time

# API Configuration
API_KEY: str = os.getenv("HAILUO_API_KEY", "")
BASE_URL: str = os.getenv("HAILUO_BASE_URL", "https://hailuoai.video/api/v1")
DEFAULT_VIDEO_MODEL: str = "hailuo-video-1.0"
DEFAULT_TTS_MODEL: str = "hailuo-tts-1.0"

# Available models
AVAILABLE_VIDEO_MODELS: List[str] = [
    "hailuo-video-1.0",
    "hailuo-video-0.5",
]

AVAILABLE_TTS_MODELS: List[str] = [
    "hailuo-tts-1.0",
    "hailuo-tts-clone",
]


def create_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """Create request headers."""
    key = api_key or API_KEY
    if not key:
        raise ValueError("Hailuo API key is required. Set HAILUO_API_KEY environment variable.")
    
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def text_to_video(
    prompt: str,
    output_path: str = "output.mp4",
    api_key: Optional[str] = None,
    model: str = DEFAULT_VIDEO_MODEL,
    duration: int = 5,
    resolution: str = "1080p",
    **kwargs
) -> bool:
    """
    Generate video from text using Hailuo.
    
    Args:
        prompt: Video generation prompt
        output_path: Output file path
        api_key: API key (optional)
        model: Model identifier
        duration: Video duration in seconds
        resolution: Video resolution
        **kwargs: Additional parameters
    
    Returns:
        True if successful, False otherwise
    """
    try:
        headers = create_headers(api_key)
        
        payload = {
            "model": model,
            "prompt": prompt,
            "duration": duration,
            "resolution": resolution,
            **kwargs
        }
        
        # Submit generation task
        response = requests.post(
            f"{BASE_URL}/videos/generations",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        task_id = result.get("task_id")
        
        if not task_id:
            print(f"Error: No task ID in response - {result}")
            return False
        
        # Poll for result
        max_retries = 60
        for _ in range(max_retries):
            time.sleep(5)
            
            status_response = requests.get(
                f"{BASE_URL}/tasks/{task_id}",
                headers=headers,
                timeout=30
            )
            status_response.raise_for_status()
            
            status_result = status_response.json()
            
            if status_result.get("status") == "completed":
                video_url = status_result.get("output", {}).get("video_url")
                if video_url:
                    video_response = requests.get(video_url, timeout=120)
                    video_response.raise_for_status()
                    
                    with open(output_path, "wb") as f:
                        f.write(video_response.content)
                    return True
            elif status_result.get("status") == "failed":
                print(f"Video generation failed: {status_result}")
                return False
        
        print("Timeout waiting for video generation")
        return False
        
    except Exception as e:
        print(f"Error generating video: {e}")
        return False


def text_to_speech(
    text: str,
    output_path: str = "output.mp3",
    api_key: Optional[str] = None,
    model: str = DEFAULT_TTS_MODEL,
    voice_id: str = "default",
    speed: float = 1.0,
    **kwargs
) -> bool:
    """
    Convert text to speech using Hailuo.
    
    Args:
        text: Text to convert
        output_path: Output audio file path
        api_key: API key (optional)
        model: Model identifier
        voice_id: Voice identifier
        speed: Speech speed
        **kwargs: Additional parameters
    
    Returns:
        True if successful, False otherwise
    """
    try:
        headers = create_headers(api_key)
        
        payload = {
            "model": model,
            "input": text,
            "voice": voice_id,
            "speed": speed,
            **kwargs
        }
        
        response = requests.post(
            f"{BASE_URL}/audio/speech",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        
        if "audio_url" in result:
            audio_response = requests.get(result["audio_url"], timeout=30)
            audio_response.raise_for_status()
            
            with open(output_path, "wb") as f:
                f.write(audio_response.content)
            return True
        else:
            print(f"Error: No audio URL in response - {result}")
            return False
            
    except Exception as e:
        print(f"Error generating speech: {e}")
        return False


def clone_voice(
    audio_path: str,
    text: str,
    output_path: str = "output_cloned.mp3",
    api_key: Optional[str] = None,
    **kwargs
) -> bool:
    """
    Clone voice from audio sample and generate speech.
    
    Args:
        audio_path: Path to voice sample audio file
        text: Text to synthesize
        output_path: Output audio file path
        api_key: API key (optional)
        **kwargs: Additional parameters
    
    Returns:
        True if successful, False otherwise
    """
    try:
        headers = create_headers(api_key)
        
        # Upload voice sample
        with open(audio_path, "rb") as f:
            files = {"audio": f}
            upload_response = requests.post(
                f"{BASE_URL}/audio/voices",
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
            "model": "hailuo-tts-clone",
            "input": text,
            "voice": voice_id,
            **kwargs
        }
        
        response = requests.post(
            f"{BASE_URL}/audio/speech",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        
        if "audio_url" in result:
            audio_response = requests.get(result["audio_url"], timeout=30)
            audio_response.raise_for_status()
            
            with open(output_path, "wb") as f:
                f.write(audio_response.content)
            return True
        else:
            print(f"Error: No audio URL in response - {result}")
            return False
            
    except Exception as e:
        print(f"Error cloning voice: {e}")
        return False


def main():
    """Demo usage."""
    import argparse
    parser = argparse.ArgumentParser(description="Hailuo AI Demo")
    parser.add_argument("--mode", choices=["video", "tts", "clone"], default="video", help="API mode")
    parser.add_argument("--input", required=True, help="Input text or audio path")
    parser.add_argument("--text", help="Text for voice cloning mode")
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--api-key", help="Hailuo API key")
    parser.add_argument("--model", help="Model to use")
    args = parser.parse_args()
    
    if args.output is None:
        if args.mode == "video":
            args.output = "output.mp4"
        else:
            args.output = "output.mp3"
    
    if args.mode == "video":
        success = text_to_video(
            args.input,
            args.output,
            api_key=args.api_key,
            model=args.model or DEFAULT_VIDEO_MODEL
        )
    elif args.mode == "tts":
        success = text_to_speech(
            args.input,
            args.output,
            api_key=args.api_key,
            model=args.model or DEFAULT_TTS_MODEL
        )
    else:  # clone
        if not args.text:
            print("Error: --text is required for clone mode")
            return
        success = clone_voice(
            args.input,
            args.text,
            args.output,
            api_key=args.api_key
        )
    
    if success:
        print(f"Output saved to: {args.output}")
    else:
        print("Generation failed")


if __name__ == "__main__":
    main()
