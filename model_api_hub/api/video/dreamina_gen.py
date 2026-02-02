"""
Dreamina (即梦) Video Generation API - 字节跳动视频生成平台

Website: https://jimeng.jianying.com/
Features: 文生视频、图生视频、视频编辑
Pricing: 免费额度 + 付费订阅
"""

import os
from typing import Optional, List, Dict, Any
import requests
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key

# API Configuration
DEFAULT_VIDEO_MODEL: str = "jimeng-video-1.0"
BASE_URL: str = "https://jimeng.jianying.com/api/v1"

# Available models
AVAILABLE_VIDEO_MODELS: List[str] = [
    "jimeng-video-1.0",
    "jimeng-video-0.5",
]


def create_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """Create request headers."""
    key = api_key or get_api_key("dreamina")
    if not key:
        raise ValueError("Dreamina API key is required. Set DREAMINA_API_KEY environment variable.")
    
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
    **kwargs
) -> bool:
    """
    Generate video from text using Dreamina.
    
    Args:
        prompt: Video generation prompt
        output_path: Output file path
        api_key: API key (optional)
        model: Model identifier
        duration: Video duration in seconds
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


def main():
    """Demo usage."""
    import argparse
    parser = argparse.ArgumentParser(description="Dreamina Video Generation Demo")
    parser.add_argument("--prompt", required=True, help="Video generation prompt")
    parser.add_argument("--output", default="dreamina_video.mp4", help="Output file path")
    parser.add_argument("--api-key", help="Dreamina API key")
    parser.add_argument("--model", default=DEFAULT_VIDEO_MODEL, help="Model to use")
    parser.add_argument("--duration", type=int, default=5, help="Video duration in seconds")
    args = parser.parse_args()
    
    print(f"Generating video with prompt: {args.prompt}")
    success = text_to_video(
        prompt=args.prompt,
        output_path=args.output,
        api_key=args.api_key,
        model=args.model,
        duration=args.duration
    )
    
    if success:
        print(f"Video saved to: {args.output}")
    else:
        print("Failed to generate video")


if __name__ == "__main__":
    main()
