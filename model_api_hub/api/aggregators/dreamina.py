"""
Dreamina (即梦) API client - 字节跳动图像/视频生成平台

Website: https://jimeng.jianying.com/
Features: AI 绘画、视频生成、智能画布
Pricing: 免费额度 + 付费订阅
"""

import os
from typing import Optional, List, Dict, Any
import requests
import time

# API Configuration
API_KEY: str = os.getenv("DREAMINA_API_KEY", "")
BASE_URL: str = os.getenv("DREAMINA_BASE_URL", "https://jimeng.jianying.com/api/v1")
DEFAULT_IMAGE_MODEL: str = "jimeng-2.0"
DEFAULT_VIDEO_MODEL: str = "jimeng-video-1.0"

# Available models
AVAILABLE_IMAGE_MODELS: List[str] = [
    "jimeng-2.0",
    "jimeng-1.5",
    "jimeng-1.0",
]

AVAILABLE_VIDEO_MODELS: List[str] = [
    "jimeng-video-1.0",
    "jimeng-video-0.5",
]


def create_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """Create request headers."""
    key = api_key or API_KEY
    if not key:
        raise ValueError("Dreamina API key is required. Set DREAMINA_API_KEY environment variable.")
    
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def text_to_image(
    prompt: str,
    output_path: str = "output.png",
    api_key: Optional[str] = None,
    model: str = DEFAULT_IMAGE_MODEL,
    width: int = 1024,
    height: int = 1024,
    **kwargs
) -> bool:
    """
    Generate image from text using Dreamina.
    
    Args:
        prompt: Image generation prompt
        output_path: Output file path
        api_key: API key (optional)
        model: Model identifier
        width: Image width
        height: Image height
        **kwargs: Additional parameters
    
    Returns:
        True if successful, False otherwise
    """
    try:
        headers = create_headers(api_key)
        
        payload = {
            "model": model,
            "prompt": prompt,
            "width": width,
            "height": height,
            **kwargs
        }
        
        response = requests.post(
            f"{BASE_URL}/images/generations",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        
        if "data" in result and len(result["data"]) > 0:
            image_url = result["data"][0]["url"]
            
            # Download image
            img_response = requests.get(image_url, timeout=30)
            img_response.raise_for_status()
            
            with open(output_path, "wb") as f:
                f.write(img_response.content)
            return True
        else:
            print(f"Error: No image data in response - {result}")
            return False
            
    except Exception as e:
        print(f"Error generating image: {e}")
        return False


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
    parser = argparse.ArgumentParser(description="Dreamina Image/Video Generation Demo")
    parser.add_argument("--mode", choices=["image", "video"], default="image", help="Generation mode")
    parser.add_argument("--prompt", required=True, help="Generation prompt")
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--api-key", help="Dreamina API key")
    parser.add_argument("--model", help="Model to use")
    args = parser.parse_args()
    
    if args.output is None:
        args.output = "output.png" if args.mode == "image" else "output.mp4"
    
    if args.mode == "image":
        success = text_to_image(
            args.prompt,
            args.output,
            api_key=args.api_key,
            model=args.model or DEFAULT_IMAGE_MODEL
        )
    else:
        success = text_to_video(
            args.prompt,
            args.output,
            api_key=args.api_key,
            model=args.model or DEFAULT_VIDEO_MODEL
        )
    
    if success:
        print(f"Output saved to: {args.output}")
    else:
        print("Generation failed")


if __name__ == "__main__":
    main()
