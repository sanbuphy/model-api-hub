"""
Kling AI (可灵AI) API client - 快手视频生成平台

Website: https://klingai.com/
Features: AI 视频生成、图像生成、视频编辑
Pricing: 免费额度 + 付费订阅
"""

import os
from typing import Optional, List, Dict, Any
import requests
import time

# API Configuration
API_KEY: str = os.getenv("KLING_API_KEY", "")
BASE_URL: str = os.getenv("KLING_BASE_URL", "https://klingai.com/api/v1")
DEFAULT_VIDEO_MODEL: str = "kling-v1.5"
DEFAULT_IMAGE_MODEL: str = "kling-image-v1"

# Available models
AVAILABLE_VIDEO_MODELS: List[str] = [
    "kling-v1.5",
    "kling-v1.0",
]

AVAILABLE_IMAGE_MODELS: List[str] = [
    "kling-image-v1",
]


def create_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """Create request headers."""
    key = api_key or API_KEY
    if not key:
        raise ValueError("Kling API key is required. Set KLING_API_KEY environment variable.")
    
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
    Generate video from text using Kling.
    
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
            f"{BASE_URL}/videos/text2video",
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


def image_to_video(
    image_path: str,
    prompt: str = "",
    output_path: str = "output.mp4",
    api_key: Optional[str] = None,
    model: str = DEFAULT_VIDEO_MODEL,
    duration: int = 5,
    **kwargs
) -> bool:
    """
    Generate video from image using Kling.
    
    Args:
        image_path: Path to input image
        prompt: Optional motion prompt
        output_path: Output file path
        api_key: API key (optional)
        model: Model identifier
        duration: Video duration in seconds
        **kwargs: Additional parameters
    
    Returns:
        True if successful, False otherwise
    """
    try:
        import base64
        
        headers = create_headers(api_key)
        
        # Encode image
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        
        payload = {
            "model": model,
            "image": f"data:image/jpeg;base64,{base64_image}",
            "prompt": prompt,
            "duration": duration,
            **kwargs
        }
        
        # Submit generation task
        response = requests.post(
            f"{BASE_URL}/videos/image2video",
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
    Generate image from text using Kling.
    
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


def main():
    """Demo usage."""
    import argparse
    parser = argparse.ArgumentParser(description="Kling AI Demo")
    parser.add_argument("--mode", choices=["text2video", "image2video", "text2image"], default="text2video", help="API mode")
    parser.add_argument("--input", required=True, help="Input text or image path")
    parser.add_argument("--prompt", help="Motion prompt for image2video mode")
    parser.add_argument("--output", default=None, help="Output file path")
    parser.add_argument("--api-key", help="Kling API key")
    parser.add_argument("--model", help="Model to use")
    args = parser.parse_args()
    
    if args.output is None:
        if args.mode == "text2image":
            args.output = "output.png"
        else:
            args.output = "output.mp4"
    
    if args.mode == "text2video":
        success = text_to_video(
            args.input,
            args.output,
            api_key=args.api_key,
            model=args.model or DEFAULT_VIDEO_MODEL
        )
    elif args.mode == "image2video":
        success = image_to_video(
            args.input,
            prompt=args.prompt or "",
            output_path=args.output,
            api_key=args.api_key,
            model=args.model or DEFAULT_VIDEO_MODEL
        )
    else:  # text2image
        success = text_to_image(
            args.input,
            args.output,
            api_key=args.api_key,
            model=args.model or DEFAULT_IMAGE_MODEL
        )
    
    if success:
        print(f"Output saved to: {args.output}")
    else:
        print("Generation failed")


if __name__ == "__main__":
    main()
