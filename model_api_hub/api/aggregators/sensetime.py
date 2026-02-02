"""
SenseTime (商汤科技) API client - 日日新大模型平台

Website: https://platform.sensetime.com/
Features: 大语言模型、多模态、文生图、文生视频
Pricing: 免费额度 + 按量付费
"""

import os
from typing import Optional, List, Dict, Any
import requests
import time

# API Configuration
API_KEY: str = os.getenv("SENSETIME_API_KEY", "")
BASE_URL: str = os.getenv("SENSETIME_BASE_URL", "https://platform.sensetime.com/api/v1")
DEFAULT_LLM_MODEL: str = "sensechat-5"
DEFAULT_VLM_MODEL: str = "sensechat-vision"

# Available models
AVAILABLE_LLM_MODELS: List[str] = [
    "sensechat-5",
    "sensechat-4",
    "sensechat-turbo",
]

AVAILABLE_VLM_MODELS: List[str] = [
    "sensechat-vision",
]


def create_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """Create request headers."""
    key = api_key or API_KEY
    if not key:
        raise ValueError("SenseTime API key is required. Set SENSETIME_API_KEY environment variable.")
    
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def chat(
    message: str,
    model: str = DEFAULT_LLM_MODEL,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
    **kwargs
) -> str:
    """
    Chat using SenseTime LLM.
    
    Args:
        message: User message
        model: Model identifier
        api_key: API key (optional)
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        system_prompt: System prompt
        **kwargs: Additional parameters
    
    Returns:
        Response text
    """
    headers = create_headers(api_key)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": message})
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        **kwargs
    }
    
    if max_tokens:
        payload["max_tokens"] = max_tokens
    
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    
    result = response.json()
    return result["choices"][0]["message"]["content"]


def analyze_image(
    image_path: str,
    prompt: str = "描述这张图片",
    api_key: Optional[str] = None,
    model: str = DEFAULT_VLM_MODEL,
    **kwargs
) -> str:
    """
    Analyze image using SenseTime VLM.
    
    Args:
        image_path: Path to image file
        prompt: Text prompt for image analysis
        api_key: API key (optional)
        model: Model identifier
        **kwargs: Additional parameters
    
    Returns:
        Response text
    """
    import base64
    
    headers = create_headers(api_key)
    
    # Encode image
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    payload = {
        "model": model,
        "messages": messages,
        **kwargs
    }
    
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    
    result = response.json()
    return result["choices"][0]["message"]["content"]


def text_to_image(
    prompt: str,
    output_path: str = "output.png",
    api_key: Optional[str] = None,
    model: str = "sensemirage",
    width: int = 1024,
    height: int = 1024,
    **kwargs
) -> bool:
    """
    Generate image from text using SenseTime.
    
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


def text_to_video(
    prompt: str,
    output_path: str = "output.mp4",
    api_key: Optional[str] = None,
    model: str = "senserave",
    duration: int = 5,
    **kwargs
) -> bool:
    """
    Generate video from text using SenseTime.
    
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
    parser = argparse.ArgumentParser(description="SenseTime API Demo")
    parser.add_argument("--mode", choices=["chat", "vlm", "image", "video"], default="chat", help="API mode")
    parser.add_argument("--input", required=True, help="Input text or image path")
    parser.add_argument("--prompt", help="Prompt for VLM mode")
    parser.add_argument("--output", default="output.png", help="Output file path")
    parser.add_argument("--api-key", help="SenseTime API key")
    parser.add_argument("--model", help="Model to use")
    args = parser.parse_args()
    
    if args.mode == "chat":
        response = chat(
            args.input,
            api_key=args.api_key,
            model=args.model or DEFAULT_LLM_MODEL
        )
        print("Response:")
        print(response)
    elif args.mode == "vlm":
        response = analyze_image(
            args.input,
            prompt=args.prompt or "描述这张图片",
            api_key=args.api_key,
            model=args.model or DEFAULT_VLM_MODEL
        )
        print("Response:")
        print(response)
    elif args.mode == "image":
        success = text_to_image(
            args.input,
            args.output,
            api_key=args.api_key,
            model=args.model or "sensemirage"
        )
        if success:
            print(f"Image saved to: {args.output}")
        else:
            print("Image generation failed")
    else:  # video
        success = text_to_video(
            args.input,
            args.output,
            api_key=args.api_key,
            model=args.model or "senserave"
        )
        if success:
            print(f"Video saved to: {args.output}")
        else:
            print("Video generation failed")


if __name__ == "__main__":
    main()
