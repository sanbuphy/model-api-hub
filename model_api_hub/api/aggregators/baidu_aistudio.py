"""
Baidu AI Studio (飞桨AI Studio) API client - 百度模型开发平台

Website: https://aistudio.baidu.com/
Features: 模型训练、推理部署、Notebook开发
Pricing: 免费额度 + 付费实例
"""

import os
from typing import Optional, List, Dict, Any
import requests

# API Configuration
API_KEY: str = os.getenv("BAIDU_AISTUDIO_API_KEY", "")
BASE_URL: str = os.getenv("BAIDU_AISTUDIO_BASE_URL", "https://aistudio.baidu.com/studio/api/v1")
DEFAULT_MODEL: str = "ernie-bot-4"

# Available models
AVAILABLE_MODELS: List[str] = [
    "ernie-bot-4",
    "ernie-bot",
    "ernie-bot-turbo",
    "ernie-4.0-8k",
    "ernie-3.5-8k",
    "ernie-speed",
    "ernie-lite",
]


def create_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """Create request headers."""
    key = api_key or API_KEY
    if not key:
        raise ValueError("Baidu AI Studio API key is required. Set BAIDU_AISTUDIO_API_KEY environment variable.")
    
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def chat(
    message: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
    **kwargs
) -> str:
    """
    Chat using Baidu AI Studio.
    
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
    model: str = "ernie-vision-4.0",
    **kwargs
) -> str:
    """
    Analyze image using Baidu AI Studio VLM.
    
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
    model: str = "ernie-image-v1",
    width: int = 1024,
    height: int = 1024,
    **kwargs
) -> bool:
    """
    Generate image from text using Baidu AI Studio.
    
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
    parser = argparse.ArgumentParser(description="Baidu AI Studio Demo")
    parser.add_argument("--mode", choices=["chat", "vlm", "image"], default="chat", help="API mode")
    parser.add_argument("--input", required=True, help="Input text or image path")
    parser.add_argument("--prompt", help="Prompt for VLM mode")
    parser.add_argument("--output", default="output.png", help="Output file path (for image mode)")
    parser.add_argument("--api-key", help="Baidu AI Studio API key")
    parser.add_argument("--model", help="Model to use")
    args = parser.parse_args()
    
    if args.mode == "chat":
        response = chat(
            args.input,
            api_key=args.api_key,
            model=args.model or DEFAULT_MODEL
        )
        print("Response:")
        print(response)
    elif args.mode == "vlm":
        response = analyze_image(
            args.input,
            prompt=args.prompt or "描述这张图片",
            api_key=args.api_key,
            model=args.model or "ernie-vision-4.0"
        )
        print("Response:")
        print(response)
    else:  # image
        success = text_to_image(
            args.input,
            args.output,
            api_key=args.api_key,
            model=args.model or "ernie-image-v1"
        )
        if success:
            print(f"Image saved to: {args.output}")
        else:
            print("Image generation failed")


if __name__ == "__main__":
    main()
