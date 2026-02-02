"""
ZhipuAI (智谱 AI) GLM-4V Vision API wrapper.

Provides interface for GLM-4V vision-language models:
- glm-4v - GLM-4V 基础版
- glm-4v-plus - GLM-4V 增强版
- glm-4v-flash - GLM-4V 轻量版

Website: https://open.bigmodel.cn/
"""

import os
from typing import Optional
import base64
from openai import OpenAI

DEFAULT_MODEL: str = "glm-4v"
BASE_URL: str = "https://open.bigmodel.cn/api/paas/v4/"

API_KEY: Optional[str] = None


def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def create_client(api_key: Optional[str] = None) -> OpenAI:
    """Create ZhipuAI OpenAI-compatible client."""
    key = api_key or API_KEY or os.getenv("ZHIPUAI_API_KEY")
    if not key:
        raise ValueError("ZhipuAI API key is required. Set ZHIPUAI_API_KEY environment variable.")
    
    return OpenAI(
        api_key=key,
        base_url=BASE_URL,
    )


def analyze_image(
    image_path: str,
    prompt: str = "描述这张图片",
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: Optional[int] = 4096,
    **kwargs
) -> str:
    """
    Analyze image using GLM-4V.
    
    Args:
        image_path: Path to image file
        prompt: Text prompt for image analysis
        api_key: ZhipuAI API key (optional)
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        **kwargs: Additional parameters
    
    Returns:
        Response text
    """
    client = create_client(api_key)
    
    # Encode image
    base64_image = encode_image(image_path)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )
    
    return response.choices[0].message.content


def main():
    """Demo usage."""
    import argparse
    parser = argparse.ArgumentParser(description="ZhipuAI GLM-4V Demo")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--prompt", default="描述这张图片", help="Prompt for image analysis")
    parser.add_argument("--api-key", help="ZhipuAI API key")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to use")
    args = parser.parse_args()
    
    if args.api_key:
        global API_KEY
        API_KEY = args.api_key
    
    response = analyze_image(
        args.image, 
        args.prompt, 
        api_key=args.api_key,
        model=args.model
    )
    print("Response:")
    print(response)


if __name__ == "__main__":
    main()
