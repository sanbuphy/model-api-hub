"""
ModelScope (魔搭社区) VLM API wrapper - 阿里开源模型平台

Website: https://www.modelscope.cn/
API Docs: https://www.modelscope.cn/docs/model-service/API-Inference/intro
API Key: https://www.modelscope.cn/my/mynotebook/authorization
Pricing: 每日免费 2000 次调用（单个模型每日上限 500 次）

Available Vision Models:
- qwen-vl-max - Qwen Vision-Language Max
- qwen-vl-plus - Qwen Vision-Language Plus
- qwen2-vl - Qwen2 Vision-Language

Quick Start:
1. 访问 https://www.modelscope.cn/my/mynotebook/authorization
2. 获取 SDK Token 作为 API Key
3. 参考文档: https://www.modelscope.cn/docs/model-service/API-Inference/intro
"""

from openai import OpenAI
from typing import Optional
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

DEFAULT_MODEL: str = "qwen-vl-max"

API_KEY: Optional[str] = None


def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_image(
    image_path: str,
    prompt: str = "这是什么?",
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    **kwargs
) -> str:
    """
    Analyze image using ModelScope VLM.

    Args:
        image_path: Path to the image file
        prompt: Question or prompt about the image
        api_key: ModelScope API key (optional)
        model: Model identifier
        **kwargs: Additional parameters

    Returns:
        Model's response text
    """
    if api_key is None:
        api_key = get_api_key("modelscope")
    
    # Create client
    client = OpenAI(
        api_key=api_key,
        base_url="https://api-inference.modelscope.cn/v1"
    )
    
    # Encode image
    base64_image = encode_image(image_path)
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            {"type": "text", "text": prompt}
        ]
    }]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs
    )
    
    return response.choices[0].message.content


def main():
    """Demo usage."""
    import argparse
    parser = argparse.ArgumentParser(description="ModelScope VLM Demo")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--prompt", default="这是什么?", help="Prompt for image analysis")
    parser.add_argument("--api-key", help="ModelScope API key")
    args = parser.parse_args()
    
    if args.api_key:
        global API_KEY
        API_KEY = args.api_key
    
    response = analyze_image(args.image, args.prompt, api_key=args.api_key)
    print("Response:")
    print(response)


if __name__ == "__main__":
    main()
