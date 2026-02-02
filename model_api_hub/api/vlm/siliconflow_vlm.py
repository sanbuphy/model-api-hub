"""
SiliconFlow VLM API wrapper.

Provides interface for SiliconFlow's Vision Language Models including:
- Qwen/Qwen3-VL-8B-Instruct - Lightweight multimodal model (8B parameters, 32K context)
- Qwen/Qwen2.5-VL-32B-Instruct - Enhanced reasoning (32B parameters, 128K context)
- Qwen/Qwen2.5-VL-72B-Instruct - Larger model with video understanding (72B parameters, 128K context)
- Qwen/QVQ-72B-Preview - Visual reasoning research model (72B parameters, 32K context)
- zai-org/GLM-4.5V - VLM with MoE architecture (106B parameters, 64K context)
- stepfun-ai/step3 - Multimodal reasoning model (321B parameters, 64K context)
- deepseek-ai/deepseek-vl2 - DeepSeek's multimodal model
"""

from openai import OpenAI
from typing import Dict, Any, List, Optional
import base64
import os
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

# Default configuration
DEFAULT_BASE_URL: str = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL: str = "Qwen/Qwen3-VL-8B-Instruct"


def create_client(api_key: Optional[str] = None, base_url: str = DEFAULT_BASE_URL) -> OpenAI:
    """
    Create OpenAI-compatible client for SiliconFlow VLM API.

    Args:
        api_key: SiliconFlow API key. If None, loads from environment.
        base_url: Base URL for SiliconFlow API.

    Returns:
        Configured OpenAI client instance.
    """
    if api_key is None:
        api_key = get_api_key("siliconflow")
    return OpenAI(api_key=api_key, base_url=base_url)


def encode_image(image_path: str) -> str:
    """
    Encode an image file to base64 string.

    Args:
        image_path: Path to the image file.

    Returns:
        Base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_completion(
    client: OpenAI,
    messages: List[Dict[str, Any]],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.7,
    stream: bool = False,
    **kwargs
) -> str:
    """
    Request a completion from SiliconFlow VLM.

    Args:
        client: OpenAI client instance.
        messages: List of message dictionaries with 'role' and 'content'.
        model: Model name to use.
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature (0-1).
        top_p: Nucleus sampling threshold (0-1).
        stream: Whether to stream the response.
        **kwargs: Additional parameters to pass to the API.

    Returns:
        The content of the model's response.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=stream,
        **kwargs
    )
    return response.choices[0].message.content


def analyze_image(
    image_path: str,
    prompt: str = "Please describe this image in detail.",
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    **kwargs
) -> str:
    """
    Analyze a single image with text prompt.

    Args:
        image_path: Path to the image file.
        prompt: Text prompt describing what to do with the image.
        api_key: Optional API key. If None, loads from environment.
        model: Model name to use.
        **kwargs: Additional parameters for get_completion.

    Returns:
        Model response text.
    """
    base64_image = encode_image(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]

    client = create_client(api_key=api_key)
    return get_completion(client, messages, model=model, **kwargs)


def analyze_multiple_images(
    image_paths: List[str],
    prompt: str = "Please describe these images.",
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    **kwargs
) -> str:
    """
    Analyze multiple images with text prompt.

    Args:
        image_paths: List of paths to image files.
        prompt: Text prompt describing what to do with the images.
        api_key: Optional API key. If None, loads from environment.
        model: Model name to use.
        **kwargs: Additional parameters for get_completion.

    Returns:
        Model response text.
    """
    content = [{"type": "text", "text": prompt}]

    for image_path in image_paths:
        base64_image = encode_image(image_path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

    messages = [{"role": "user", "content": content}]

    client = create_client(api_key=api_key)
    return get_completion(client, messages, model=model, **kwargs)


def main() -> None:
    """
    Main function demonstrating usage with config file support.
    """
    # Option 1: Direct API key
    api_key = None  # Set your API key directly or load from .env

    # Option 2: Use config manager
    config_mgr = ConfigManager()
    vlm_config = config_mgr.get_vlm_config("siliconflow")
    model = vlm_config.get("model", DEFAULT_MODEL)
    max_tokens = vlm_config.get("max_tokens", 4096)
    temperature = vlm_config.get("temperature", 0.7)

    # Test with single image
    image_path = "test_image.jpg"

    if os.path.exists(image_path):
        print(f"Analyzing image: {image_path}")
        print(f"Using model: {model}")
        print("-" * 50)

        response = analyze_image(
            image_path,
            prompt="Please describe this image in detail.",
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )

        print("Response:")
        print(response)
    else:
        print(f"Image file not found: {image_path}")
        print("Please provide a valid image path to test the VLM API.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SiliconFlow VLM API Demo")
    parser.add_argument("--api-key", type=str, help="SiliconFlow API key")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--prompt", type=str, default="Please describe this image in detail.",
                        help="Text prompt for the image")
    args = parser.parse_args()

    if args.image:
        client = create_client(api_key=args.api_key)
        response = analyze_image(args.image, args.prompt, api_key=args.api_key)
        print("Response:")
        print(response)
    else:
        main()
