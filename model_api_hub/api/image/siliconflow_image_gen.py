"""
SiliconFlow Image Generation API wrapper.

Provides interface for SiliconFlow's image generation models including:
- Kwai-Kolors/Kolors - Kuaishou's Kolors model
- black-forest-labs/FLUX.1-dev - FLUX.1 dev version
- black-forest-labs/FLUX.1-pro - FLUX.1 pro version
- black-forest-labs/FLUX.1-schnell - FLUX.1 fast version
- stabilityai/stable-diffusion-3-5b - Stable Diffusion 3 5B
- stabilityai/stable-diffusion-xl-base-1.0 - SDXL base model
"""

import requests
import os
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

# Default configuration
DEFAULT_BASE_URL: str = "https://api.siliconflow.cn/v1/images/generations"
DEFAULT_MODEL: str = "Kwai-Kolors/Kolors"
DEFAULT_IMAGE_SIZE: str = "1024x1024"


def generate_image(
    prompt: str,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    image_size: str = DEFAULT_IMAGE_SIZE,
    batch_size: int = 1,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    image: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Generate an image using SiliconFlow API.

    Args:
        prompt: Text description for image generation.
        api_key: SiliconFlow API key. If None, loads from environment.
        model: AI model identifier to use.
        image_size: Dimensions of output image (e.g., "1024x1024", "960x1280", "720x1440").
        batch_size: Number of images to generate (1-4).
        num_inference_steps: Number of denoising steps (1-100).
        guidance_scale: Strength of adherence to prompt (0-20).
        negative_prompt: Optional text describing what should NOT appear.
        seed: Optional random seed for reproducibility.
        image: Optional base64 encoded input image for img2img.

    Returns:
        API response dictionary with generated image URLs, or None if failed.
    """
    if api_key is None:
        api_key = get_api_key("siliconflow")

    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "image_size": image_size,
        "batch_size": batch_size,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale
    }

    if negative_prompt is not None:
        payload["negative_prompt"] = negative_prompt
    if seed is not None:
        payload["seed"] = seed
    if image is not None:
        payload["image"] = image

    headers: Dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(DEFAULT_BASE_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error generating image: {e}")
        return None


def save_image_from_url(image_url: str, output_path: str = "generated_image.png") -> bool:
    """
    Download and save an image from URL to local file.

    Args:
        image_url: URL of the image to download.
        output_path: Local path where to save the image.

    Returns:
        True if successful, False otherwise.
    """
    try:
        response = requests.get(image_url)
        response.raise_for_status()

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Image saved successfully to: {output_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return False
    except Exception as e:
        print(f"Error saving image: {e}")
        return False


def text_to_image(
    prompt: str,
    output_path: str = "generated_image.png",
    api_key: Optional[str] = None,
    **kwargs
) -> bool:
    """
    Generate an image from text prompt and save it.

    Args:
        prompt: Text description for image generation.
        output_path: Local path to save the generated image.
        api_key: Optional API key. If None, loads from environment.
        **kwargs: Additional parameters for generate_image.

    Returns:
        True if successful, False otherwise.
    """
    result = generate_image(prompt, api_key=api_key, **kwargs)

    if result and "data" in result:
        images = result["data"]
        if images and len(images) > 0:
            image_url = images[0]["url"]
            return save_image_from_url(image_url, output_path)
    return False


def image_to_image(
    prompt: str,
    input_image_path: str,
    output_path: str = "generated_image.png",
    api_key: Optional[str] = None,
    **kwargs
) -> bool:
    """
    Transform an image using text prompt.

    Args:
        prompt: Text description for image transformation.
        input_image_path: Path to the input image.
        output_path: Local path to save the transformed image.
        api_key: Optional API key. If None, loads from environment.
        **kwargs: Additional parameters for generate_image.

    Returns:
        True if successful, False otherwise.
    """
    import base64

    with open(input_image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode('utf-8')
    base64_image = f"data:image/png;base64,{base64_image}"

    result = generate_image(prompt, api_key=api_key, image=base64_image, **kwargs)

    if result and "data" in result:
        images = result["data"]
        if images and len(images) > 0:
            image_url = images[0]["url"]
            return save_image_from_url(image_url, output_path)
    return False


def main() -> None:
    """
    Main function demonstrating usage with config file support.
    """
    # Option 1: Direct API key
    api_key = None  # Set your API key directly or load from .env

    # Option 2: Use config manager
    config_mgr = ConfigManager()
    img_config = config_mgr.get_image_config("siliconflow")
    model = img_config.get("model", DEFAULT_MODEL)
    image_size = img_config.get("image_size", DEFAULT_IMAGE_SIZE)
    num_inference_steps = img_config.get("num_inference_steps", 20)
    guidance_scale = img_config.get("guidance_scale", 7.5)

    # Test text to image
    prompt = "a beautiful landscape with mountains and a lake, sunset, peaceful atmosphere, high quality"
    negative_prompt = "blurry, low quality, distorted, ugly"
    output_path = "generated_image.png"

    print(f"Generating image with prompt: {prompt}")
    print(f"Model: {model}")
    print(f"Image size: {image_size}")
    print(f"Inference steps: {num_inference_steps}")
    print(f"Guidance scale: {guidance_scale}")
    print("-" * 50)

    success = text_to_image(
        prompt=prompt,
        output_path=output_path,
        api_key=api_key,
        model=model,
        image_size=image_size,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        seed=12345
    )

    if success:
        print(f"Image generation completed successfully!")
    else:
        print("Image generation failed")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SiliconFlow Image Generation API Demo")
    parser.add_argument("--api-key", type=str, help="SiliconFlow API key")
    parser.add_argument("--prompt", type=str,
                        default="a beautiful landscape with mountains and a lake",
                        help="Text prompt for image generation")
    parser.add_argument("--output", type=str, default="generated_image.png",
                        help="Output file path")
    args = parser.parse_args()

    if args.prompt:
        text_to_image(args.prompt, args.output, api_key=args.api_key)
    else:
        main()
