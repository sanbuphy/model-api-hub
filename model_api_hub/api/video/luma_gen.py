"""
Luma AI Video Generation API wrapper.

Provides interface for Luma's Dream Machine video generation.
"""

import requests
import os
import time
from typing import Optional, Dict, Any
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

# Default configuration
DEFAULT_BASE_URL: str = "https://api.lumalabs.ai/dream-machine/v1"
DEFAULT_MODEL: str = "genie-1.0"


def create_video_generation(
    prompt: str,
    api_key: Optional[str] = None,
    image_url: Optional[str] = None,
    loop: bool = False
) -> Optional[str]:
    """
    Create a video generation task.

    Args:
        prompt: Text description for video generation.
        api_key: Luma API key. If None, loads from environment.
        image_url: Optional image URL for image-to-video generation.
        loop: Whether to generate looping video.

    Returns:
        Task ID string if successful, None otherwise.
    """
    if api_key is None:
        api_key = get_api_key("luma")

    url = f"{DEFAULT_BASE_URL}/generations"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "prompt": prompt,
        "loop": loop
    }

    if image_url is not None:
        payload["image_url"] = image_url

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        return result.get("id")
    except requests.exceptions.RequestException as e:
        print(f"Error creating video generation: {e}")
        return None


def get_task_status(
    task_id: str,
    api_key: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Get status of a video generation task.

    Args:
        task_id: Video generation task ID.
        api_key: Luma API key. If None, loads from environment.

    Returns:
        Task status dictionary, or None if failed.
    """
    if api_key is None:
        api_key = get_api_key("luma")

    url = f"{DEFAULT_BASE_URL}/generations/{task_id}"

    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting task status: {e}")
        return None


def wait_for_completion(
    task_id: str,
    api_key: Optional[str] = None,
    check_interval: int = 10,
    timeout: int = 600
) -> Optional[Dict[str, Any]]:
    """
    Wait for video generation task to complete.

    Args:
        task_id: Video generation task ID.
        api_key: Luma API key. If None, loads from environment.
        check_interval: Seconds between status checks.
        timeout: Maximum seconds to wait.

    Returns:
        Final task status dictionary, or None if timeout/error.
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        status = get_task_status(task_id, api_key)

        if status is None:
            return None

        task_status = status.get("state", "")

        print(f"Status: {task_status}")

        if task_state in ["completed", "failed"]:
            return status

        time.sleep(check_interval)

    print("Timeout waiting for video generation")
    return None


def download_video(
    video_url: str,
    output_path: str = "generated_video.mp4"
) -> bool:
    """
    Download generated video from URL.

    Args:
        video_url: URL of the generated video.
        output_path: Local path to save the video.

    Returns:
        True if successful, False otherwise.
    """
    try:
        response = requests.get(video_url, stream=True)
        response.raise_for_status()

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Video saved successfully to: {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading video: {e}")
        return False


def generate_video(
    prompt: str,
    output_path: str = "generated_video.mp4",
    api_key: Optional[str] = None,
    **kwargs
) -> bool:
    """
    Generate a video from text prompt.

    Args:
        prompt: Text description for video generation.
        output_path: Local path to save the video.
        api_key: Optional API key. If None, loads from environment.
        **kwargs: Additional parameters for create_video_generation.

    Returns:
        True if successful, False otherwise.
    """
    print(f"Creating video generation task...")
    task_id = create_video_generation(prompt, api_key=api_key, **kwargs)

    if not task_id:
        print("Failed to create video generation task")
        return False

    print(f"Task ID: {task_id}")
    print("Waiting for video generation to complete...")

    result = wait_for_completion(task_id, api_key=api_key)

    if result and result.get("state") == "completed":
        video_url = result.get("video", {}).get("url")
        if video_url:
            return download_video(video_url, output_path)

    print("Video generation failed")
    return False


def main() -> None:
    """
    Main function demonstrating usage with config file support.
    """
    # Option 1: Direct API key
    api_key = None  # Set your API key directly or load from .env

    # Test video generation
    prompt = "A peaceful sunset over a calm lake, with mountains in the background, cinematic quality"
    output_path = "generated_video.mp4"

    print(f"Generating video with prompt: {prompt}")
    print("-" * 50)

    success = generate_video(
        prompt=prompt,
        output_path=output_path,
        api_key=api_key
    )

    if success:
        print(f"Video generation completed successfully!")
    else:
        print("Video generation failed")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Luma Video Generation API Demo")
    parser.add_argument("--api-key", type=str, help="Luma API key")
    parser.add_argument("--prompt", type=str,
                        default="A peaceful sunset over a calm lake",
                        help="Text prompt for video generation")
    parser.add_argument("--output", type=str, default="generated_video.mp4",
                        help="Output video file path")
    args = parser.parse_args()

    generate_video(args.prompt, args.output, api_key=args.api_key)
