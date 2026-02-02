"""
Baidu ERNIE Image (百度文心一格) Image Generation API wrapper.

Provides interface for Baidu's image generation models:
- ernie-image-v1 - 文心一格 V1

Website: https://yige.baidu.com/
"""

import os
import requests
import json
from typing import Optional
import time

DEFAULT_MODEL: str = "ernie-image-v1"
TOKEN_URL: str = "https://aip.baidubce.com/oauth/2.0/token"
GENERATE_URL: str = "https://aip.baidubce.com/rpc/2.0/ernievilg/v1/txt2img"
QUERY_URL: str = "https://aip.baidubce.com/rpc/2.0/ernievilg/v1/getImg"

API_KEY: Optional[str] = None
SECRET_KEY: Optional[str] = None


def get_access_token(api_key: Optional[str] = None, secret_key: Optional[str] = None) -> str:
    """Get Baidu access token."""
    key = api_key or API_KEY or os.getenv("BAIDU_API_KEY")
    secret = secret_key or SECRET_KEY or os.getenv("BAIDU_SECRET_KEY")
    
    if not key or not secret:
        raise ValueError("Baidu API key and secret key are required.")
    
    params = {
        "grant_type": "client_credentials",
        "client_id": key,
        "client_secret": secret
    }
    
    response = requests.post(TOKEN_URL, params=params)
    return response.json()["access_token"]


def text_to_image(
    prompt: str,
    output_path: str = "output.png",
    api_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    width: int = 1024,
    height: int = 1024,
    **kwargs
) -> bool:
    """
    Generate image from text using Baidu ERNIE Image.
    
    Args:
        prompt: Image generation prompt
        output_path: Output file path
        api_key: Baidu API key (optional)
        secret_key: Baidu secret key (optional)
        model: Model identifier
        width: Image width
        height: Image height
        **kwargs: Additional parameters
    
    Returns:
        True if successful, False otherwise
    """
    try:
        access_token = get_access_token(api_key, secret_key)
        
        # Submit generation task
        headers = {"Content-Type": "application/json"}
        payload = {
            "text": prompt,
            "width": width,
            "height": height,
            "version": "v2"
        }
        
        response = requests.post(
            f"{GENERATE_URL}?access_token={access_token}",
            headers=headers,
            json=payload
        )
        
        result = response.json()
        if "taskId" not in result:
            print(f"Error: {result}")
            return False
        
        task_id = result["taskId"]
        
        # Poll for result
        max_retries = 30
        for _ in range(max_retries):
            time.sleep(2)
            
            query_payload = {"taskId": task_id}
            query_response = requests.post(
                f"{QUERY_URL}?access_token={access_token}",
                headers=headers,
                json=query_payload
            )
            
            query_result = query_response.json()
            
            if query_result.get("status") == "SUCCESS":
                image_url = query_result["data"][0]["image"]
                img_response = requests.get(image_url)
                
                with open(output_path, "wb") as f:
                    f.write(img_response.content)
                return True
            elif query_result.get("status") == "FAILED":
                print(f"Generation failed: {query_result}")
                return False
        
        print("Timeout waiting for image generation")
        return False
        
    except Exception as e:
        print(f"Error generating image: {e}")
        return False


def main():
    """Demo usage."""
    import argparse
    parser = argparse.ArgumentParser(description="Baidu ERNIE Image Generation Demo")
    parser.add_argument("--prompt", required=True, help="Image generation prompt")
    parser.add_argument("--output", default="output.png", help="Output file path")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--api-key", help="Baidu API key")
    parser.add_argument("--secret-key", help="Baidu secret key")
    args = parser.parse_args()
    
    if args.api_key:
        global API_KEY
        API_KEY = args.api_key
    if args.secret_key:
        global SECRET_KEY
        SECRET_KEY = args.secret_key
    
    success = text_to_image(
        args.prompt,
        args.output,
        api_key=args.api_key,
        secret_key=args.secret_key,
        width=args.width,
        height=args.height
    )
    
    if success:
        print(f"Image saved to: {args.output}")
    else:
        print("Image generation failed")


if __name__ == "__main__":
    main()
