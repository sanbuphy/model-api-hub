"""
Xunfei Spark (讯飞星火) Image Generation API wrapper.

Provides interface for Xunfei's image generation models:
- spark-image-v1 - 星火绘画 V1

Website: https://xinghuo.xfyun.cn/
"""

import os
import base64
import json
import hashlib
import hmac
import datetime
from typing import Optional
from urllib.parse import urlencode, urlparse
import websocket
import time

DEFAULT_MODEL: str = "spark-image-v1"
SPARK_URL: str = "wss://spark-api.xf-yun.com/v3.1/chat"

API_KEY: Optional[str] = None
API_SECRET: Optional[str] = None
APP_ID: Optional[str] = None


def generate_auth_url(api_key: str, api_secret: str, url: str = SPARK_URL) -> str:
    """Generate authenticated WebSocket URL."""
    parsed_url = urlparse(url)
    host = parsed_url.netloc
    path = parsed_url.path
    
    # Generate RFC1123 date
    date = datetime.datetime.utcnow().strftime('%a, %d %b %Y %H:%M:%S GMT')
    
    # Create signature
    signature_origin = f"host: {host}\ndate: {date}\nGET {path} HTTP/1.1"
    signature_sha = hmac.new(
        api_secret.encode('utf-8'),
        signature_origin.encode('utf-8'),
        digestmod=hashlib.sha256
    ).digest()
    signature = base64.b64encode(signature_sha).decode(encoding='utf-8')
    
    # Create authorization
    authorization_origin = f'api_key="{api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature}"'
    authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
    
    # Generate URL
    params = {
        "authorization": authorization,
        "date": date,
        "host": host
    }
    
    return f"{url}?{urlencode(params)}"


def text_to_image(
    prompt: str,
    output_path: str = "output.png",
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    app_id: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    width: int = 1024,
    height: int = 1024,
    **kwargs
) -> bool:
    """
    Generate image from text using Xunfei Spark Image.
    
    Args:
        prompt: Image generation prompt
        output_path: Output file path
        api_key: Xunfei API key (optional)
        api_secret: Xunfei API secret (optional)
        app_id: Xunfei App ID (optional)
        model: Model identifier
        width: Image width
        height: Image height
        **kwargs: Additional parameters
    
    Returns:
        True if successful, False otherwise
    """
    try:
        key = api_key or API_KEY or os.getenv("XUNFEI_API_KEY")
        secret = api_secret or API_SECRET or os.getenv("XUNFEI_API_SECRET")
        app = app_id or APP_ID or os.getenv("XUNFEI_APP_ID")
        
        if not all([key, secret, app]):
            raise ValueError("Xunfei API key, secret, and app_id are required.")
        
        # Generate auth URL
        auth_url = generate_auth_url(key, secret)
        
        # Prepare request for image generation
        request_data = {
            "header": {
                "app_id": app,
                "uid": "12345"
            },
            "parameter": {
                "chat": {
                    "domain": "generalv3.5",
                    "temperature": 0.5,
                    "max_tokens": 4096
                }
            },
            "payload": {
                "message": {
                    "text": [
                        {"role": "user", "content": f"请根据描述生成图片: {prompt}"}
                    ]
                }
            }
        }
        
        # WebSocket connection
        image_data = None
        
        def on_message(ws, message):
            nonlocal image_data
            data = json.loads(message)
            if "payload" in data and "choices" in data["payload"]:
                content = data["payload"]["choices"]["text"][0]["content"]
                # Check if content contains image data
                if content.startswith("data:image"):
                    image_data = content.split(",")[1]
        
        def on_error(ws, error):
            raise Exception(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            pass
        
        def on_open(ws):
            ws.send(json.dumps(request_data))
        
        ws = websocket.WebSocketApp(
            auth_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        ws.run_forever()
        
        # Save image
        if image_data:
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(image_data))
            return True
        else:
            print("No image data received")
            return False
        
    except Exception as e:
        print(f"Error generating image: {e}")
        return False


def main():
    """Demo usage."""
    import argparse
    parser = argparse.ArgumentParser(description="Xunfei Spark Image Generation Demo")
    parser.add_argument("--prompt", required=True, help="Image generation prompt")
    parser.add_argument("--output", default="output.png", help="Output file path")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--api-key", help="Xunfei API key")
    parser.add_argument("--api-secret", help="Xunfei API secret")
    parser.add_argument("--app-id", help="Xunfei App ID")
    args = parser.parse_args()
    
    if args.api_key:
        global API_KEY
        API_KEY = args.api_key
    if args.api_secret:
        global API_SECRET
        SECRET_KEY = args.api_secret
    if args.app_id:
        global APP_ID
        APP_ID = args.app_id
    
    success = text_to_image(
        args.prompt,
        args.output,
        api_key=args.api_key,
        api_secret=args.api_secret,
        app_id=args.app_id,
        width=args.width,
        height=args.height
    )
    
    if success:
        print(f"Image saved to: {args.output}")
    else:
        print("Image generation failed")


if __name__ == "__main__":
    main()
