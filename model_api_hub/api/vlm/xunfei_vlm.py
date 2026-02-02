"""
Xunfei Spark (讯飞星火) Vision API wrapper.

Provides interface for Xunfei's vision-language models:
- generalv3.5-vision - 星火视觉理解模型

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

DEFAULT_MODEL: str = "generalv3.5-vision"
SPARK_URL: str = "wss://spark-api.xf-yun.com/v3.5/chat"

API_KEY: Optional[str] = None
API_SECRET: Optional[str] = None
APP_ID: Optional[str] = None


def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


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


def analyze_image(
    image_path: str,
    prompt: str = "描述这张图片",
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    app_id: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    **kwargs
) -> str:
    """
    Analyze image using Xunfei Spark VLM.
    
    Args:
        image_path: Path to image file
        prompt: Text prompt for image analysis
        api_key: Xunfei API key (optional)
        api_secret: Xunfei API secret (optional)
        app_id: Xunfei App ID (optional)
        model: Model identifier
        **kwargs: Additional parameters
    
    Returns:
        Response text
    """
    key = api_key or API_KEY or os.getenv("XUNFEI_API_KEY")
    secret = api_secret or API_SECRET or os.getenv("XUNFEI_API_SECRET")
    app = app_id or APP_ID or os.getenv("XUNFEI_APP_ID")
    
    if not all([key, secret, app]):
        raise ValueError("Xunfei API key, secret, and app_id are required.")
    
    # Encode image
    base64_image = encode_image(image_path)
    
    # Generate auth URL
    auth_url = generate_auth_url(key, secret)
    
    # Prepare request
    request_data = {
        "header": {
            "app_id": app,
            "uid": "12345"
        },
        "parameter": {
            "chat": {
                "domain": model,
                "temperature": 0.5,
                "max_tokens": 4096
            }
        },
        "payload": {
            "message": {
                "text": [
                    {"role": "user", "content": prompt, "content_type": "text"},
                    {"role": "user", "content": base64_image, "content_type": "image"}
                ]
            }
        }
    }
    
    # WebSocket connection
    response_text = ""
    
    def on_message(ws, message):
        nonlocal response_text
        data = json.loads(message)
        if "payload" in data and "choices" in data["payload"]:
            content = data["payload"]["choices"]["text"][0]["content"]
            response_text += content
    
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
    
    return response_text


def main():
    """Demo usage."""
    import argparse
    parser = argparse.ArgumentParser(description="Xunfei Spark VLM Demo")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--prompt", default="描述这张图片", help="Prompt for image analysis")
    parser.add_argument("--api-key", help="Xunfei API key")
    parser.add_argument("--api-secret", help="Xunfei API secret")
    parser.add_argument("--app-id", help="Xunfei App ID")
    args = parser.parse_args()
    
    if args.api_key:
        global API_KEY
        API_KEY = args.api_key
    if args.api_secret:
        global API_SECRET
        API_SECRET = args.api_secret
    if args.app_id:
        global APP_ID
        APP_ID = args.app_id
    
    response = analyze_image(
        args.image,
        args.prompt,
        api_key=args.api_key,
        api_secret=args.api_secret,
        app_id=args.app_id
    )
    print("Response:")
    print(response)


if __name__ == "__main__":
    main()
