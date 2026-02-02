"""
Xunfei Spark (iFlytek) LLM API wrapper.

Provides interface for Xunfei Spark models:
- generalv3.5 - Spark V3.5
- generalv3 - Spark V3
- generalv2 - Spark V2
- 4.0Ultra - Spark 4.0 Ultra
"""

import websocket
import json
import base64
import hmac
from datetime import datetime
from urllib.parse import urlencode
import hashlib
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

DEFAULT_DOMAIN: str = "generalv3.5"
SPARK_URL: str = "wss://spark-api.xf-yun.com/v3.5/chat"

API_KEY: Optional[str] = None
APP_ID: Optional[str] = None
API_SECRET: Optional[str] = None


def create_auth_url(api_key: str, api_secret: str, spark_url: str = SPARK_URL) -> str:
    """Create authenticated WebSocket URL."""
    url = spark_url
    now = datetime.utcnow()
    date = now.strftime('%a, %d %b %Y %H:%M:%S GMT')
    
    signature_origin = f"host: spark-api.xf-yun.com\ndate: {date}\nGET /v3.5/chat HTTP/1.1"
    signature_sha = hmac.new(
        api_secret.encode('utf-8'),
        signature_origin.encode('utf-8'),
        digestmod=hashlib.sha256
    ).digest()
    signature_sha_base64 = base64.b64encode(signature_sha).decode()
    
    authorization_origin = f'api_key="{api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'
    authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode()
    
    params = {
        "authorization": authorization,
        "date": date,
        "host": "spark-api.xf-yun.com"
    }
    
    return f"{url}?{urlencode(params)}"


def chat(
    prompt: str,
    system_prompt: Optional[str] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    app_id: Optional[str] = None,
    domain: str = DEFAULT_DOMAIN,
    **kwargs
) -> str:
    """Quick chat function using WebSocket."""
    if api_key is None:
        api_key = get_api_key("xunfei_spark")
    if api_secret is None:
        # Try to get from config
        config_mgr = ConfigManager()
        api_secret = config_mgr.get_api_key("xunfei_spark_api_secret")
    if app_id is None:
        config_mgr = ConfigManager()
        app_id = config_mgr.get_api_key("xunfei_spark_app_id")
    
    ws_url = create_auth_url(api_key, api_secret)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    data = {
        "header": {"app_id": app_id},
        "parameter": {"chat": {"domain": domain}},
        "payload": {"message": {"text": messages}}
    }
    
    # Simple HTTP-based implementation for demo
    import requests
    response = requests.post(
        "https://spark-api-open.xf-yun.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": domain,
            "messages": messages
        }
    )
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Error: {response.text}")


def main() -> None:
    """Demo usage."""
    prompt = "你好！你能做什么？"
    
    print(f"Sending request to Xunfei Spark model: {DEFAULT_DOMAIN}")
    response = chat(prompt)
    print("Response:")
    print(response)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Xunfei Spark LLM API Demo")
    parser.add_argument("--api-key", type=str, help="Xunfei API key")
    parser.add_argument("--api-secret", type=str, help="Xunfei API secret")
    parser.add_argument("--app-id", type=str, help="Xunfei App ID")
    args = parser.parse_args()
    
    if args.api_key:
        API_KEY = args.api_key
    if args.api_secret:
        API_SECRET = args.api_secret
    if args.app_id:
        APP_ID = args.app_id
    main()
