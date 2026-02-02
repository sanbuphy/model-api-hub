"""
Tencent Hunyuan (腾讯混元) API client - 腾讯大模型平台

Website: https://hunyuan.tencent.com/
Features: 大语言模型、多模态、文生图、文生视频
Pricing: 免费额度 + 按量付费
"""

import os
from typing import Optional, List, Dict, Any
import requests
import time

# API Configuration
API_KEY: str = os.getenv("TENCENT_HUNYUAN_API_KEY", "")
SECRET_ID: str = os.getenv("TENCENT_SECRET_ID", "")
SECRET_KEY: str = os.getenv("TENCENT_SECRET_KEY", "")
BASE_URL: str = os.getenv("HUNYUAN_BASE_URL", "https://hunyuan.tencentcloudapi.com")
DEFAULT_LLM_MODEL: str = "hunyuan-pro"

# Available models
AVAILABLE_LLM_MODELS: List[str] = [
    "hunyuan-pro",
    "hunyuan-standard",
    "hunyuan-lite",
    "hunyuan-role",
]


def create_signature(secret_id: str, secret_key: str, payload: Dict[str, Any]) -> Dict[str, str]:
    """Create Tencent Cloud API signature."""
    import hashlib
    import hmac
    import json
    from datetime import datetime
    
    service = "hunyuan"
    host = "hunyuan.tencentcloudapi.com"
    region = "ap-guangzhou"
    action = "ChatCompletions"
    version = "2023-09-01"
    
    timestamp = int(datetime.utcnow().timestamp())
    date = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")
    
    # Create canonical request
    http_request_method = "POST"
    canonical_uri = "/"
    canonical_querystring = ""
    canonical_headers = f"content-type:application/json\nhost:{host}\nx-tc-action:{action.lower()}\n"
    signed_headers = "content-type;host;x-tc-action"
    
    payload_json = json.dumps(payload)
    hashed_request_payload = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    canonical_request = (http_request_method + "\n" +
                        canonical_uri + "\n" +
                        canonical_querystring + "\n" +
                        canonical_headers + "\n" +
                        signed_headers + "\n" +
                        hashed_request_payload)
    
    # Create string to sign
    algorithm = "TC3-HMAC-SHA256"
    credential_scope = date + "/" + service + "/" + "tc3_request"
    hashed_canonical_request = hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()
    string_to_sign = (algorithm + "\n" +
                     str(timestamp) + "\n" +
                     credential_scope + "\n" +
                     hashed_canonical_request)
    
    # Calculate signature
    secret_date = hmac.new(("TC3" + secret_key).encode("utf-8"), date.encode("utf-8"), hashlib.sha256).digest()
    secret_service = hmac.new(secret_date, service.encode("utf-8"), hashlib.sha256).digest()
    secret_signing = hmac.new(secret_service, "tc3_request".encode("utf-8"), hashlib.sha256).digest()
    signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
    
    # Create authorization
    authorization = (algorithm + " " +
                    "Credential=" + secret_id + "/" + credential_scope + ", " +
                    "SignedHeaders=" + signed_headers + ", " +
                    "Signature=" + signature)
    
    return {
        "Authorization": authorization,
        "Content-Type": "application/json",
        "Host": host,
        "X-TC-Action": action,
        "X-TC-Version": version,
        "X-TC-Timestamp": str(timestamp),
        "X-TC-Region": region,
    }


def chat(
    message: str,
    model: str = DEFAULT_LLM_MODEL,
    api_key: Optional[str] = None,
    secret_id: Optional[str] = None,
    secret_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
    **kwargs
) -> str:
    """
    Chat using Tencent Hunyuan.
    
    Args:
        message: User message
        model: Model identifier
        api_key: API key (optional)
        secret_id: Secret ID (optional)
        secret_key: Secret key (optional)
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        system_prompt: System prompt
        **kwargs: Additional parameters
    
    Returns:
        Response text
    """
    sid = secret_id or SECRET_ID or os.getenv("TENCENT_SECRET_ID")
    skey = secret_key or SECRET_KEY or os.getenv("TENCENT_SECRET_KEY")
    
    if not sid or not skey:
        raise ValueError("Tencent Secret ID and Secret Key are required.")
    
    messages = []
    if system_prompt:
        messages.append({"Role": "system", "Content": system_prompt})
    messages.append({"Role": "user", "Content": message})
    
    payload = {
        "Model": model,
        "Messages": messages,
        **kwargs
    }
    
    if temperature is not None:
        payload["Temperature"] = temperature
    if max_tokens is not None:
        payload["MaxTokens"] = max_tokens
    
    headers = create_signature(sid, skey, payload)
    
    response = requests.post(
        f"{BASE_URL}",
        headers=headers,
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    
    result = response.json()
    
    if "Response" in result and "Choices" in result["Response"]:
        return result["Response"]["Choices"][0]["Message"]["Content"]
    else:
        raise Exception(f"Unexpected response: {result}")


def analyze_image(
    image_path: str,
    prompt: str = "描述这张图片",
    api_key: Optional[str] = None,
    secret_id: Optional[str] = None,
    secret_key: Optional[str] = None,
    model: str = "hunyuan-vision",
    **kwargs
) -> str:
    """
    Analyze image using Hunyuan VLM.
    
    Args:
        image_path: Path to image file
        prompt: Text prompt for image analysis
        api_key: API key (optional)
        secret_id: Secret ID (optional)
        secret_key: Secret key (optional)
        model: Model identifier
        **kwargs: Additional parameters
    
    Returns:
        Response text
    """
    import base64
    
    sid = secret_id or SECRET_ID or os.getenv("TENCENT_SECRET_ID")
    skey = secret_key or SECRET_KEY or os.getenv("TENCENT_SECRET_KEY")
    
    if not sid or not skey:
        raise ValueError("Tencent Secret ID and Secret Key are required.")
    
    # Encode image
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    messages = [
        {
            "Role": "user",
            "Content": [
                {"Type": "image_url", "ImageUrl": {"Url": f"data:image/jpeg;base64,{base64_image}"}},
                {"Type": "text", "Text": prompt}
            ]
        }
    ]
    
    payload = {
        "Model": model,
        "Messages": messages,
        **kwargs
    }
    
    headers = create_signature(sid, skey, payload)
    
    response = requests.post(
        f"{BASE_URL}",
        headers=headers,
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    
    result = response.json()
    
    if "Response" in result and "Choices" in result["Response"]:
        return result["Response"]["Choices"][0]["Message"]["Content"]
    else:
        raise Exception(f"Unexpected response: {result}")


def text_to_image(
    prompt: str,
    output_path: str = "output.png",
    api_key: Optional[str] = None,
    secret_id: Optional[str] = None,
    secret_key: Optional[str] = None,
    model: str = "hunyuan-image",
    width: int = 1024,
    height: int = 1024,
    **kwargs
) -> bool:
    """
    Generate image from text using Hunyuan.
    
    Args:
        prompt: Image generation prompt
        output_path: Output file path
        api_key: API key (optional)
        secret_id: Secret ID (optional)
        secret_key: Secret key (optional)
        model: Model identifier
        width: Image width
        height: Image height
        **kwargs: Additional parameters
    
    Returns:
        True if successful, False otherwise
    """
    try:
        sid = secret_id or SECRET_ID or os.getenv("TENCENT_SECRET_ID")
        skey = secret_key or SECRET_KEY or os.getenv("TENCENT_SECRET_KEY")
        
        if not sid or not skey:
            raise ValueError("Tencent Secret ID and Secret Key are required.")
        
        payload = {
            "Model": model,
            "Prompt": prompt,
            "Width": width,
            "Height": height,
            **kwargs
        }
        
        headers = create_signature(sid, skey, payload)
        
        response = requests.post(
            f"{BASE_URL}",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        
        if "Response" in result and "ImageUrl" in result["Response"]:
            image_url = result["Response"]["ImageUrl"]
            
            img_response = requests.get(image_url, timeout=30)
            img_response.raise_for_status()
            
            with open(output_path, "wb") as f:
                f.write(img_response.content)
            return True
        else:
            print(f"Error: No image URL in response - {result}")
            return False
            
    except Exception as e:
        print(f"Error generating image: {e}")
        return False


def main():
    """Demo usage."""
    import argparse
    parser = argparse.ArgumentParser(description="Tencent Hunyuan Demo")
    parser.add_argument("--mode", choices=["chat", "vlm", "image"], default="chat", help="API mode")
    parser.add_argument("--input", required=True, help="Input text or image path")
    parser.add_argument("--prompt", help="Prompt for VLM mode")
    parser.add_argument("--output", default="output.png", help="Output file path")
    parser.add_argument("--secret-id", help="Tencent Secret ID")
    parser.add_argument("--secret-key", help="Tencent Secret Key")
    parser.add_argument("--model", help="Model to use")
    args = parser.parse_args()
    
    if args.mode == "chat":
        response = chat(
            args.input,
            secret_id=args.secret_id,
            secret_key=args.secret_key,
            model=args.model or DEFAULT_LLM_MODEL
        )
        print("Response:")
        print(response)
    elif args.mode == "vlm":
        response = analyze_image(
            args.input,
            prompt=args.prompt or "描述这张图片",
            secret_id=args.secret_id,
            secret_key=args.secret_key,
            model=args.model or "hunyuan-vision"
        )
        print("Response:")
        print(response)
    else:  # image
        success = text_to_image(
            args.input,
            args.output,
            secret_id=args.secret_id,
            secret_key=args.secret_key,
            model=args.model or "hunyuan-image"
        )
        if success:
            print(f"Image saved to: {args.output}")
        else:
            print("Image generation failed")


if __name__ == "__main__":
    main()
