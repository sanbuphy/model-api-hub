"""
ModelScope (魔搭社区) API client - 阿里开源模型平台

Website: https://www.modelscope.cn/
API Docs: https://www.modelscope.cn/docs/model-service/API-Inference/intro
API Key: https://www.modelscope.cn/my/mynotebook/authorization
Features: 开源模型托管、推理API、模型微调
Pricing: 每日免费 2000 次调用（单个模型每日上限 500 次）

Available Models:
- Qwen series: qwen-max, qwen-plus, qwen-turbo, qwen2.5-72b, qwen2.5-32b, qwen2.5-14b, qwen2.5-7b
- Llama series: llama3.1-70b, llama3.1-8b, llama3-70b, llama3-8b
- DeepSeek: deepseek-v3, deepseek-r1
- GLM: glm-4, glm-4-plus
- Other: yi-large, baichuan2-13b, chatglm3-6b

Quick Start:
1. 访问 https://www.modelscope.cn/my/mynotebook/authorization
2. 获取 SDK Token 作为 API Key
3. 参考文档: https://www.modelscope.cn/docs/model-service/API-Inference/intro
"""

import os
from typing import Optional, List, Dict, Any
import requests

# API Configuration
API_KEY: str = os.getenv("MODELSCOPE_API_KEY", "")
BASE_URL: str = os.getenv("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1")
DEFAULT_MODEL: str = "qwen-max"

# Available models
AVAILABLE_MODELS: List[str] = [
    # Qwen series
    "qwen-max",
    "qwen-plus", 
    "qwen-turbo",
    "qwen2.5-72b-instruct",
    "qwen2.5-32b-instruct",
    "qwen2.5-14b-instruct",
    "qwen2.5-7b-instruct",
    "qwen2-72b-instruct",
    "qwen2-7b-instruct",
    "qwen-vl-max",
    "qwen-vl-plus",
    # Llama series
    "llama3.1-70b-instruct",
    "llama3.1-8b-instruct",
    "llama3-70b-instruct",
    "llama3-8b-instruct",
    # DeepSeek
    "deepseek-v3",
    "deepseek-r1",
    # GLM
    "glm-4",
    "glm-4-plus",
    "glm-4-flash",
    # Others
    "yi-large",
    "baichuan2-13b-chat",
    "chatglm3-6b",
]


def create_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    """Create request headers."""
    key = api_key or API_KEY
    if not key:
        raise ValueError("ModelScope API key is required. Set MODELSCOPE_API_KEY environment variable.")
    
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def chat(
    message: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
    **kwargs
) -> str:
    """
    Chat using ModelScope LLM.
    
    Args:
        message: User message
        model: Model identifier (e.g., qwen-max, llama3.1-70b-instruct, deepseek-v3)
        api_key: API key (optional)
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        system_prompt: System prompt
        **kwargs: Additional parameters
    
    Returns:
        Response text
    """
    headers = create_headers(api_key)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": message})
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        **kwargs
    }
    
    if max_tokens:
        payload["max_tokens"] = max_tokens
    
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    
    result = response.json()
    return result["choices"][0]["message"]["content"]


def stream_chat(
    message: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
    **kwargs
):
    """
    Stream chat using ModelScope LLM.
    
    Args:
        message: User message
        model: Model identifier
        api_key: API key (optional)
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        system_prompt: System prompt
        **kwargs: Additional parameters
    
    Yields:
        Response text chunks
    """
    headers = create_headers(api_key)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": message})
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": True,
        **kwargs
    }
    
    if max_tokens:
        payload["max_tokens"] = max_tokens
    
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        stream=True,
        timeout=60
    )
    response.raise_for_status()
    
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = line[6:]
                if data == '[DONE]':
                    break
                import json
                try:
                    chunk = json.loads(data)
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        delta = chunk['choices'][0].get('delta', {})
                        if 'content' in delta:
                            yield delta['content']
                except json.JSONDecodeError:
                    pass


def analyze_image(
    image_path: str,
    prompt: str = "描述这张图片",
    model: str = "qwen-vl-max",
    api_key: Optional[str] = None,
    **kwargs
) -> str:
    """
    Analyze image using ModelScope VLM (Qwen-VL).
    
    Args:
        image_path: Path to image file
        prompt: Question about the image
        model: VLM model (qwen-vl-max, qwen-vl-plus)
        api_key: API key (optional)
        **kwargs: Additional parameters
    
    Returns:
        Response text
    """
    import base64
    
    headers = create_headers(api_key)
    
    # Read and encode image
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            {"type": "text", "text": prompt}
        ]
    }]
    
    payload = {
        "model": model,
        "messages": messages,
        **kwargs
    }
    
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    
    result = response.json()
    return result["choices"][0]["message"]["content"]


def get_embedding(
    text: str,
    model: str = "BAAI/bge-large-zh-v1.5",
    api_key: Optional[str] = None,
    **kwargs
) -> List[float]:
    """
    Get text embedding using ModelScope.
    
    Args:
        text: Input text
        model: Embedding model
        api_key: API key (optional)
        **kwargs: Additional parameters
    
    Returns:
        Embedding vector
    """
    headers = create_headers(api_key)
    
    payload = {
        "model": model,
        "input": text,
        **kwargs
    }
    
    response = requests.post(
        f"{BASE_URL}/embeddings",
        headers=headers,
        json=payload,
        timeout=60
    )
    response.raise_for_status()
    
    result = response.json()
    return result["data"][0]["embedding"]


def main():
    """Demo usage."""
    import argparse
    parser = argparse.ArgumentParser(description="ModelScope API Demo")
    parser.add_argument("--mode", choices=["chat", "image", "embedding"], default="chat", help="API mode")
    parser.add_argument("--message", default="Hello!", help="Input message")
    parser.add_argument("--image", help="Path to image file (for image mode)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to use")
    parser.add_argument("--api-key", help="ModelScope API key")
    parser.add_argument("--stream", action="store_true", help="Stream response")
    args = parser.parse_args()
    
    if args.mode == "chat":
        print(f"Chat with ModelScope model: {args.model}")
        print(f"Message: {args.message}")
        print("-" * 50)
        
        if args.stream:
            for chunk in stream_chat(args.message, model=args.model, api_key=args.api_key):
                print(chunk, end="", flush=True)
            print()
        else:
            response = chat(args.message, model=args.model, api_key=args.api_key)
            print("Response:")
            print(response)
    
    elif args.mode == "image":
        if not args.image:
            print("Error: --image is required for image mode")
            return
        
        print(f"Analyze image with ModelScope VLM: {args.model}")
        print(f"Image: {args.image}")
        print(f"Prompt: {args.message}")
        print("-" * 50)
        
        response = analyze_image(args.image, args.message, model=args.model, api_key=args.api_key)
        print("Response:")
        print(response)
    
    elif args.mode == "embedding":
        print(f"Get embedding with ModelScope")
        print(f"Text: {args.message}")
        print("-" * 50)
        
        embedding = get_embedding(args.message, api_key=args.api_key)
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")


if __name__ == "__main__":
    main()
