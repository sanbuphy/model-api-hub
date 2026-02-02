"""
ModelScope (魔搭社区) API client - 阿里开源模型平台

Website: https://www.modelscope.cn/
Features: 开源模型托管、推理API、模型微调
Pricing: 免费额度 + 按量付费
"""

import os
from typing import Optional, List, Dict, Any
import requests

# API Configuration
API_KEY: str = os.getenv("MODELSCOPE_API_KEY", "")
BASE_URL: str = os.getenv("MODELSCOPE_BASE_URL", "https://api.modelscope.cn/v1")
DEFAULT_MODEL: str = "damo/nlp_structbert_sentence-similarity_chinese-base"

# Available models
AVAILABLE_MODELS: List[str] = [
    # LLM
    "damo/nlp_chatgpt_alpaca_chinese",
    "damo/nlp_polylm_13b_text_generation",
    # Embedding
    "damo/nlp_structbert_sentence-similarity_chinese-base",
    "damo/nlp_corom_sentence-embedding_chinese-base",
    # CV
    "damo/cv_resnet50_image-classification",
    # Audio
    "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
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
    model: str = "damo/nlp_chatgpt_alpaca_chinese",
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    **kwargs
) -> str:
    """
    Chat using ModelScope LLM.
    
    Args:
        message: User message
        model: Model identifier
        api_key: API key (optional)
        temperature: Sampling temperature
        max_tokens: Maximum tokens
        **kwargs: Additional parameters
    
    Returns:
        Response text
    """
    headers = create_headers(api_key)
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
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


def text_to_image(
    prompt: str,
    output_path: str = "output.png",
    api_key: Optional[str] = None,
    model: str = "damo/cv_diffusion_text-to-image-synthesis",
    **kwargs
) -> bool:
    """
    Generate image from text using ModelScope.
    
    Args:
        prompt: Image generation prompt
        output_path: Output file path
        api_key: API key (optional)
        model: Model identifier
        **kwargs: Additional parameters
    
    Returns:
        True if successful, False otherwise
    """
    try:
        headers = create_headers(api_key)
        
        payload = {
            "model": model,
            "prompt": prompt,
            **kwargs
        }
        
        response = requests.post(
            f"{BASE_URL}/images/generations",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        
        if "data" in result and len(result["data"]) > 0:
            image_url = result["data"][0]["url"]
            
            img_response = requests.get(image_url, timeout=30)
            img_response.raise_for_status()
            
            with open(output_path, "wb") as f:
                f.write(img_response.content)
            return True
        else:
            print(f"Error: No image data in response - {result}")
            return False
            
    except Exception as e:
        print(f"Error generating image: {e}")
        return False


def get_embedding(
    text: str,
    model: str = "damo/nlp_structbert_sentence-similarity_chinese-base",
    api_key: Optional[str] = None,
    **kwargs
) -> List[float]:
    """
    Get text embedding using ModelScope.
    
    Args:
        text: Input text
        model: Model identifier
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
        timeout=30
    )
    response.raise_for_status()
    
    result = response.json()
    return result["data"][0]["embedding"]


def main():
    """Demo usage."""
    import argparse
    parser = argparse.ArgumentParser(description="ModelScope API Demo")
    parser.add_argument("--mode", choices=["chat", "image", "embedding"], default="chat", help="API mode")
    parser.add_argument("--input", required=True, help="Input text")
    parser.add_argument("--output", default="output.png", help="Output file path (for image mode)")
    parser.add_argument("--api-key", help="ModelScope API key")
    parser.add_argument("--model", help="Model to use")
    args = parser.parse_args()
    
    if args.mode == "chat":
        response = chat(
            args.input,
            api_key=args.api_key,
            model=args.model or "damo/nlp_chatgpt_alpaca_chinese"
        )
        print("Response:")
        print(response)
    elif args.mode == "image":
        success = text_to_image(
            args.input,
            args.output,
            api_key=args.api_key,
            model=args.model or "damo/cv_diffusion_text-to-image-synthesis"
        )
        if success:
            print(f"Image saved to: {args.output}")
        else:
            print("Image generation failed")
    else:  # embedding
        embedding = get_embedding(
            args.input,
            api_key=args.api_key,
            model=args.model or "damo/nlp_structbert_sentence-similarity_chinese-base"
        )
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")


if __name__ == "__main__":
    main()
