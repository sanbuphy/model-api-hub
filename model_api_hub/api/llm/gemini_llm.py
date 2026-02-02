"""
Google Gemini LLM API wrapper.

Provides interface for Google's Gemini language models:
- gemini-1.5-pro - Gemini 1.5 Pro
- gemini-1.5-flash - Gemini 1.5 Flash
- gemini-1.0-pro - Gemini 1.0 Pro

Migration: Updated to use google.genai (new SDK) instead of deprecated google.generativeai
"""

from google import genai
from google.genai import types
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

DEFAULT_MODEL: str = "gemini-1.5-pro"

API_KEY: Optional[str] = None


def create_client(api_key: Optional[str] = None) -> genai.Client:
    """Create Gemini client."""
    if api_key is None:
        api_key = get_api_key("gemini")
    return genai.Client(api_key=api_key)


def get_completion(
    client: genai.Client,
    prompt: str,
    model: str = DEFAULT_MODEL,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    **kwargs
) -> str:
    """Request completion from Gemini LLM."""
    contents = prompt
    if system_prompt:
        contents = f"{system_prompt}\n\n{prompt}"
    
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            **kwargs
        )
    )
    return response.text


def chat(
    prompt: str,
    system_prompt: Optional[str] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    **kwargs
) -> str:
    """Quick chat function."""
    client = create_client(api_key)
    return get_completion(client, prompt, model=model, system_prompt=system_prompt, **kwargs)


def main(api_key: Optional[str] = None) -> None:
    """Demo usage."""
    try:
        config_mgr = ConfigManager()
        llm_config = config_mgr.get_llm_config("gemini")
    except Exception:
        llm_config = {}
    
    model_name = llm_config.get("model", DEFAULT_MODEL)
    
    client = create_client(api_key)
    prompt = "Hello! What can you do?"
    
    print(f"Sending request to Gemini model: {model_name}")
    response = get_completion(client, prompt, model=model_name)
    print("Response:")
    print(response)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gemini LLM API Demo")
    parser.add_argument("--api-key", type=str, help="Gemini API key")
    args = parser.parse_args()
    
    main(api_key=args.api_key)
