"""
Google Gemini LLM API wrapper.

Provides interface for Google's Gemini language models:
- gemini-1.5-pro - Gemini 1.5 Pro
- gemini-1.5-flash - Gemini 1.5 Flash
- gemini-1.0-pro - Gemini 1.0 Pro
"""

import google.generativeai as genai
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from model_api_hub.utils.config import get_api_key, ConfigManager

DEFAULT_MODEL: str = "gemini-1.5-pro"

API_KEY: Optional[str] = None


def create_client(api_key: Optional[str] = None) -> genai.GenerativeModel:
    """Create Gemini client."""
    if api_key is None:
        api_key = get_api_key("gemini")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(DEFAULT_MODEL)


def get_completion(
    model: genai.GenerativeModel,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    **kwargs
) -> str:
    """Request completion from Gemini LLM."""
    if system_prompt:
        prompt = f"{system_prompt}\n\n{prompt}"
    
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
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
    model_name: str = DEFAULT_MODEL,
    **kwargs
) -> str:
    """Quick chat function."""
    if api_key is None:
        api_key = get_api_key("gemini")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    return get_completion(model, prompt, system_prompt=system_prompt, **kwargs)


def main() -> None:
    """Demo usage."""
    config_mgr = ConfigManager()
    llm_config = config_mgr.get_llm_config("gemini")
    model_name = llm_config.get("model", DEFAULT_MODEL)
    
    model = create_client()
    prompt = "Hello! What can you do?"
    
    print(f"Sending request to Gemini model: {model_name}")
    response = get_completion(model, prompt)
    print("Response:")
    print(response)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Gemini LLM API Demo")
    parser.add_argument("--api-key", type=str, help="Gemini API key")
    args = parser.parse_args()
    
    if args.api_key:
        API_KEY = args.api_key
    main()
