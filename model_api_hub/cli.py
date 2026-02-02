"""
Model API Hub - CLI Tool

Command-line interface for testing different API providers.
"""

import argparse
import sys
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Model API Hub - Unified interface for AI model APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available providers
  model-api-hub ls
  modelhub ls

  # Test DeepSeek LLM
  model-api-hub deepseek "Hello, how are you?"

  # Test with custom API key
  model-api-hub deepseek "Hello!" --api-key your_key_here

  # Test SiliconFlow VLM with image
  model-api-hub siliconflow-vlm "Describe this image" --image photo.jpg

  # Generate image
  model-api-hub siliconflow-image "A beautiful sunset" --output sunset.png

Available providers:
  LLM: deepseek, siliconflow, zhipuai, kimi, yiyan, minimax
  VLM: siliconflow-vlm, yiyan-vlm
  Image: siliconflow-image, recraft-image
  Audio: elevenlabs-tts, openai-tts
  Video: runway-video, luma-video
        """
    )

    parser.add_argument("provider", nargs="?", help="Provider to use (or 'ls' to list)")
    parser.add_argument("prompt", nargs="?", help="Prompt for the model")
    parser.add_argument("--api-key", type=str, help="API key (overrides env variable)")
    parser.add_argument("--model", type=str, help="Model identifier")
    parser.add_argument("--image", type=str, help="Image path for VLM")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--config", type=str, default=".env", help="Config file path (default: .env)")

    args = parser.parse_args()

    # Handle 'ls' command or no provider
    if args.provider == "ls" or args.provider is None:
        print("Available providers:")
        print("\n  LLM:")
        for p in ["deepseek", "siliconflow", "zhipuai", "kimi", "yiyan", "minimax"]:
            print(f"    - {p}")
        print("\n  VLM:")
        for p in ["siliconflow-vlm", "yiyan-vlm"]:
            print(f"    - {p}")
        print("\n  Image:")
        for p in ["siliconflow-image", "recraft-image"]:
            print(f"    - {p}")
        print("\n  Audio:")
        for p in ["elevenlabs-tts", "openai-tts"]:
            print(f"    - {p}")
        print("\n  Video:")
        for p in ["runway-video", "luma-video"]:
            print(f"    - {p}")
        print("\nUsage: model-api-hub <provider> <prompt> [options]")
        return

    # Handle provider calls
    call_provider(args)


def call_provider(args):
    """Call the appropriate provider based on args."""
    provider = args.provider
    prompt = args.prompt

    if not prompt:
        print(f"Error: Please provide a prompt for {provider}")
        sys.exit(1)

    # Import the appropriate module
    if provider == "deepseek":
        from model_api_hub.api.llm.deepseek_llm import chat
        response = chat(prompt, api_key=args.api_key)
        print(response)

    elif provider == "siliconflow":
        from model_api_hub.api.llm.siliconflow_llm import chat
        response = chat(prompt, api_key=args.api_key)
        print(response)

    elif provider == "zhipuai":
        from model_api_hub.api.llm.zhipuai_llm import chat
        response = chat(prompt, api_key=args.api_key)
        print(response)

    elif provider == "kimi":
        from model_api_hub.api.llm.kimi_llm import chat
        response = chat(prompt, api_key=args.api_key)
        print(response)

    elif provider == "yiyan":
        from model_api_hub.api.llm.yiyan_llm import chat
        response = chat(prompt, api_key=args.api_key)
        print(response)

    elif provider == "minimax":
        from model_api_hub.api.llm.minimax_llm import chat
        response = chat(prompt, api_key=args.api_key)
        print(response)

    elif provider == "siliconflow-vlm":
        from model_api_hub.api.vlm.siliconflow_vlm import analyze_image
        if not args.image:
            print("Error: VLM requires --image parameter")
            sys.exit(1)
        response = analyze_image(args.image, prompt, api_key=args.api_key)
        print(response)

    elif provider == "yiyan-vlm":
        from model_api_hub.api.vlm.yiyan_vlm import analyze_image
        if not args.image:
            print("Error: VLM requires --image parameter")
            sys.exit(1)
        response = analyze_image(args.image, prompt, api_key=args.api_key)
        print(response)

    elif provider == "siliconflow-image":
        from model_api_hub.api.image.siliconflow_image_gen import text_to_image
        output = args.output or "generated_image.png"
        success = text_to_image(prompt, output, api_key=args.api_key)
        if success:
            print(f"Image saved to: {output}")
        else:
            print("Image generation failed")
            sys.exit(1)

    elif provider == "recraft-image":
        from model_api_hub.api.image.recraft_image_gen import text_to_image
        output = args.output or "generated_image.png"
        success = text_to_image(prompt, output, api_key=args.api_key)
        if success:
            print(f"Image saved to: {output}")
        else:
            print("Image generation failed")
            sys.exit(1)

    elif provider == "elevenlabs-tts":
        from model_api_hub.api.audio.elevenlabs_tts import text_to_speech
        output = args.output or "output.mp3"
        success = text_to_speech(prompt, output, api_key=args.api_key)
        if success:
            print(f"Audio saved to: {output}")
        else:
            print("TTS conversion failed")
            sys.exit(1)

    elif provider == "openai-tts":
        from model_api_hub.api.audio.openai_tts import text_to_speech
        output = args.output or "output.mp3"
        success = text_to_speech(prompt, output, api_key=args.api_key)
        if success:
            print(f"Audio saved to: {output}")
        else:
            print("TTS conversion failed")
            sys.exit(1)

    elif provider == "runway-video":
        from model_api_hub.api.video.runway_gen import generate_video
        output = args.output or "generated_video.mp4"
        print("Starting video generation (this may take several minutes)...")
        success = generate_video(prompt, output, api_key=args.api_key)
        if success:
            print(f"Video saved to: {output}")
        else:
            print("Video generation failed")
            sys.exit(1)

    elif provider == "luma-video":
        from model_api_hub.api.video.luma_gen import generate_video
        output = args.output or "generated_video.mp4"
        print("Starting video generation (this may take several minutes)...")
        success = generate_video(prompt, output, api_key=args.api_key)
        if success:
            print(f"Video saved to: {output}")
        else:
            print("Video generation failed")
            sys.exit(1)

    else:
        print(f"Unknown provider: {provider}")
        print("Use 'model-api-hub ls' to see available providers")
        sys.exit(1)


if __name__ == "__main__":
    main()
