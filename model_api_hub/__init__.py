"""
Model API Hub - Unified Python interface for multiple AI model APIs

Quick start:
    # Import and use directly with API key
    from model_api_hub.api.llm.deepseek_llm import chat
    response = chat("Hello!", api_key="your_key")

    # Or use config file
    from model_api_hub.utils.config import ConfigManager
    config = ConfigManager()
    api_key = config.get_api_key("deepseek")
"""

__version__ = "0.1.0"

# LLM - Synchronous
from model_api_hub.api.llm.deepseek_llm import chat as deepseek_chat
from model_api_hub.api.llm.siliconflow_llm import chat as siliconflow_chat
from model_api_hub.api.llm.zhipuai_llm import chat as zhipuai_chat
from model_api_hub.api.llm.kimi_llm import chat as kimi_chat
from model_api_hub.api.llm.yiyan_llm import chat as yiyan_chat
from model_api_hub.api.llm.minimax_llm import chat as minimax_chat
from model_api_hub.api.llm.openai_llm import chat as openai_chat
from model_api_hub.api.llm.gemini_llm import chat as gemini_chat
from model_api_hub.api.llm.dashscope_llm import chat as dashscope_chat
from model_api_hub.api.llm.xunfei_llm import chat as xunfei_chat
from model_api_hub.api.llm.groq_llm import chat as groq_chat
from model_api_hub.api.llm.together_llm import chat as together_chat
from model_api_hub.api.llm.cohere_llm import chat as cohere_chat
from model_api_hub.api.llm.mistral_llm import chat as mistral_chat
from model_api_hub.api.llm.modelscope_llm import chat as modelscope_chat
from model_api_hub.api.llm.stepfun_llm import chat as stepfun_chat

# LLM - Streaming
from model_api_hub.api.llm.deepseek_llm import chat_stream as deepseek_chat_stream
from model_api_hub.api.llm.openai_llm import chat_stream as openai_chat_stream

# VLM
from model_api_hub.api.vlm.siliconflow_vlm import analyze_image as siliconflow_analyze_image
from model_api_hub.api.vlm.yiyan_vlm import analyze_image as yiyan_analyze_image

# New VLM providers
from model_api_hub.api.vlm.openai_vlm import analyze_image as openai_analyze_image
from model_api_hub.api.vlm.gemini_vlm import analyze_image as gemini_analyze_image
from model_api_hub.api.vlm.dashscope_vlm import analyze_image as dashscope_analyze_image
from model_api_hub.api.vlm.modelscope_vlm import analyze_image as modelscope_analyze_image

# Image
from model_api_hub.api.image.siliconflow_image_gen import text_to_image as siliconflow_text_to_image
from model_api_hub.api.image.recraft_image_gen import text_to_image as recraft_text_to_image

# New Image providers
from model_api_hub.api.image.openai_image_gen import text_to_image as openai_text_to_image
from model_api_hub.api.image.stability_image_gen import text_to_image as stability_text_to_image
from model_api_hub.api.image.dashscope_image_gen import text_to_image as dashscope_text_to_image
from model_api_hub.api.image.dreamina_gen import text_to_image as dreamina_text_to_image

# Audio
from model_api_hub.api.audio.elevenlabs_tts import text_to_speech as elevenlabs_tts
from model_api_hub.api.audio.openai_tts import text_to_speech as openai_tts

# New Audio providers
from model_api_hub.api.audio.azure_tts import text_to_speech as azure_tts
from model_api_hub.api.audio.baidu_tts import text_to_speech as baidu_tts

# STT
from model_api_hub.api.stt.openai_whisper import speech_to_text as whisper_stt
from model_api_hub.api.stt.openai_whisper import translate_to_english as whisper_translate

# Video
from model_api_hub.api.video.runway_gen import generate_video as runway_generate_video
from model_api_hub.api.video.luma_gen import generate_video as luma_generate_video
from model_api_hub.api.video.dreamina_gen import text_to_video as dreamina_text_to_video

# Config
from model_api_hub.utils.config import ConfigManager, get_api_key, load_config

__all__ = [
    # Version
    "__version__",

    # LLM - Synchronous
    "deepseek_chat",
    "siliconflow_chat",
    "zhipuai_chat",
    "kimi_chat",
    "yiyan_chat",
    "minimax_chat",
    "openai_chat",
    "gemini_chat",
    "dashscope_chat",
    "xunfei_chat",
    "groq_chat",
    "together_chat",
    "cohere_chat",
    "mistral_chat",
    "modelscope_chat",
    "stepfun_chat",

    # LLM - Streaming
    "deepseek_chat_stream",
    "openai_chat_stream",

    # VLM
    "siliconflow_analyze_image",
    "yiyan_analyze_image",
    "openai_analyze_image",
    "gemini_analyze_image",
    "dashscope_analyze_image",
    "modelscope_analyze_image",

    # Image
    "siliconflow_text_to_image",
    "recraft_text_to_image",
    "openai_text_to_image",
    "stability_text_to_image",
    "dashscope_text_to_image",
    "dreamina_text_to_image",

    # Audio
    "elevenlabs_tts",
    "openai_tts",
    "azure_tts",
    "baidu_tts",

    # STT
    "whisper_stt",
    "whisper_translate",

    # Video
    "runway_generate_video",
    "luma_generate_video",
    "dreamina_text_to_video",

    # Config
    "ConfigManager",
    "get_api_key",
    "load_config",
]
