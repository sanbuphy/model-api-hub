<div align="center">

<img src="assets/logo.png" alt="Model API Hub Logo" width="120" height="120">

# Model API Hub

**One Line of Code, Access to 100+ AI Models**

[![GitHub release](https://img.shields.io/github/v/release/username/translamate)](https://github.com/username/translamate/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/username/translamate/ci.yml)](https://github.com/username/translamate/actions)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/model-api-hub.svg)](https://badge.fury.io/py/model-api-hub)

[English](README.md) · [简体中文](README.zh-CN.md)

</div>

---

## What is Model API Hub?

Model API Hub is a **unified Python SDK** that lets you access multiple AI model APIs across different platforms and modalities with a **consistent, simple interface**.

Stop juggling different SDKs for each provider. Use one library for everything.

```python
# Same interface, different providers
from model_api_hub import deepseek_chat, siliconflow_chat, kimi_chat

# All work the same way
response = deepseek_chat("Hello!")
response = siliconflow_chat("Hello!")
response = kimi_chat("Hello!")
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **18+ LLM Providers** | OpenAI, Anthropic, DeepSeek, ZhipuAI, Kimi, SiliconFlow, and more |
| **Streaming Support** | Real-time streaming responses for all major providers |
| **5 Modalities** | LLM, Vision-Language, Image Gen, Audio TTS, Video Gen |
| **One-Line Setup** | `pip install model-api-hub` and you're ready |
| **Unified API** | Same interface across all providers |
| **Flexible Config** | `.env`, YAML, or direct API keys |
| **CLI Included** | Test models directly from command line |
| **Type Hints** | Full type safety support |

---

## Quick Start

### Installation

```bash
pip install model-api-hub
```

### 1. Set Your API Key

```bash
# Create .env file
echo 'DEEPSEEK_API_KEY=your_key_here' > .env
```

### 2. Start Coding

```python
from model_api_hub import deepseek_chat

# That's it. You're done.
response = deepseek_chat("Explain quantum computing in simple terms")
print(response)
```

---

## Usage Examples

### Language Models (LLM)

#### Synchronous Chat

```python
from model_api_hub import deepseek_chat, kimi_chat, siliconflow_chat

# DeepSeek
response = deepseek_chat(
    "Write a Python function to sort a list",
    system_prompt="You are a coding expert."
)

# Kimi (Moonshot)
response = kimi_chat(
    "Summarize this article",
    temperature=0.5
)

# SiliconFlow - access 50+ models
response = siliconflow_chat("Hello!", model="deepseek-ai/DeepSeek-V3")
```

#### Streaming Chat

```python
from model_api_hub import deepseek_chat_stream

# Stream responses in real-time
for chunk in deepseek_chat_stream("Tell me a long story"):
    print(chunk, end="", flush=True)
```

#### Multi-turn Conversation

```python
from model_api_hub.api.llm.deepseek_llm import create_client, get_completion

client = create_client()
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language..."},
    {"role": "user", "content": "What are its main features?"}
]

response = get_completion(client, messages)
```

### Vision-Language Models (VLM)

```python
from model_api_hub.api.vlm.openai_vlm import chat

response = chat(
    prompt="What's in this image?",
    image_path="photo.jpg"
)
```

### Image Generation

```python
from model_api_hub.api.image.siliconflow_image_gen import generate

image_url = generate("A beautiful sunset over mountains")
```

### Text-to-Speech

```python
from model_api_hub.api.audio.openai_tts import synthesize

audio = synthesize("Hello, world!", voice="alloy", output_path="hello.mp3")
```

---

## Supported Models

### Language Models (LLM)

<table>
  <tr>
    <td valign="top" width="33%">
      • <a href="./support_model.md#deepseek-chat">DeepSeek-Chat</a><br>
      • <a href="./support_model.md#deepseek-r1-distill">DeepSeek-R1</a><br>
      • <a href="./support_model.md#deepseek-coder-v2">DeepSeek-Coder-V2</a><br>
      • <a href="./support_model.md#deepseek-深度求索">DeepSeek-V3</a><br>
      • <a href="./support_model.md#gpt-4o">GPT-4o</a><br>
      • <a href="./support_model.md#gpt-4o-mini">GPT-4o-mini</a><br>
      • <a href="./support_model.md#gpt-4-turbo">GPT-4-Turbo</a><br>
      • <a href="./support_model.md#claude-sonnet-4-5">Claude-Sonnet-4.5</a><br>
      • <a href="./support_model.md#claude-opus-4">Claude-Opus-4</a><br>
      • <a href="./support_model.md#gemini-pro">Gemini-Pro</a><br>
      • <a href="./support_model.md#gemini-flash">Gemini-Flash</a><br>
      • <a href="./support_model.md#glm-47-flash">GLM-4.7-Flash</a><br>
      • <a href="./support_model.md#glm-4">GLM-4</a><br>
      • <a href="./support_model.md#glm-4-plus">GLM-4-Plus</a><br>
      • <a href="./support_model.md#glm-45-air">GLM-4.5-Air</a><br>
      • <a href="./support_model.md#glm-41-thinking">GLM-4.1-Thinking</a><br>
      • <a href="./support_model.md#moonshot-v1-128k">Moonshot-v1-128k</a><br>
      • <a href="./support_model.md#moonshot-v1-32k">Moonshot-v1-32k</a>
    </td>
    <td valign="top" width="33%">
      • <a href="./support_model.md#kimi">Kimi-K2</a><br>
      • <a href="./support_model.md#abab65s-chat">MiniMax-ABAB6.5s</a><br>
      • <a href="./support_model.md#minimax-m2">MiniMax-M2</a><br>
      • <a href="./support_model.md#ernie-4">ERNIE-4.0</a><br>
      • <a href="./support_model.md#ernie-45">ERNIE-4.5</a><br>
      • <a href="./support_model.md#qwen-max">Qwen-Max</a><br>
      • <a href="./support_model.md#qwen-plus">Qwen-Plus</a><br>
      • <a href="./support_model.md#qwen-turbo">Qwen-Turbo</a><br>
      • <a href="./support_model.md#qwen25">Qwen2.5</a><br>
      • <a href="./support_model.md#qwen25-coder">Qwen2.5-Coder</a><br>
      • <a href="./support_model.md#qwen3">Qwen3</a><br>
      • <a href="./support_model.md#qwen2">Qwen2</a><br>
      • <a href="./support_model.md#qwen-15">Qwen 1.5</a><br>
      • <a href="./support_model.md#llama4">Llama4</a><br>
      • <a href="./support_model.md#llama31-8b-instruct">Llama3.1</a><br>
      • <a href="./support_model.md#llama3-70b">Llama3-70B</a><br>
      • <a href="./support_model.md#gemma3">Gemma3</a><br>
      • <a href="./support_model.md#gemma-2-9b-it">Gemma-2</a>
    </td>
    <td valign="top" width="33%">
      • <a href="./support_model.md#mistral-large">Mistral-Large</a><br>
      • <a href="./support_model.md#mixtral-8x22b">Mixtral-8x22B</a><br>
      • <a href="./support_model.md#command-r-plus">Command-R-Plus</a><br>
      • <a href="./support_model.md#internlm3">InternLM3</a><br>
      • <a href="./support_model.md#internlm">InternLM</a><br>
      • <a href="./support_model.md#internlm2-20b">InternLM2-20B</a><br>
      • <a href="./support_model.md#baichuan-百川智能">Baichuan</a><br>
      • <a href="./support_model.md#yi-零一万物">Yi 零一万物</a><br>
      • <a href="./support_model.md#yuan20">Yuan2.0</a><br>
      • <a href="./support_model.md#yuan20-m32">Yuan2.0-M32</a><br>
      • <a href="./support_model.md#hunyuan-a13b-instruct">Hunyuan-A13B</a><br>
      • <a href="./support_model.md#hunyuan3d-2">Hunyuan3D-2</a><br>
      • <a href="./support_model.md#spark-v35">Spark-v3.5</a><br>
      • <a href="./support_model.md#phi4">Phi4</a><br>
      • <a href="./support_model.md#phi-3">Phi-3</a><br>
      • <a href="./support_model.md#minicpm">MiniCPM</a><br>
      • <a href="./support_model.md#characterglm-6b">CharacterGLM</a>
    </td>
  </tr>
</table>

### Vision-Language Models (VLM)

<table>
  <tr>
    <td valign="top" width="33%">
      • <a href="./support_model.md#gpt-4o-vision">GPT-4o-Vision</a><br>
      • <a href="./support_model.md#gpt-4v">GPT-4V</a><br>
      • <a href="./support_model.md#gemini-pro-vision">Gemini-Pro-Vision</a><br>
      • <a href="./support_model.md#qwen3-vl-4b-instruct">Qwen3-VL</a><br>
      • <a href="./support_model.md#qwen2-vl">Qwen2-VL</a><br>
      • <a href="./support_model.md#qwen-vl-plus">Qwen-VL-Plus</a>
    </td>
    <td valign="top" width="33%">
      • <a href="./support_model.md#glm-4v">GLM-4V</a><br>
      • <a href="./support_model.md#minicpm-o-2_6">MiniCPM-o</a><br>
      • <a href="./support_model.md#yi-vl">Yi-VL</a><br>
      • <a href="./support_model.md#internvl">InternVL</a><br>
      • <a href="./support_model.md#deepseek-vl">DeepSeek-VL</a>
    </td>
    <td valign="top" width="33%">
      • <a href="./support_model.md#spatiallm">SpatialLM</a><br>
      • <a href="./support_model.md#llava">LLaVA</a><br>
      • <a href="./support_model.md#cogvlm">CogVLM</a><br>
      • <a href="./support_model.md#bluelm-vivo-蓝心大模型">BlueLM-Vision</a>
    </td>
  </tr>
</table>

### Image Generation Models

<table>
  <tr>
    <td valign="top" width="33%">
      • <a href="./support_model.md#dall-e-3">DALL-E 3</a><br>
      • <a href="./support_model.md#dall-e-2">DALL-E 2</a><br>
      • <a href="./support_model.md#kolors">Kolors</a><br>
      • <a href="./support_model.md#stable-diffusion-xl">Stable Diffusion XL</a><br>
      • <a href="./support_model.md#stable-diffusion-3">Stable Diffusion 3</a>
    </td>
    <td valign="top" width="33%">
      • <a href="./support_model.md#recraft-v3">Recraft-v3</a><br>
      • <a href="./support_model.md#wanx">Wanx</a><br>
      • <a href="./support_model.md#ernie-vilg">ERNIE-ViLG</a><br>
      • <a href="./support_model.md#jimeng">Jimeng (Dreamina)</a><br>
      • <a href="./support_model.md#cogview">CogView</a>
    </td>
    <td valign="top" width="33%">
      • <a href="./support_model.md#hunyuan-image">Hunyuan-Image</a><br>
      • <a href="./support_model.md#playground-v2">Playground-v2</a><br>
      • <a href="./support_model.md#kandinsky">Kandinsky</a><br>
      • <a href="./support_model.md#deepfloyd">DeepFloyd IF</a>
    </td>
  </tr>
</table>

### Audio Models

<table>
  <tr>
    <td valign="top" width="33%">
      • <a href="./support_model.md#whisper">Whisper</a><br>
      • <a href="./support_model.md#whisper-large-v3">Whisper-Large-v3</a><br>
      • <a href="./support_model.md#tts-1">TTS-1</a><br>
      • <a href="./support_model.md#tts-1-hd">TTS-1-HD</a><br>
      • <a href="./support_model.md#elevenlabs-multilingual-v2">ElevenLabs-Multilingual-v2</a>
    </td>
    <td valign="top" width="33%">
      • <a href="./support_model.md#elevenlabs-flash">ElevenLabs-Flash</a><br>
      • <a href="./support_model.md#azure-tts">Azure-TTS</a><br>
      • <a href="./support_model.md#azure-speech">Azure-Speech</a><br>
      • <a href="./support_model.md#minimax-tts">MiniMax-TTS</a><br>
      • <a href="./support_model.md#baidu-tts">Baidu-TTS</a>
    </td>
    <td valign="top" width="33%">
      • <a href="./support_model.md#qwen-audio">Qwen-Audio</a><br>
      • <a href="./support_model.md#chattts">ChatTTS</a><br>
      • <a href="./support_model.md#fish-speech">Fish-Speech</a><br>
      • <a href="./support_model.md# GPT-SoVITS">GPT-SoVITS</a>
    </td>
  </tr>
</table>

### Video Generation Models

<table>
  <tr>
    <td valign="top" width="33%">
      • <a href="./support_model.md#runway-gen3">Runway-Gen3</a><br>
      • <a href="./support_model.md#runway-gen2">Runway-Gen2</a><br>
      • <a href="./support_model.md#luma-dream-machine">Luma-Dream-Machine</a><br>
      • <a href="./support_model.md#luma-genie">Luma-Genie</a>
    </td>
    <td valign="top" width="33%">
      • <a href="./support_model.md#pika">Pika</a><br>
      • <a href="./support_model.md#stable-video-diffusion">Stable-Video-Diffusion</a><br>
      • <a href="./support_model.md#jimeng-video">Jimeng-Video</a><br>
      • <a href="./support_model.md#cogvideo">CogVideo</a>
    </td>
    <td valign="top" width="33%">
      • <a href="./support_model.md#videocrafter">VideoCrafter</a><br>
      • <a href="./support_model.md#modelscope-video">ModelScope-Video</a><br>
      • <a href="./support_model.md#animatediff">AnimateDiff</a>
    </td>
  </tr>
</table>

---

## Configuration

### Environment Variables (.env)

Create a `.env` file in your project root:

```bash
# LLM Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
KIMI_API_KEY=sk-...
ZHIPUAI_API_KEY=...
SILICONFLOW_API_KEY=sk-...
MINIMAX_API_KEY=...
YIYAN_API_KEY=...
DASHSCOPE_API_KEY=sk-...
MODELSCOPE_API_KEY=ms-...
XUNFEI_SPARK_API_KEY=...
GROQ_API_KEY=gsk_...
TOGETHER_API_KEY=...
MISTRAL_API_KEY=...
COHERE_API_KEY=...
PERPLEXITY_API_KEY=pplx-...
AZURE_OPENAI_API_KEY=...

# Other Services
ELEVENLABS_API_KEY=...
AZURE_SPEECH_KEY=...
STABILITY_API_KEY=...
RECRAFT_API_KEY=...
RUNWAY_API_KEY=...
LUMA_API_KEY=...
```

### YAML Configuration

Create `config.yaml`:

```yaml
llm:
  openai:
    model: "gpt-4o"
    temperature: 0.7
    max_tokens: 4096
  
  deepseek:
    model: "deepseek-chat"
    temperature: 0.7
    max_tokens: 4096

vlm:
  openai:
    model: "gpt-4o"
    
image:
  siliconflow:
    model: "Kwai-Kolors/Kolors"
    size: "1024x1024"
```

---

## Documentation

- [LLM Usage Guide](docs/llm_usage.md) - Complete LLM documentation
- [API Reference](docs/api_reference.md) - Full API reference
- [llm.txt](llm.txt) - Quick reference for AI assistants

---

## Testing

Run tests for all providers:

```bash
# Test all LLMs (sync)
python tests/test_llm.py

# Test streaming
python tests/test_llm_streaming.py

# Test other modalities
python tests/test_vlm.py
python tests/test_image.py
python tests/test_audio.py
python tests/test_video.py
```

---

## CLI Usage

```bash
# Chat with a provider
model-api-hub chat deepseek "Hello!"

# List available providers
model-api-hub list

# Test a provider
model-api-hub test deepseek
```

---

## Architecture

```
model_api_hub/
├── api/
│   ├── llm/           # Language Models (18+ providers)
│   ├── vlm/           # Vision-Language Models
│   ├── image/         # Image Generation
│   ├── audio/         # Text-to-Speech
│   └── video/         # Video Generation
├── utils/
│   └── config.py      # Configuration management
├── cli.py             # Command-line interface
└── __init__.py        # Public API exports
```

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Adding a New Provider

1. Create a new file in `model_api_hub/api/llm/{provider}_llm.py`
2. Implement `chat()`, `chat_stream()` (optional), and `create_client()`
3. Add exports to `model_api_hub/api/llm/__init__.py`
4. Add tests in `tests/test_llm.py`
5. Update documentation

See [llm.txt](llm.txt) for detailed implementation guide.

---

## License

MIT License - see [LICENSE](LICENSE) file.

---

## Support

- 📖 [Documentation](docs/)
- 🐛 [Issue Tracker](https://github.com/username/model-api-hub/issues)
- 💬 [Discussions](https://github.com/username/model-api-hub/discussions)

---

## Acknowledgments

Thanks to all the AI providers for their amazing APIs!
