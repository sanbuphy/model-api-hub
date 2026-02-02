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

## Supported Providers

### LLM Providers (18+)

| Provider | Import | Default Model | Streaming |
|----------|--------|---------------|-----------|
| **OpenAI** | `openai_chat` | gpt-4o | ✅ |
| **Anthropic** | `anthropic_chat` | claude-sonnet-4-5 | ✅ |
| **DeepSeek** | `deepseek_chat` | deepseek-chat | ✅ |
| **Google Gemini** | `gemini_chat` | gemini-pro | ✅ |
| **Kimi** | `kimi_chat` | moonshot-v1-128k | ✅ |
| **ZhipuAI** | `zhipuai_chat` | glm-4-plus | ✅ |
| **SiliconFlow** | `siliconflow_chat` | DeepSeek-V3 | ✅ |
| **MiniMax** | `minimax_chat` | abab6.5s-chat | ✅ |
| **Baidu Yiyan** | `yiyan_chat` | ernie-4.0-8k | ✅ |
| **Alibaba DashScope** | `dashscope_chat` | qwen-max | ✅ |
| **ModelScope** | `modelscope_chat` | Qwen2.5-72B | ✅ |
| **Xunfei Spark** | `xunfei_chat` | Spark-v3.5 | ✅ |
| **Groq** | `groq_chat` | llama3-70b | ✅ |
| **Together AI** | `together_chat` | Llama-3-70b | ✅ |
| **Mistral** | `mistral_chat` | mistral-large | ✅ |
| **Cohere** | `cohere_chat` | command-r-plus | ✅ |
| **Perplexity** | `perplexity_chat` | sonar-pro | ✅ |
| **Azure OpenAI** | `azure_chat` | gpt-4o | ✅ |

### Other Modalities

- **VLM**: OpenAI, Gemini, Qwen-VL, GLM-4V
- **Image**: SiliconFlow, Stability, Recraft, Baidu
- **Audio**: OpenAI, ElevenLabs, Azure, Minimax
- **Video**: Runway, Luma, Dreamina

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
