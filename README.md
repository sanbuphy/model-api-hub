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

## ✨ What is Model API Hub?

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

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| 🔌 **15+ Providers** | OpenAI, Anthropic, DeepSeek, ZhipuAI, Kimi, SiliconFlow, and more |
| 🎨 **5 Modalities** | LLM, Vision-Language, Image Gen, Audio TTS, Video Gen |
| 🚀 **One-Line Setup** | `pip install model-api-hub` and you're ready |
| 🔄 **Unified API** | Same interface across all providers |
| ⚙️ **Flexible Config** | `.env`, YAML, or direct API keys |
| 🛠️ **CLI Included** | Test models directly from command line |
| 📦 **Zero Dependencies** | Lightweight, only essential packages |

---

## 🚀 Quick Start

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

## 📖 Usage Examples

### 🤖 Language Models (LLM)

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

# SiliconFlow
response = siliconflow_chat("Hello!", model="deepseek-ai/DeepSeek-V3")
```

### 👁️ Vision-Language Models (VLM)

```python
from model_api_hub import siliconflow_analyze_image

response = siliconflow_analyze_image(
    image_path="photo.jpg",
    prompt="What's in this image?"
)
```

### 🎨 Image Generation

```python
from model_api_hub import siliconflow_text_to_image

siliconflow_text_to_image(
    prompt="A serene mountain landscape at sunset",
    output_path="landscape.png",
    image_size="1024x1024"
)
```

### 🔊 Text-to-Speech

```python
from model_api_hub import elevenlabs_tts

elevenlabs_tts(
    text="Hello, this is a test.",
    output_path="output.mp3",
    voice_id="21m00Tcm4TlvDq8ikWAM"
)
```

### 🎬 Video Generation

```python
from model_api_hub import runway_generate_video

runway_generate_video(
    prompt="A drone flying over a tropical forest",
    output_path="video.mp4",
    duration=5
)
```

---

## 🛠️ Command Line Interface

```bash
# List all available providers
model-api-hub ls

# Quick test with DeepSeek
model-api-hub deepseek "Hello, how are you?"

# Generate an image
model-api-hub siliconflow-image "A beautiful sunset" --output sunset.png

# Analyze an image
model-api-hub siliconflow-vlm "Describe this image" --image photo.jpg

# Text-to-speech
model-api-hub elevenlabs-tts "Hello world" --output hello.mp3
```

---

## 📋 Supported Providers

### Language Models

| Provider | Import | Models |
|----------|--------|--------|
| **DeepSeek** | `deepseek_chat` | deepseek-chat, deepseek-reasoner |
| **SiliconFlow** | `siliconflow_chat` | DeepSeek-V3, GLM-4.5, Kimi-K2, Qwen3 |
| **Kimi** | `kimi_chat` | moonshot-v1-128k, moonshot-v1-32k |
| **ZhipuAI** | `zhipuai_chat` | glm-4-plus, glm-4-air, glm-4-flash |
| **Yiyan** | `yiyan_chat` | ernie-4.0-8k, ernie-3.5-8k |
| **MiniMax** | `minimax_chat` | abab6.5s-chat, abab6.5-chat |

### Vision-Language Models

| Provider | Import | Models |
|----------|--------|--------|
| **SiliconFlow** | `siliconflow_analyze_image` | Qwen3-VL, GLM-4.5V, step3 |
| **Yiyan** | `yiyan_analyze_image` | ernie-vision-4.0 |

### Image Generation

| Provider | Import | Models |
|----------|--------|--------|
| **SiliconFlow** | `siliconflow_text_to_image` | Kolors, FLUX.1, SD3 |
| **Recraft** | `recraft_text_to_image` | recraft-v3 |

### Audio (TTS)

| Provider | Import | Models |
|----------|--------|--------|
| **ElevenLabs** | `elevenlabs_tts` | eleven_multilingual_v2 |
| **OpenAI** | `openai_tts` | tts-1, tts-1-hd |

### Video Generation

| Provider | Import | Models |
|----------|--------|--------|
| **Runway** | `runway_generate_video` | gen3a_turbo |
| **Luma** | `luma_generate_video` | genie-1.0 |

---

## ⚙️ Configuration

### Option 1: Environment Variables (Recommended)

Create a `.env` file in your project root:

```bash
DEEPSEEK_API_KEY=your_key_here
KIMI_API_KEY=your_key_here
SILICONFLOW_API_KEY=your_key_here
# ... add more as needed
```

The package will automatically load these variables using `python-dotenv`.

### Option 2: Direct Import & Modify

You can directly import the module and modify the API key in code:

```python
from model_api_hub.api.llm import deepseek_llm

# Modify the API key directly
deepseek_llm.API_KEY = "your_api_key_here"

# Now use the functions
response = deepseek_llm.chat("Hello!")
```

Or import specific functions and pass API key as parameter:

```python
from model_api_hub.api.llm.deepseek_llm import chat

response = chat("Hello!", api_key="your_key_here")
```

### Option 3: YAML Config

Create `config.yaml`:

```yaml
llm:
  deepseek:
    model: "deepseek-chat"
    temperature: 0.7
    max_tokens: 4096

vlm:
  siliconflow:
    model: "Qwen/Qwen3-VL-8B-Instruct"
```

Then load it in your code:

```python
from model_api_hub.utils.config import ConfigManager

config = ConfigManager()
api_key = config.get_api_key("deepseek")
```

---

## 🧪 Testing

All test files support direct execution with `if __name__ == "__main__"`:

```bash
# Test LLM providers
python tests/test_llm.py

# Test VLM providers (requires test image)
python tests/test_vlm.py

# Test Image Generation
python tests/test_image.py

# Test Audio (TTS)
python tests/test_audio.py

# Test Video Generation (takes time)
python tests/test_video.py
```

Before running tests, set your API key in the `.env` file or modify it directly in the test file.

---

## 🤝 Contributing

We welcome contributions! Here's how to add a new provider:

1. **Fork** the repository
2. Create a new file in `model_api_hub/api/{category}/`
3. Follow the naming convention: `{provider}_{category}.py`
4. Implement standard functions: `create_client()`, `chat()` or `generate_*()`
5. Add CLI support in `model_api_hub/cli.py`
6. Update `model_api_hub/__init__.py` exports
7. Add tests in `tests/`
8. Submit a **Pull Request**

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📊 Project Stats

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=username/translamate&type=Date)](https://star-history.com/#username/translamate&Date)

</div>

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Thanks to all the AI providers for their amazing APIs:

[DeepSeek](https://www.deepseek.com/) • [Kimi](https://www.moonshot.cn/) • [ZhipuAI](https://open.bigmodel.cn/) • [SiliconFlow](https://cloud.siliconflow.cn/) • [OpenRouter](https://openrouter.ai/) • [MiniMax](https://www.minimaxi.com/) • [Runway](https://runwayml.com/) • [Luma AI](https://lumalabs.ai/) • [ElevenLabs](https://elevenlabs.io/)

---

<div align="center">

**⭐ Star us on GitHub — it motivates us a lot!**

[Report Bug](https://github.com/username/translamate/issues) • [Request Feature](https://github.com/username/translamate/issues) • [Documentation](https://github.com/username/translamate/wiki)

</div>
