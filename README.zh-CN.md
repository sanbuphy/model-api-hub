<div align="center">

<img src="assets/logo.png" alt="Model API Hub Logo" width="120" height="120">

# Model API Hub

**一行代码，访问 100+ AI 模型**

[![GitHub release](https://img.shields.io/github/v/release/username/translamate)](https://github.com/username/translamate/releases)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/username/translamate/ci.yml)](https://github.com/username/translamate/actions)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/model-api-hub.svg)](https://badge.fury.io/py/model-api-hub)

[English](README.md) · [简体中文](README.zh-CN.md)

</div>

---

## ✨ Model API Hub 是什么？

Model API Hub 是一个**统一的 Python SDK**，让你可以用**一致、简洁的接口**访问多个平台和模态的 AI 模型 API。

不再为每个提供商使用不同的 SDK，一个库搞定所有需求。

```python
# 相同接口，不同提供商
from model_api_hub import deepseek_chat, siliconflow_chat, kimi_chat

# 使用方式完全一致
response = deepseek_chat("你好！")
response = siliconflow_chat("你好！")
response = kimi_chat("你好！")
```

---

## 🎯 核心特性

| 特性 | 描述 |
|------|------|
| 🔌 **15+ 提供商** | OpenAI、Anthropic、DeepSeek、智谱AI、Kimi、SiliconFlow 等 |
| 🎨 **5 种模态** | 大语言模型、视觉语言模型、图像生成、语音合成、视频生成 |
| 🚀 **一行代码安装** | `pip install model-api-hub` 即可使用 |
| 🔄 **统一 API** | 所有提供商使用相同接口 |
| ⚙️ **灵活配置** | 支持 `.env`、YAML 或直接传入 API key |
| 🛠️ **内置 CLI** | 命令行直接测试模型 |
| 📦 **轻量依赖** | 仅包含必要的依赖包 |

---

## 🚀 快速开始

### 安装

```bash
pip install model-api-hub
```

### 1. 设置 API Key

```bash
# 创建 .env 文件
echo 'DEEPSEEK_API_KEY=your_key_here' > .env
```

### 2. 开始编码

```python
from model_api_hub import deepseek_chat

# 就这么简单
response = deepseek_chat("用简单的语言解释量子计算")
print(response)
```

---

## 📖 使用示例

### 🤖 语言模型 (LLM)

#### 同步对话

```python
from model_api_hub import deepseek_chat, kimi_chat, siliconflow_chat, stepfun_chat

# DeepSeek
response = deepseek_chat(
    "写一个 Python 排序函数",
    system_prompt="你是一位编程专家。"
)

# Kimi (Moonshot)
response = kimi_chat(
    "总结这篇文章",
    temperature=0.5
)

# SiliconFlow - 访问 50+ 模型
response = siliconflow_chat("你好！", model="deepseek-ai/DeepSeek-V3")

# StepFun - OpenAI 兼容接口
response = stepfun_chat(
    "你好，请介绍一下阶跃星辰的人工智能！",
    system_prompt=(
        "你是由阶跃星辰提供的AI聊天助手，你擅长中文、英文以及多种其他语言的对话。"
        "在保证用户数据安全的前提下，你能对用户的问题和请求作出快速和精准的回答。"
        "同时，你的回答和建议应该拒绝黄赌毒、暴力恐怖主义的内容。"
    ),
)
```

#### 流式对话

```python
from model_api_hub import deepseek_chat_stream

# 实时流式响应
for chunk in deepseek_chat_stream("讲一个长故事"):
    print(chunk, end="", flush=True)
```

#### 多轮对话

```python
from model_api_hub.api.llm.deepseek_llm import create_client, get_completion

client = create_client()
messages = [
    {"role": "system", "content": "你是一位有帮助的助手。"},
    {"role": "user", "content": "什么是 Python？"},
    {"role": "assistant", "content": "Python 是一种编程语言..."},
    {"role": "user", "content": "它的主要特性是什么？"}
]

response = get_completion(client, messages)
```

### 👁️ 视觉语言模型 (VLM)

```python
from model_api_hub.api.vlm.openai_vlm import chat

response = chat(
    prompt="这张图片里有什么？",
    image_path="photo.jpg"
)
```

### 🎨 图像生成

```python
from model_api_hub.api.image.siliconflow_image_gen import generate

image_url = generate("宁静的山湖日落景色")
```

### 🔊 语音合成

```python
from model_api_hub.api.audio.elevenlabs_tts import text_to_speech

audio_data = text_to_speech(
    text="你好，这是一个测试。",
    voice_id="21m00Tcm4TlvDq8ikWAM"
)
```

### 🎬 视频生成

```python
from model_api_hub.api.video.runway_video_gen import generate_video

task_id = generate_video(
    prompt="无人机飞越热带雨林的视角",
    duration=5
)
```

---

## 🛠️ 命令行工具

```bash
# 与提供商对话
model-api-hub chat deepseek "你好！"

# 列出所有可用的提供商
model-api-hub list

# 测试提供商
model-api-hub test deepseek
```

---

## 📋 支持的模型

### 语言模型 (LLM)

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

### 视觉语言模型 (VLM)

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

### 图像生成模型

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

### 语音模型

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

### 视频生成模型

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

## ⚙️ 配置

### 环境变量 (.env)

在项目根目录创建 `.env` 文件：

```bash
# LLM 提供商
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
STEP_API_KEY=...

# 其他服务
ELEVENLABS_API_KEY=...
AZURE_SPEECH_KEY=...
STABILITY_API_KEY=...
RECRAFT_API_KEY=...
RUNWAY_API_KEY=...
LUMA_API_KEY=...
```

### YAML 配置

创建 `config.yaml`：

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

## 📚 文档

- [LLM 使用指南](docs/llm_usage.md) - 完整的 LLM 文档
- [API 参考](docs/api_reference.md) - 完整的 API 参考
- [llm.txt](llm.txt) - AI 助手快速参考

---

## 🧪 测试

运行所有提供商的测试：

```bash
# 测试所有 LLM（同步）
python tests/test_llm.py

# 测试流式响应
python tests/test_llm_streaming.py

# 测试其他模态
python tests/test_vlm.py
python tests/test_image.py
python tests/test_audio.py
python tests/test_video.py
```

---

## 🏗️ 项目架构

```
model_api_hub/
├── api/
│   ├── llm/           # 语言模型 (18+ 提供商)
│   ├── vlm/           # 视觉语言模型
│   ├── image/         # 图像生成
│   ├── audio/         # 语音合成
│   └── video/         # 视频生成
├── utils/
│   └── config.py      # 配置管理
├── cli.py             # 命令行接口
└── __init__.py        # 公开 API 导出
```

---

## 🤝 贡献指南

欢迎贡献！详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

### 添加新提供商

1. 在 `model_api_hub/api/llm/{provider}_llm.py` 创建新文件
2. 实现 `chat()`、`chat_stream()`（可选）和 `create_client()`
3. 添加导出到 `model_api_hub/api/llm/__init__.py`
4. 在 `tests/test_llm.py` 中添加测试
5. 更新文档

详见 [llm.txt](llm.txt) 了解详细实现指南。

---

## 📄 许可证

**Apache License 2.0** - 详见 [LICENSE](LICENSE) 文件。

---

## 💬 支持

- 📖 [文档](docs/)
- 🐛 [问题追踪](https://github.com/username/model-api-hub/issues)
- 💬 [讨论](https://github.com/username/model-api-hub/discussions)

---

## 🙏 致谢

感谢所有 AI 提供商提供的优秀 API：

[DeepSeek](https://www.deepseek.com/) • [Kimi](https://www.moonshot.cn/) • [智谱AI](https://open.bigmodel.cn/) • [SiliconFlow](https://cloud.siliconflow.cn/) • [OpenRouter](https://openrouter.ai/) • [MiniMax](https://www.minimaxi.com/) • [Runway](https://runwayml.com/) • [Luma AI](https://lumalabs.ai/) • [ElevenLabs](https://elevenlabs.io/)
