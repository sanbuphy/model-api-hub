<div align="center">

<img src="assets/logo.png" alt="Model API Hub Logo" width="120" height="120">

# Model API Hub

**一行代码，访问 100+ AI 模型**

[![GitHub release](https://img.shields.io/github/v/release/username/translamate)](https://github.com/username/translamate/releases)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
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

```python
from model_api_hub import deepseek_chat, kimi_chat, siliconflow_chat

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

# SiliconFlow
response = siliconflow_chat("你好！", model="deepseek-ai/DeepSeek-V3")
```

### 👁️ 视觉语言模型 (VLM)

```python
from model_api_hub import siliconflow_analyze_image

response = siliconflow_analyze_image(
    image_path="photo.jpg",
    prompt="这张图片里有什么？"
)
```

### 🎨 图像生成

```python
from model_api_hub import siliconflow_text_to_image

siliconflow_text_to_image(
    prompt="宁静的山湖日落景色",
    output_path="landscape.png",
    image_size="1024x1024"
)
```

### 🔊 语音合成

```python
from model_api_hub import elevenlabs_tts

elevenlabs_tts(
    text="你好，这是一个测试。",
    output_path="output.mp3",
    voice_id="21m00Tcm4TlvDq8ikWAM"
)
```

### 🎬 视频生成

```python
from model_api_hub import runway_generate_video

runway_generate_video(
    prompt="无人机飞越热带雨林的视角",
    output_path="video.mp4",
    duration=5
)
```

---

## 🛠️ 命令行工具

```bash
# 列出所有可用的提供商
model-api-hub ls

# 快速测试 DeepSeek
model-api-hub deepseek "你好，最近怎么样？"

# 生成图片
model-api-hub siliconflow-image "美丽的日落" --output sunset.png

# 分析图片
model-api-hub siliconflow-vlm "描述这张图片" --image photo.jpg

# 语音合成
model-api-hub elevenlabs-tts "你好世界" --output hello.mp3
```

---

## 📋 支持的提供商

### 语言模型

| 提供商 | 导入方式 | 模型 |
|--------|----------|------|
| **DeepSeek** | `deepseek_chat` | deepseek-chat, deepseek-reasoner |
| **SiliconFlow** | `siliconflow_chat` | DeepSeek-V3, GLM-4.5, Kimi-K2, Qwen3 |
| **Kimi** | `kimi_chat` | moonshot-v1-128k, moonshot-v1-32k |
| **智谱AI** | `zhipuai_chat` | glm-4-plus, glm-4-air, glm-4-flash |
| **文心一言** | `yiyan_chat` | ernie-4.0-8k, ernie-3.5-8k |
| **MiniMax** | `minimax_chat` | abab6.5s-chat, abab6.5-chat |

### 视觉语言模型

| 提供商 | 导入方式 | 模型 |
|--------|----------|------|
| **SiliconFlow** | `siliconflow_analyze_image` | Qwen3-VL, GLM-4.5V, step3 |
| **文心一言** | `yiyan_analyze_image` | ernie-vision-4.0 |

### 图像生成

| 提供商 | 导入方式 | 模型 |
|--------|----------|------|
| **SiliconFlow** | `siliconflow_text_to_image` | Kolors, FLUX.1, SD3 |
| **Recraft** | `recraft_text_to_image` | recraft-v3 |

### 语音合成 (TTS)

| 提供商 | 导入方式 | 模型 |
|--------|----------|------|
| **ElevenLabs** | `elevenlabs_tts` | eleven_multilingual_v2 |
| **OpenAI** | `openai_tts` | tts-1, tts-1-hd |

### 视频生成

| 提供商 | 导入方式 | 模型 |
|--------|----------|------|
| **Runway** | `runway_generate_video` | gen3a_turbo |
| **Luma** | `luma_generate_video` | genie-1.0 |

---

## ⚙️ 配置方式

### 方式 1：环境变量（推荐）

在项目根目录创建 `.env` 文件：

```bash
DEEPSEEK_API_KEY=your_key_here
KIMI_API_KEY=your_key_here
SILICONFLOW_API_KEY=your_key_here
# ... 按需添加更多
```

包会自动使用 `python-dotenv` 加载这些变量。

### 方式 2：直接导入修改

你可以直接导入模块并在代码中修改 API key：

```python
from model_api_hub.api.llm import deepseek_llm

# 直接修改 API key
deepseek_llm.API_KEY = "your_api_key_here"

# 然后使用函数
response = deepseek_llm.chat("你好！")
```

或者导入具体函数并传入 API key 参数：

```python
from model_api_hub.api.llm.deepseek_llm import chat

response = chat("你好！", api_key="your_key_here")
```

### 方式 3：YAML 配置

创建 `config.yaml`：

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

然后在代码中加载：

```python
from model_api_hub.utils.config import ConfigManager

config = ConfigManager()
api_key = config.get_api_key("deepseek")
```

---

## 🧪 测试

所有测试文件都支持使用 `if __name__ == "__main__"` 直接执行：

```bash
# 测试 LLM 提供商
python tests/test_llm.py

# 测试 VLM 提供商（需要测试图片）
python tests/test_vlm.py

# 测试图像生成
python tests/test_image.py

# 测试语音合成
python tests/test_audio.py

# 测试视频生成（需要较长时间）
python tests/test_video.py
```

运行测试前，请在 `.env` 文件中设置 API key，或直接在测试文件中修改。

---

## 🤝 贡献指南

欢迎贡献！添加新提供商的步骤：

1. **Fork** 本仓库
2. 在 `model_api_hub/api/{category}/` 创建新文件
3. 遵循命名规范：`{provider}_{category}.py`
4. 实现标准函数：`create_client()`、`chat()` 或 `generate_*()`
5. 在 `model_api_hub/cli.py` 中添加 CLI 支持
6. 更新 `model_api_hub/__init__.py` 导出
7. 在 `tests/` 中添加测试
8. 提交 **Pull Request**

详见 [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 📊 项目统计

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=username/translamate&type=Date)](https://star-history.com/#username/translamate&Date)

</div>

---

## 📄 许可证

本项目采用 **MIT 许可证** - 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

感谢所有 AI 提供商提供的优秀 API：

[DeepSeek](https://www.deepseek.com/) • [Kimi](https://www.moonshot.cn/) • [智谱AI](https://open.bigmodel.cn/) • [SiliconFlow](https://cloud.siliconflow.cn/) • [OpenRouter](https://openrouter.ai/) • [MiniMax](https://www.minimaxi.com/) • [Runway](https://runwayml.com/) • [Luma AI](https://lumalabs.ai/) • [ElevenLabs](https://elevenlabs.io/)

---

<div align="center">

**⭐ 在 GitHub 上给我们点 Star —— 这对我们是很大的鼓励！**

[报告 Bug](https://github.com/username/translamate/issues) • [功能建议](https://github.com/username/translamate/issues) • [文档](https://github.com/username/translamate/wiki)

</div>
