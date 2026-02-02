# Supported Models & Providers

This document lists all supported AI models and API providers in Model API Hub.

## Table of Contents

- [LLM (Large Language Models)](#llm-large-language-models)
- [VLM (Vision Language Models)](#vlm-vision-language-models)
- [Image Generation](#image-generation)
- [Audio (TTS/STT)](#audio-ttsstt)
- [Video Generation](#video-generation)
- [Embedding & Rerank](#embedding--rerank)

---

## LLM (Large Language Models)

### 国内 LLM Providers

#### DeepSeek
- **Models**: `deepseek-chat`, `deepseek-reasoner`, `deepseek-coder`
- **Provider**: DeepSeek Official
- **File**: `model_api_hub/api/llm/deepseek_llm.py`

#### GLM (智谱 AI)
- **Models**: `glm-4`, `glm-4-plus`, `glm-4-flash`, `glm-4.7`
- **Provider**: ZhipuAI
- **File**: `model_api_hub/api/llm/zhipuai_llm.py`

#### Qwen (阿里通义千问)
- **Models**: `qwen-max`, `qwen-plus`, `qwen-turbo`, `qwen2.5-72b`, `qwen3-32b`
- **Provider**: Alibaba DashScope
- **File**: `model_api_hub/api/llm/dashscope_llm.py`

#### Kimi (月之暗面)
- **Models**: `moonshot-v1-8k`, `moonshot-v1-32k`, `moonshot-v1-128k`, `kimi-k2.5`
- **Provider**: Moonshot AI
- **File**: `model_api_hub/api/llm/kimi_llm.py`

#### ERNIE (百度文心一言)
- **Models**: `ernie-bot-4`, `ernie-bot`, `ernie-4.5`
- **Provider**: Baidu
- **File**: `model_api_hub/api/llm/yiyan_llm.py`

#### MiniMax
- **Models**: `abab6.5s-chat`, `abab6.5-chat`, `minimax-m2.1`
- **Provider**: MiniMax
- **File**: `model_api_hub/api/llm/minimax_llm.py`

#### Xunfei Spark (讯飞星火)
- **Models**: `generalv3.5`, `4.0Ultra`, `pro-128k`
- **Provider**: iFlytek
- **File**: `model_api_hub/api/llm/xunfei_llm.py`

### 国际 LLM Providers

#### GPT (OpenAI)
- **Models**: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo`
- **Provider**: OpenAI
- **File**: `model_api_hub/api/llm/openai_llm.py`

#### Claude (Anthropic)
- **Models**: `claude-3-5-sonnet`, `claude-3-opus`, `claude-3-haiku`
- **Provider**: Anthropic
- **File**: `model_api_hub/api/llm/anthropic_llm.py`

#### Gemini (Google)
- **Models**: `gemini-1.5-pro`, `gemini-1.5-flash`, `gemini-1.0-pro`
- **Provider**: Google
- **File**: `model_api_hub/api/llm/gemini_llm.py`

#### Llama (Meta)
- **Models**: `llama-3.1-405b`, `llama-3.1-70b`, `llama-3.1-8b`, `llama-3-70b`
- **Providers**: Groq, Together, Fireworks
- **Files**: Multiple aggregator modules

---

## VLM (Vision Language Models)

### 国内 VLM Providers

#### Qwen-VL (阿里通义千问视觉)
- **Models**: `qwen-vl-max`, `qwen-vl-plus`, `qwen2-vl`
- **Provider**: Alibaba DashScope
- **File**: `model_api_hub/api/vlm/dashscope_vlm.py`

#### GLM-4V (智谱 AI 视觉)
- **Models**: `glm-4v`, `glm-4v-plus`
- **Provider**: ZhipuAI
- **File**: `model_api_hub/api/vlm/zhipuai_vlm.py`

#### Xunfei Spark VLM (讯飞星火视觉)
- **Models**: `generalv3.5-vision`
- **Provider**: iFlytek
- **File**: `model_api_hub/api/vlm/xunfei_vlm.py`

### 国际 VLM Providers

#### GPT-4V (OpenAI)
- **Models**: `gpt-4o-vision`, `gpt-4-turbo-vision`
- **Provider**: OpenAI
- **File**: `model_api_hub/api/vlm/openai_vlm.py`

#### Gemini Vision (Google)
- **Models**: `gemini-1.5-pro-vision`, `gemini-1.5-flash-vision`
- **Provider**: Google
- **File**: `model_api_hub/api/vlm/gemini_vlm.py`

#### Claude Vision (Anthropic)
- **Models**: `claude-3-5-sonnet-vision`, `claude-3-opus-vision`
- **Provider**: Anthropic
- **File**: `model_api_hub/api/vlm/anthropic_vlm.py`

---

## Image Generation

### 国内 Image Providers

#### Tongyi Wanxiang (阿里通义万相)
- **Models**: `wanx-v1`, `wanx-v2`
- **Provider**: Alibaba DashScope
- **File**: `model_api_hub/api/image/dashscope_image_gen.py`

#### ERNIE Image (百度文心一格)
- **Models**: `ernie-image-v1`
- **Provider**: Baidu
- **File**: `model_api_hub/api/image/baidu_image_gen.py`

#### Xinghuo Image (讯飞星火绘画)
- **Models**: `spark-image-v1`
- **Provider**: iFlytek
- **File**: `model_api_hub/api/image/xunfei_image_gen.py`

### 国际 Image Providers

#### DALL-E (OpenAI)
- **Models**: `dall-e-3`, `dall-e-2`
- **Provider**: OpenAI
- **File**: `model_api_hub/api/image/openai_image_gen.py`

#### Stable Diffusion (Stability AI)
- **Models**: `sdxl`, `sd-3`, `sd-3.5`
- **Provider**: Stability AI
- **File**: `model_api_hub/api/image/stability_image_gen.py`

#### Midjourney
- **Models**: `midjourney-v6`, `midjourney-v5.2`
- **Provider**: Midjourney (via API)
- **File**: `model_api_hub/api/image/midjourney_image_gen.py`

---

## Audio (TTS/STT)

### 国内 Audio Providers

#### Xunfei TTS (讯飞语音)
- **Models**: `xunfei-tts-v3`, `xunfei-tts-v2`
- **Provider**: iFlytek
- **File**: `model_api_hub/api/audio/xunfei_tts.py`

#### Baidu TTS (百度语音)
- **Models**: `baidu-tts-v1`
- **Provider**: Baidu
- **File**: `model_api_hub/api/audio/baidu_tts.py`

#### Alibaba TTS (阿里语音合成)
- **Models**: `alibaba-tts-v1`
- **Provider**: Alibaba
- **File**: `model_api_hub/api/audio/alibaba_tts.py`

### 国际 Audio Providers

#### ElevenLabs
- **Models**: `eleven-multilingual-v2`, `eleven-turbo-v2`
- **Provider**: ElevenLabs
- **File**: `model_api_hub/api/audio/elevenlabs_tts.py`

#### OpenAI TTS
- **Models**: `tts-1`, `tts-1-hd`
- **Provider**: OpenAI
- **File**: `model_api_hub/api/audio/openai_tts.py`

#### Whisper (OpenAI STT)
- **Models**: `whisper-1`
- **Provider**: OpenAI
- **File**: `model_api_hub/api/stt/openai_whisper.py`

---

## Video Generation

### 国内 Video Providers

#### Kling AI (快手可灵)
- **Models**: `kling-v1`, `kling-v1.5`
- **Provider**: Kuaishou
- **File**: `model_api_hub/api/video/kling_gen.py`

### 国际 Video Providers

#### Runway
- **Models**: `gen-2`, `gen-3-alpha`
- **Provider**: RunwayML
- **File**: `model_api_hub/api/video/runway_gen.py`

#### Luma AI
- **Models**: `dream-machine`
- **Provider**: Luma Labs
- **File**: `model_api_hub/api/video/luma_gen.py`

#### Pika Labs
- **Models**: `pika-1.0`, `pika-1.5`
- **Provider**: Pika
- **File**: `model_api_hub/api/video/pika_gen.py`

---

## API Aggregators (中转站)

### 国内 Aggregators

| Platform | URL | Features |
|----------|-----|----------|
| 七牛云 AI | qiniu.com/products/ai | 中国版 OpenRouter |
| PPIO 派欧云 | ppio.cn | 一站式 AI 云服务 |
| 基石智算 | qingcloud.com | 青云科技 AI 算力 |
| UCloud | ucloud.cn | 孔明智算平台 |
| 快手万擎 | vanchin.kuaishou.com | KAT-Coder 编程 |
| 金山云星流 | ksyun.com | AI 训推平台 |
| 无问芯穹 | infinigence.com | 异构算力集群 |
| 蓝耘元生代 | lanyun.net | 高性能推理 |
| 模力方舟 | ai.gitee.com | Gitee AI 广场 |
| 并行智算云 | ai.paratera.com | 模型广场 |
| 火山方舟 | volcengine.com | 字节 MaaS |
| SophNet | sophnet.com | 算能科技 |
| SiliconFlow | siliconflow.cn | 50+ 开源模型 |
| 302.AI | 302.ai | 一站式 AI 服务 |

### 国际 Aggregators

| Platform | URL | Features |
|----------|-----|----------|
| OpenRouter | openrouter.ai | 200+ 模型 |
| Poe | poe.com | 多模型聚合 |
| Groq | groq.com | 极速推理 |
| Together AI | together.ai | 开源模型 |
| Fireworks | fireworks.ai | 快速推理 |
| Novita | novita.ai | 开源模型 |
| Anyscale | anyscale.com | 生产级部署 |
| Perplexity | perplexity.ai | 搜索增强 |
| Mistral | mistral.ai | 欧洲模型 |
| Cohere | cohere.com | 企业级 NLP |
| AI21 Labs | ai21.com | Jurassic 模型 |

---

## Aggregator Details

### 七牛云 AI {#qiniu-ai}
- **Website**: https://www.qiniu.com/products/ai
- **Features**: 中国版 OpenRouter，统一API架构、多模型调度、Agent+MCP服务
- **Models**: DeepSeek, Qwen, GLM, Llama, Moonshot
- **File**: `model_api_hub/api/aggregators/qiniu_ai.py`

### PPIO 派欧云 {#ppio}
- **Website**: https://ppio.cn/
- **Features**: 一站式 AI 云服务，GPU算力+模型API
- **Models**: DeepSeek-V3, DeepSeek-R1, Llama3, Qwen2.5
- **File**: `model_api_hub/api/aggregators/ppio.py`

### 基石智算 {#coreshub}
- **Website**: https://www.qingcloud.com/products/ai
- **Features**: 青云科技 AI 算力云平台
- **Models**: DeepSeek, Qwen, GLM, Kimi, Llama
- **File**: `model_api_hub/api/aggregators/coreshub.py`

### UCloud 优刻得 {#ucloud}
- **Website**: https://www.ucloud.cn/site/active/agi.html
- **Features**: 孔明智算平台，模型微调+推理
- **Models**: DeepSeek, Qwen, ChatGLM, Llama
- **File**: `model_api_hub/api/aggregators/ucloud_ai.py`

### 快手万擎 {#kuaishou-vanchin}
- **Website**: https://vanchin.kuaishou.com/
- **Features**: KAT-Coder 编程模型，代码生成优化
- **Models**: KAT-Coder, Kwai-Yii
- **File**: `model_api_hub/api/aggregators/kuaishou_vanchin.py`

### 金山云星流 {#ksyun-starflow}
- **Website**: https://www.ksyun.com/product/ai
- **Features**: AI 训推全流程平台
- **Models**: DeepSeek, Qwen, ChatGLM
- **File**: `model_api_hub/api/aggregators/ksyun_starflow.py`

### 无问芯穹 {#infinigence}
- **Website**: https://www.infinigence.com/
- **Features**: 异构算力集群，多芯片支持
- **Models**: Llama, Qwen, Baichuan, ChatGLM
- **File**: `model_api_hub/api/aggregators/infinigence.py`

### 蓝耘元生代 {#lanyun-maas}
- **Website**: https://www.lanyun.net/
- **Features**: 高性能大模型推理服务
- **Models**: DeepSeek, Qwen, GLM, Llama
- **File**: `model_api_hub/api/aggregators/lanyun_maas.py`

### 模力方舟 {#gitee-moark}
- **Website**: https://ai.gitee.com/
- **Features**: Gitee AI 模型广场，开源模型托管
- **Models**: Qwen, ChatGLM, Llama, BERT
- **File**: `model_api_hub/api/aggregators/gitee_moark.py`

### 并行智算云 {#paratera-ai}
- **Website**: https://ai.paratera.com/
- **Features**: 模型广场+API服务，科研算力支持
- **Models**: DeepSeek, Qwen, Llama, ChatGLM
- **File**: `model_api_hub/api/aggregators/paratera_ai.py`

### 火山方舟 {#volcengine-ark}
- **Website**: https://www.volcengine.com/product/ark
- **Features**: 字节跳动 MaaS 平台，模型精调+推理
- **Models**: Doubao, DeepSeek, Qwen, Llama
- **File**: `model_api_hub/api/aggregators/volcengine_ark.py`

### SophNet {#sophnet}
- **Website**: https://sophnet.com/
- **Features**: 算能科技 DeepSeek 极速版
- **Models**: DeepSeek-R1-Distill, DeepSeek-V3
- **File**: `model_api_hub/api/aggregators/sophnet.py`

### SiliconFlow {#siliconflow}
- **Website**: https://siliconflow.cn/
- **Features**: 50+ 开源模型，高性价比
- **Models**: DeepSeek-V3, Qwen2.5, GLM-4, Llama3.1
- **File**: `model_api_hub/api/aggregators/siliconflow.py`

### 302.AI {#ai302}
- **Website**: https://302.ai/
- **Features**: 一站式 AI 服务，Bot+API
- **Models**: GPT-4, Claude, Gemini, 国内大模型
- **File**: `model_api_hub/api/aggregators/ai302.py`

---

## Model Details

### GLM-4.7-Flash
- **Provider**: ZhipuAI
- **Context**: 128K
- **Features**: Fast, cost-effective
- **API**: OpenAI-compatible

### DeepSeek-R1
- **Provider**: DeepSeek
- **Context**: 64K
- **Features**: Reasoning, coding
- **API**: DeepSeek API

### Qwen3
- **Provider**: Alibaba
- **Context**: 128K
- **Features**: Multilingual, coding
- **API**: DashScope

### MiniMax-M2
- **Provider**: MiniMax
- **Context**: 200K
- **Features**: Long context
- **API**: MiniMax API

### Kimi K2.5
- **Provider**: Moonshot AI
- **Context**: 200K
- **Features**: Long context, Chinese optimized
- **API**: Moonshot API

---

## Usage Examples

### LLM
```python
from model_api_hub.api.llm.deepseek_llm import chat
response = chat("Hello!", api_key="your_key")
```

### VLM
```python
from model_api_hub.api.vlm.dashscope_vlm import analyze_image
response = analyze_image("image.jpg", "Describe this image", api_key="your_key")
```

### Image
```python
from model_api_hub.api.image.openai_image_gen import text_to_image
image_url = text_to_image("A cat", api_key="your_key")
```

### Aggregator
```python
from model_api_hub.api.aggregators import qiniu_ai_chat
response = qiniu_ai_chat("Hello!", api_key="your_key", model="deepseek-chat")
```

---

## Contributing

To add a new model or provider:

1. Create a new module in the appropriate `api/` subdirectory
2. Follow the existing code patterns
3. Update this documentation
4. Add environment variables to `.env.example`
5. Submit a pull request

---

*Last updated: 2026-02-02*
