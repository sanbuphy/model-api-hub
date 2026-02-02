# LLM Usage Guide

Complete guide for using Language Model (LLM) APIs in Model API Hub.

## Table of Contents

- [Quick Start](#quick-start)
- [Supported Providers](#supported-providers)
- [Synchronous API](#synchronous-api)
- [Streaming API](#streaming-api)
- [Advanced Usage](#advanced-usage)
- [Configuration](#configuration)
- [Error Handling](#error-handling)

## Quick Start

### Installation

```bash
pip install model-api-hub
```

### Basic Usage

```python
from model_api_hub import deepseek_chat

# Simple chat
response = deepseek_chat("Hello, how are you?")
print(response)
```

## Supported Providers

| Provider | Import Path | Default Model | Streaming |
|----------|-------------|---------------|-----------|
| OpenAI | `model_api_hub.api.llm.openai_llm` | gpt-4o | ✅ |
| Anthropic Claude | `model_api_hub.api.llm.anthropic_llm` | claude-sonnet-4-5 | ✅ |
| DeepSeek | `model_api_hub.api.llm.deepseek_llm` | deepseek-chat | ✅ |
| Google Gemini | `model_api_hub.api.llm.gemini_llm` | gemini-pro | ✅ |
| Kimi (Moonshot) | `model_api_hub.api.llm.kimi_llm` | moonshot-v1-128k | ✅ |
| ZhipuAI | `model_api_hub.api.llm.zhipuai_llm` | glm-4-plus | ✅ |
| SiliconFlow | `model_api_hub.api.llm.siliconflow_llm` | deepseek-ai/DeepSeek-V3 | ✅ |
| MiniMax | `model_api_hub.api.llm.minimax_llm` | abab6.5s-chat | ✅ |
| Baidu Yiyan | `model_api_hub.api.llm.yiyan_llm` | ernie-4.0-8k | ✅ |
| Alibaba DashScope | `model_api_hub.api.llm.dashscope_llm` | qwen-max | ✅ |
| ModelScope | `model_api_hub.api.llm.modelscope_llm` | Qwen/Qwen2.5-72B-Instruct | ✅ |
| Xunfei Spark | `model_api_hub.api.llm.xunfei_llm` | Spark-v3.5 | ✅ |
| Groq | `model_api_hub.api.llm.groq_llm` | llama3-70b | ✅ |
| Together AI | `model_api_hub.api.llm.together_llm` | meta-llama/Llama-3-70b | ✅ |
| Mistral | `model_api_hub.api.llm.mistral_llm` | mistral-large | ✅ |
| Cohere | `model_api_hub.api.llm.cohere_llm` | command-r-plus | ✅ |
| Perplexity | `model_api_hub.api.llm.perplexity_llm` | sonar-pro | ✅ |
| Azure OpenAI | `model_api_hub.api.llm.azure_openai_llm` | gpt-4o | ✅ |

## Synchronous API

### Basic Chat

```python
from model_api_hub.api.llm.deepseek_llm import chat

response = chat("What is the capital of France?")
print(response)  # "The capital of France is Paris."
```

### With System Prompt

```python
response = chat(
    prompt="Explain quantum computing",
    system_prompt="You are a physics professor teaching undergraduates."
)
```

### With Custom Parameters

```python
response = chat(
    prompt="Write a creative story",
    system_prompt="You are a creative writer",
    model="deepseek-chat",
    temperature=0.9,      # More creative (0-2)
    max_tokens=2000,      # Longer response
    top_p=0.95           # Nucleus sampling
)
```

### Using Environment Variables

Create `.env` file:

```bash
DEEPSEEK_API_KEY=sk-your-key-here
```

Then use without passing api_key:

```python
response = chat("Hello!")  # Automatically loads from .env
```

## Streaming API

### Basic Streaming

```python
from model_api_hub.api.llm.deepseek_llm import chat_stream

# Stream yields chunks as they arrive
for chunk in chat_stream("Tell me a long story"):
    print(chunk, end="", flush=True)
```

### Streaming with Parameters

```python
for chunk in chat_stream(
    prompt="Write a poem about nature",
    system_prompt="You are a poet",
    temperature=0.8,
    model="deepseek-chat"
):
    print(chunk, end="", flush=True)
```

### Collecting Streamed Response

```python
chunks = []
for chunk in chat_stream("Hello!"):
    chunks.append(chunk)
    print(chunk, end="")

full_response = "".join(chunks)
```

## Advanced Usage

### Multi-turn Conversation

```python
from model_api_hub.api.llm.deepseek_llm import create_client, get_completion

client = create_client(api_key="your-key")

# Build conversation history
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language..."},
    {"role": "user", "content": "What are its main features?"}
]

response = get_completion(client, messages)
print(response)
```

### Switching Providers

```python
from model_api_hub.api.llm import (
    deepseek_chat,
    openai_chat,
    kimi_chat,
    siliconflow_chat
)

# Same interface, different providers
providers = [deepseek_chat, openai_chat, kimi_chat, siliconflow_chat]

for provider in providers:
    try:
        response = provider("Hello!")
        print(f"{provider.__module__}: {response[:50]}...")
    except Exception as e:
        print(f"{provider.__module__}: Error - {e}")
```

### Provider-Specific Models

```python
# DeepSeek
response = deepseek_chat("Hello", model="deepseek-reasoner")

# OpenAI
response = openai_chat("Hello", model="gpt-4o-mini")

# SiliconFlow - access 50+ models
response = siliconflow_chat("Hello", model="Qwen/Qwen2.5-72B-Instruct")
response = siliconflow_chat("Hello", model="meta-llama/Llama-3.1-70B")
```

## Configuration

### Environment Variables

```bash
# Required API Keys
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
```

Load configuration:

```python
from model_api_hub.utils.config import load_config

config = load_config("config.yaml")
```

## Error Handling

### Common Errors

```python
from model_api_hub.api.llm.deepseek_llm import chat

try:
    response = chat("Hello")
except ValueError as e:
    # API key not found
    print(f"Configuration error: {e}")
except Exception as e:
    # API error (rate limit, invalid model, etc.)
    print(f"API error: {e}")
```

### Graceful Degradation

```python
def chat_with_fallback(prompt, providers=None):
    """Try multiple providers until one works."""
    if providers is None:
        from model_api_hub.api.llm import (
            deepseek_chat,
            siliconflow_chat,
            kimi_chat
        )
        providers = [deepseek_chat, siliconflow_chat, kimi_chat]
    
    for provider in providers:
        try:
            return provider(prompt)
        except Exception as e:
            print(f"{provider.__name__} failed: {e}")
            continue
    
    raise Exception("All providers failed")

# Usage
response = chat_with_fallback("Hello!")
```

## Best Practices

1. **Use environment variables** for API keys instead of hardcoding
2. **Handle streaming** for better UX with long responses
3. **Set appropriate temperature**:
   - 0.0-0.3: Factual, deterministic
   - 0.4-0.7: Balanced
   - 0.8-1.0: Creative
   - 1.1-2.0: Very creative/unpredictable
4. **Use system prompts** to guide model behavior
5. **Implement retry logic** for production use
6. **Monitor token usage** to control costs

## Examples

### Chatbot Application

```python
from model_api_hub.api.llm.deepseek_llm import chat

def simple_chatbot():
    print("Chatbot: Hello! Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        response = chat(
            prompt=user_input,
            system_prompt="You are a helpful assistant."
        )
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    simple_chatbot()
```

### Streaming Chatbot

```python
from model_api_hub.api.llm.deepseek_llm import chat_stream

def streaming_chatbot():
    print("Streaming Chatbot: Hello! Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        print("Bot: ", end="", flush=True)
        for chunk in chat_stream(
            prompt=user_input,
            system_prompt="You are a helpful assistant."
        ):
            print(chunk, end="", flush=True)
        print("\n")

if __name__ == "__main__":
    streaming_chatbot()
```

### Multi-Provider Comparison

```python
from model_api_hub.api.llm import (
    deepseek_chat,
    openai_chat,
    kimi_chat
)

def compare_providers(prompt):
    """Compare responses from multiple providers."""
    providers = {
        "DeepSeek": deepseek_chat,
        "OpenAI": openai_chat,
        "Kimi": kimi_chat,
    }
    
    results = {}
    for name, provider in providers.items():
        try:
            response = provider(prompt)
            results[name] = response
        except Exception as e:
            results[name] = f"Error: {e}"
    
    return results

# Usage
prompt = "Explain machine learning in one sentence"
results = compare_providers(prompt)

for provider, response in results.items():
    print(f"\n{provider}:")
    print(f"  {response}")
```
