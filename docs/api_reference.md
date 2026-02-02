# API Reference

Complete API reference for Model API Hub.

## Table of Contents

- [LLM API](#llm-api)
- [VLM API](#vlm-api)
- [Image Generation API](#image-generation-api)
- [Audio API](#audio-api)
- [Video API](#video-api)

## LLM API

### Synchronous Chat

All LLM providers support the same synchronous chat interface.

#### Function Signature

```python
def chat(
    prompt: str,
    system_prompt: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    top_p: float = 0.9,
    **kwargs
) -> str
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `prompt` | str | Yes | - | User message/prompt |
| `system_prompt` | str | No | None | System instruction |
| `api_key` | str | No | None | API key (loads from env if None) |
| `model` | str | No | Provider default | Model name |
| `temperature` | float | No | 0.7 | Sampling temperature (0-2) |
| `max_tokens` | int | No | 4096 | Maximum tokens in response |
| `top_p` | float | No | 0.9 | Nucleus sampling threshold |
| `**kwargs` | dict | No | - | Additional provider-specific params |

#### Returns

- **str**: The model's response text

#### Example

```python
from model_api_hub.api.llm.deepseek_llm import chat

response = chat(
    prompt="Explain quantum computing",
    system_prompt="You are a physics professor",
    temperature=0.8
)
```

### Streaming Chat

Streaming APIs yield response chunks as they arrive from the model.

#### Function Signature

```python
def chat_stream(
    prompt: str,
    system_prompt: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    top_p: float = 0.9,
    **kwargs
) -> Iterator[str]
```

#### Parameters

Same as synchronous `chat()` function.

#### Returns

- **Iterator[str]**: Yields text chunks

#### Example

```python
from model_api_hub.api.llm.deepseek_llm import chat_stream

for chunk in chat_stream("Tell me a story"):
    print(chunk, end="", flush=True)
```

### Low-Level Client API

For advanced use cases, use the client API directly.

#### create_client

```python
def create_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> Client
```

Create a provider-specific client instance.

#### get_completion

```python
def get_completion(
    client: Client,
    messages: List[Dict[str, str]],
    model: str,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    stream: bool = False,
    **kwargs
) -> Union[str, Iterator[str]]
```

Send a completion request with full message history.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `client` | Client | Client instance from `create_client()` |
| `messages` | List[Dict] | List of message dicts with `role` and `content` |
| `model` | str | Model name |
| `max_tokens` | int | Maximum tokens |
| `temperature` | float | Temperature (0-2) |
| `stream` | bool | Whether to stream response |

**Example:**

```python
from model_api_hub.api.llm.deepseek_llm import create_client, get_completion

client = create_client(api_key="sk-...")

messages = [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello!"}
]

response = get_completion(client, messages, model="deepseek-chat")
```

## Provider-Specific APIs

### DeepSeek

```python
from model_api_hub.api.llm.deepseek_llm import (
    chat,
    chat_stream,
    create_client,
    get_completion
)

# Available models:
# - deepseek-chat (default)
# - deepseek-reasoner

response = chat("Hello", model="deepseek-chat")
```

### OpenAI

```python
from model_api_hub.api.llm.openai_llm import (
    chat,
    chat_stream,
    create_client,
    get_completion
)

# Available models:
# - gpt-4o (default)
# - gpt-4o-mini
# - gpt-4-turbo
# - gpt-3.5-turbo

response = chat("Hello", model="gpt-4o")
```

### Anthropic Claude

```python
from model_api_hub.api.llm.anthropic_llm import (
    chat,
    create_client,
    get_completion
)

# Available models:
# - claude-sonnet-4-5 (default)
# - claude-opus-4
# - claude-haiku-3-5

response = chat("Hello", model="claude-sonnet-4-5")
```

### Kimi (Moonshot)

```python
from model_api_hub.api.llm.kimi_llm import (
    chat,
    create_client,
    get_completion
)

# Available models:
# - moonshot-v1-128k (default)
# - moonshot-v1-32k
# - moonshot-v1-8k

response = chat("Hello", model="moonshot-v1-128k")
```

### SiliconFlow

```python
from model_api_hub.api.llm.siliconflow_llm import (
    chat,
    create_client,
    get_completion
)

# Available models (50+ models):
# - deepseek-ai/DeepSeek-V3 (default)
# - Qwen/Qwen2.5-72B-Instruct
# - meta-llama/Llama-3.1-70B
# - THUDM/glm-4-9b-chat
# - And more...

response = chat("Hello", model="deepseek-ai/DeepSeek-V3")
```

### ZhipuAI

```python
from model_api_hub.api.llm.zhipuai_llm import (
    chat,
    create_client,
    get_completion
)

# Available models:
# - glm-4-plus (default)
# - glm-4-flash
# - glm-4-air

response = chat("你好", model="glm-4-plus")
```

### MiniMax

```python
from model_api_hub.api.llm.minimax_llm import (
    chat,
    create_client,
    get_completion
)

# Available models:
# - abab6.5s-chat (default)
# - abab6.5-chat

response = chat("你好", model="abab6.5s-chat")
```

## VLM API

Vision-Language Models for image understanding.

### chat

```python
def chat(
    prompt: str,
    image_path: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs
) -> str
```

**Example:**

```python
from model_api_hub.api.vlm.openai_vlm import chat

response = chat(
    prompt="What's in this image?",
    image_path="path/to/image.jpg"
)
```

## Image Generation API

### generate

```python
def generate(
    prompt: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    size: str = "1024x1024",
    **kwargs
) -> str
```

**Example:**

```python
from model_api_hub.api.image.siliconflow_image_gen import generate

image_url = generate(
    prompt="A beautiful sunset over mountains",
    size="1024x1024"
)
```

## Audio API

### Text-to-Speech

```python
def synthesize(
    text: str,
    api_key: Optional[str] = None,
    voice: Optional[str] = None,
    output_path: Optional[str] = None,
    **kwargs
) -> Union[bytes, str]
```

**Example:**

```python
from model_api_hub.api.audio.openai_tts import synthesize

audio_data = synthesize(
    text="Hello, world!",
    voice="alloy",
    output_path="output.mp3"
)
```

## Video API

### generate

```python
def generate(
    prompt: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    duration: int = 5,
    **kwargs
) -> str
```

**Example:**

```python
from model_api_hub.api.video.runway_gen import generate

video_url = generate(
    prompt="A cat playing with a ball",
    duration=5
)
```

## Configuration API

### load_config

```python
from model_api_hub.utils.config import load_config

config = load_config("config.yaml")
```

### get_api_key

```python
from model_api_hub.utils.config import get_api_key

api_key = get_api_key("deepseek")  # Loads DEEPSEEK_API_KEY from env
```

## Error Handling

All APIs raise standard Python exceptions:

- `ValueError`: Configuration errors (missing API key, invalid params)
- `RuntimeError`: API errors (rate limits, model errors)
- `ConnectionError`: Network issues

**Example:**

```python
from model_api_hub.api.llm.deepseek_llm import chat

try:
    response = chat("Hello")
except ValueError as e:
    print(f"Config error: {e}")
except RuntimeError as e:
    print(f"API error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Type Hints

All functions include complete type hints:

```python
from typing import Optional, Iterator, List, Dict, Any

def chat(
    prompt: str,
    system_prompt: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any
) -> str: ...

def chat_stream(
    prompt: str,
    **kwargs: Any
) -> Iterator[str]: ...
```
