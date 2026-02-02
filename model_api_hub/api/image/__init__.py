"""
Image Generation API providers.

Available providers:
- openai_image_gen - OpenAI DALL-E
- stability_image_gen - Stability AI
- dashscope_image_gen - Alibaba Tongyi Wanxiang
- baidu_image_gen - Baidu ERNIE Image
- xunfei_image_gen - Xunfei Spark Image
- dreamina_gen - Dreamina (即梦) Image
"""

from .openai_image_gen import text_to_image as openai_text_to_image
from .stability_image_gen import text_to_image as stability_text_to_image
from .dashscope_image_gen import text_to_image as dashscope_text_to_image
from .baidu_image_gen import text_to_image as baidu_text_to_image
from .xunfei_image_gen import text_to_image as xunfei_text_to_image
from .dreamina_gen import text_to_image as dreamina_text_to_image

__all__ = [
    "openai_text_to_image",
    "stability_text_to_image",
    "dashscope_text_to_image",
    "baidu_text_to_image",
    "xunfei_text_to_image",
    "dreamina_text_to_image",
]
