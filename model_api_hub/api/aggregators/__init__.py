"""
API Aggregators / API中转站

提供统一的接口访问多个AI模型提供商。

国内平台 (China):
- qiniu_ai - 七牛云 AI 大模型广场
- ppio - PPIO 派欧云
- coreshub - 基石智算 (青云科技)
- ucloud_ai - UCloud 优刻得
- kuaishou_vanchin - 快手万擎
- ksyun_starflow - 金山云星流
- infinigence - 无问芯穹
- lanyun_maas - 蓝耘元生代
- gitee_moark - 模力方舟 (Gitee)
- paratera_ai - 并行智算云
- volcengine_ark - 火山方舟 (字节跳动)
- sophnet - SophNet (算能科技)
- dreamina - 即梦 (字节跳动)
- modelscope - ModelScope (魔搭社区)
- baidu_aistudio - 百度 AI Studio
- hailuo - 海螺AI (MiniMax)
- kling - 可灵AI (快手)
- sensetime - 商汤日日新
- tencent_hunyuan - 腾讯混元

国际平台 (International):
- openrouter - OpenRouter
- siliconflow - SiliconFlow
- poe - Poe
- ai302 - 302.AI
- fireworks - Fireworks AI
- novita - Novita AI
- groq - Groq
- together - Together AI
- anyscale - Anyscale
- perplexity - Perplexity
- mistral - Mistral AI
- cohere - Cohere
- ai21 - AI21 Labs
"""

# 国内平台
from .qiniu_ai import chat as qiniu_ai_chat, create_client as qiniu_ai_client
from .ppio import chat as ppio_chat, create_client as ppio_client
from .coreshub import chat as coreshub_chat, create_client as coreshub_client
from .ucloud_ai import chat as ucloud_ai_chat, create_client as ucloud_ai_client
from .kuaishou_vanchin import chat as kuaishou_vanchin_chat, create_client as kuaishou_vanchin_client
from .ksyun_starflow import chat as ksyun_starflow_chat, create_client as ksyun_starflow_client
from .infinigence import chat as infinigence_chat, create_client as infinigence_client
from .lanyun_maas import chat as lanyun_maas_chat, create_client as lanyun_maas_client
from .gitee_moark import chat as gitee_moark_chat, create_client as gitee_moark_client
from .paratera_ai import chat as paratera_ai_chat, create_client as paratera_ai_client
from .volcengine_ark import chat as volcengine_ark_chat, create_client as volcengine_ark_client
from .sophnet import chat as sophnet_chat, create_client as sophnet_client
from .dreamina import text_to_image as dreamina_text_to_image, text_to_video as dreamina_text_to_video
from .modelscope import chat as modelscope_chat, text_to_image as modelscope_text_to_image, get_embedding as modelscope_embedding
from .baidu_aistudio import chat as baidu_aistudio_chat, analyze_image as baidu_aistudio_analyze_image, text_to_image as baidu_aistudio_text_to_image
from .hailuo import text_to_video as hailuo_text_to_video, text_to_speech as hailuo_tts, clone_voice as hailuo_clone_voice
from .kling import text_to_video as kling_text_to_video, image_to_video as kling_image_to_video, text_to_image as kling_text_to_image
from .sensetime import chat as sensetime_chat, analyze_image as sensetime_analyze_image, text_to_image as sensetime_text_to_image, text_to_video as sensetime_text_to_video
from .tencent_hunyuan import chat as tencent_hunyuan_chat, analyze_image as tencent_hunyuan_analyze_image, text_to_image as tencent_hunyuan_text_to_image

# 国际平台
from .openrouter import chat as openrouter_chat, create_client as openrouter_client
from .siliconflow import chat as siliconflow_chat, create_client as siliconflow_client
from .poe import chat as poe_chat, create_client as poe_client
from .ai302 import chat as ai302_chat, create_client as ai302_client
from .fireworks import chat as fireworks_chat, create_client as fireworks_client
from .novita import chat as novita_chat, create_client as novita_client
from .groq import chat as groq_chat, create_client as groq_client
from .together import chat as together_chat, create_client as together_client
from .anyscale import chat as anyscale_chat, create_client as anyscale_client
from .perplexity import chat as perplexity_chat, create_client as perplexity_client
from .mistral import chat as mistral_chat, create_client as mistral_client
from .cohere import chat as cohere_chat, create_client as cohere_client
from .ai21 import chat as ai21_chat, create_client as ai21_client

__all__ = [
    # 国内平台
    "qiniu_ai_chat", "qiniu_ai_client",
    "ppio_chat", "ppio_client",
    "coreshub_chat", "coreshub_client",
    "ucloud_ai_chat", "ucloud_ai_client",
    "kuaishou_vanchin_chat", "kuaishou_vanchin_client",
    "ksyun_starflow_chat", "ksyun_starflow_client",
    "infinigence_chat", "infinigence_client",
    "lanyun_maas_chat", "lanyun_maas_client",
    "gitee_moark_chat", "gitee_moark_client",
    "paratera_ai_chat", "paratera_ai_client",
    "volcengine_ark_chat", "volcengine_ark_client",
    "sophnet_chat", "sophnet_client",
    # 新增国内平台
    "dreamina_text_to_image", "dreamina_text_to_video",
    "modelscope_chat", "modelscope_text_to_image", "modelscope_embedding",
    "baidu_aistudio_chat", "baidu_aistudio_analyze_image", "baidu_aistudio_text_to_image",
    "hailuo_text_to_video", "hailuo_tts", "hailuo_clone_voice",
    "kling_text_to_video", "kling_image_to_video", "kling_text_to_image",
    "sensetime_chat", "sensetime_analyze_image", "sensetime_text_to_image", "sensetime_text_to_video",
    "tencent_hunyuan_chat", "tencent_hunyuan_analyze_image", "tencent_hunyuan_text_to_image",
    
    # 国际平台
    "openrouter_chat", "openrouter_client",
    "siliconflow_chat", "siliconflow_client",
    "poe_chat", "poe_client",
    "ai302_chat", "ai302_client",
    "fireworks_chat", "fireworks_client",
    "novita_chat", "novita_client",
    "groq_chat", "groq_client",
    "together_chat", "together_client",
    "anyscale_chat", "anyscale_client",
    "perplexity_chat", "perplexity_client",
    "mistral_chat", "mistral_client",
    "cohere_chat", "cohere_client",
    "ai21_chat", "ai21_client",
]
