"""
测试所有配置了 API Key 的 LLM 可用性
"""
import os
import sys
import asyncio
from dotenv import load_dotenv
from datetime import datetime

# 加载环境变量
load_dotenv()

# 添加项目路径
sys.path.insert(0, '/Users/sanbu/Code/2026重要开源项目/model-api-hub')

# 导入各个 LLM 模块
from model_api_hub.api.llm.deepseek_llm import chat as deepseek_chat
from model_api_hub.api.llm.siliconflow_llm import chat as siliconflow_chat
from model_api_hub.api.llm.zhipuai_llm import chat as zhipuai_chat
from model_api_hub.api.llm.kimi_llm import chat as kimi_chat
from model_api_hub.api.llm.minimax_llm import chat as minimax_chat
from model_api_hub.api.llm.yiyan_llm import chat as yiyan_chat
from model_api_hub.api.llm.modelscope_llm import chat as modelscope_chat
from model_api_hub.api.llm.dashscope_llm import chat as dashscope_chat
from model_api_hub.api.llm.openai_llm import chat as openai_chat
from model_api_hub.api.llm.anthropic_llm import chat as anthropic_chat
from model_api_hub.api.llm.gemini_llm import chat as gemini_chat
from model_api_hub.api.llm.groq_llm import chat as groq_chat
from model_api_hub.api.llm.together_llm import chat as together_chat
from model_api_hub.api.llm.mistral_llm import chat as mistral_chat
from model_api_hub.api.llm.cohere_llm import chat as cohere_chat
from model_api_hub.api.llm.xunfei_llm import chat as xunfei_chat
from model_api_hub.api.llm.perplexity_llm import chat as perplexity_chat
from model_api_hub.api.llm.azure_openai_llm import chat as azure_chat


def test_llm(name, chat_func, api_key, prompt="你好，请回复'测试通过'四个字。"):
    """测试单个 LLM"""
    if not api_key:
        return {"name": name, "status": "SKIPPED", "reason": "未配置 API Key", "response": None}
    
    print(f"\n{'='*50}")
    print(f"正在测试: {name}")
    print(f"{'='*50}")
    
    try:
        response = chat_func(prompt=prompt, api_key=api_key)
        if response and len(response) > 0:
            print(f"✅ {name} - 测试通过")
            print(f"响应: {response[:100]}...")
            return {"name": name, "status": "✅ PASS", "response": response[:100]}
        else:
            print(f"❌ {name} - 返回空响应")
            return {"name": name, "status": "❌ FAIL", "reason": "返回空响应", "response": None}
    except Exception as e:
        print(f"❌ {name} - 测试失败: {str(e)}")
        return {"name": name, "status": "❌ FAIL", "reason": str(e), "response": None}


def main():
    print(f"\n{'#'*60}")
    print(f"# LLM 可用性测试")
    print(f"# 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}\n")
    
    # 定义要测试的 LLM 列表
    llm_tests = [
        ("DeepSeek", deepseek_chat, os.getenv("DEEPSEEK_API_KEY")),
        ("SiliconFlow", siliconflow_chat, os.getenv("SILICONFLOW_API_KEY")),
        ("ZhipuAI (智谱)", zhipuai_chat, os.getenv("ZHIPUAI_API_KEY")),
        ("Kimi (月之暗面)", kimi_chat, os.getenv("KIMI_API_KEY")),
        ("MiniMax", minimax_chat, os.getenv("MINIMAX_API_KEY")),
        ("百度文心一言", yiyan_chat, os.getenv("YIYAN_API_KEY")),
        ("ModelScope", modelscope_chat, os.getenv("MODELSCOPE_API_KEY")),
        ("阿里 DashScope", dashscope_chat, os.getenv("DASHSCOPE_API_KEY")),
        ("OpenAI", openai_chat, os.getenv("OPENAI_API_KEY")),
        ("Anthropic Claude", anthropic_chat, os.getenv("ANTHROPIC_API_KEY")),
        ("Google Gemini", gemini_chat, os.getenv("GEMINI_API_KEY")),
        ("Groq", groq_chat, os.getenv("GROQ_API_KEY")),
        ("Together AI", together_chat, os.getenv("TOGETHER_API_KEY")),
        ("Mistral", mistral_chat, os.getenv("MISTRAL_API_KEY")),
        ("Cohere", cohere_chat, os.getenv("COHERE_API_KEY")),
        ("讯飞星火", xunfei_chat, os.getenv("XUNFEI_SPARK_API_KEY")),
        ("Perplexity", perplexity_chat, os.getenv("PERPLEXITY_API_KEY")),
        ("Azure OpenAI", azure_chat, os.getenv("AZURE_OPENAI_API_KEY")),
    ]
    
    results = []
    
    for name, chat_func, api_key in llm_tests:
        result = test_llm(name, chat_func, api_key)
        results.append(result)
    
    # 打印汇总结果
    print(f"\n\n{'='*60}")
    print("测试结果汇总")
    print(f"{'='*60}")
    
    passed = [r for r in results if r["status"] == "✅ PASS"]
    failed = [r for r in results if r["status"] == "❌ FAIL"]
    skipped = [r for r in results if r["status"] == "SKIPPED"]
    
    print(f"\n✅ 通过: {len(passed)} 个")
    for r in passed:
        print(f"  - {r['name']}")
    
    print(f"\n❌ 失败: {len(failed)} 个")
    for r in failed:
        reason = r.get('reason', '未知错误')
        print(f"  - {r['name']}: {reason}")
    
    print(f"\n⏭️  跳过: {len(skipped)} 个 (未配置 API Key)")
    for r in skipped:
        print(f"  - {r['name']}")
    
    print(f"\n{'='*60}")
    print(f"总计: {len(results)} 个 LLM")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    main()
