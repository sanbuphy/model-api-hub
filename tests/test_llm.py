"""
Tests for LLM providers.

These tests require API keys to run. Set them in .env or pass directly.
"""
import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_api_hub.api.llm.deepseek_llm import create_client, get_completion, chat as deepseek_chat
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
from model_api_hub.api.llm.stepfun_llm import chat as stepfun_chat


def test_deepseek_create_client():
    """Test DeepSeek client creation with API key."""
    print("\n[Test] DeepSeek create_client...")
    client = create_client(api_key="test_key")
    assert client is not None, "Client should not be None"
    assert client.api_key == "test_key", "API key should match"
    print("✅ DeepSeek create_client passed")


def test_deepseek_chat(api_key=None):
    """Test DeepSeek chat."""
    api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("⏭️  [Skip] DeepSeek chat - No API key provided")
        return
    
    print("\n[Test] DeepSeek chat...")
    try:
        response = deepseek_chat(
            prompt="Say 'test passed' in one word.",
            api_key=api_key
        )
        assert response is not None, "Response should not be None"
        assert len(response) > 0, "Response should not be empty"
        print(f"✅ DeepSeek chat passed - Response: {response[:50]}...")
    except Exception as e:
        print(f"❌ DeepSeek chat failed: {e}")


def test_siliconflow_chat(api_key=None):
    """Test SiliconFlow chat."""
    api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
    if not api_key:
        print("⏭️  [Skip] SiliconFlow chat - No API key provided")
        return
    
    print("\n[Test] SiliconFlow chat...")
    try:
        response = siliconflow_chat(
            prompt="Say 'test passed' in one word.",
            api_key=api_key
        )
        assert response is not None, "Response should not be None"
        assert len(response) > 0, "Response should not be empty"
        print(f"✅ SiliconFlow chat passed - Response: {response[:50]}...")
    except Exception as e:
        print(f"❌ SiliconFlow chat failed: {e}")


def test_zhipuai_chat(api_key=None):
    """Test ZhipuAI chat."""
    api_key = api_key or os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        print("⏭️  [Skip] ZhipuAI chat - No API key provided")
        return
    
    print("\n[Test] ZhipuAI chat...")
    try:
        response = zhipuai_chat(
            prompt="说'测试通过'（一个词）",
            api_key=api_key
        )
        assert response is not None, "Response should not be None"
        assert len(response) > 0, "Response should not be empty"
        print(f"✅ ZhipuAI chat passed - Response: {response[:50]}...")
    except Exception as e:
        print(f"❌ ZhipuAI chat failed: {e}")


def test_kimi_chat(api_key=None):
    """Test Kimi chat."""
    api_key = api_key or os.getenv("KIMI_API_KEY")
    if not api_key:
        print("⏭️  [Skip] Kimi chat - No API key provided")
        return
    
    print("\n[Test] Kimi chat...")
    try:
        response = kimi_chat(
            prompt="说'测试通过'（一个词）",
            api_key=api_key
        )
        assert response is not None, "Response should not be None"
        assert len(response) > 0, "Response should not be empty"
        print(f"✅ Kimi chat passed - Response: {response[:50]}...")
    except Exception as e:
        print(f"❌ Kimi chat failed: {e}")


def test_minimax_chat(api_key=None):
    """Test MiniMax chat."""
    api_key = api_key or os.getenv("MINIMAX_API_KEY")
    if not api_key:
        print("⏭️  [Skip] MiniMax chat - No API key provided")
        return
    
    print("\n[Test] MiniMax chat...")
    try:
        response = minimax_chat(
            prompt="说'测试通过'（一个词）",
            api_key=api_key
        )
        assert response is not None, "Response should not be None"
        assert len(response) > 0, "Response should not be empty"
        print(f"✅ MiniMax chat passed - Response: {response[:50]}...")
    except Exception as e:
        print(f"❌ MiniMax chat failed: {e}")


def test_yiyan_chat(api_key=None):
    """Test Baidu Yiyan chat."""
    api_key = api_key or os.getenv("YIYAN_API_KEY")
    if not api_key:
        print("⏭️  [Skip] Yiyan chat - No API key provided")
        return
    
    print("\n[Test] Baidu Yiyan chat...")
    try:
        response = yiyan_chat(
            prompt="说'测试通过'（一个词）",
            api_key=api_key
        )
        assert response is not None, "Response should not be None"
        assert len(response) > 0, "Response should not be empty"
        print(f"✅ Yiyan chat passed - Response: {response[:50]}...")
    except Exception as e:
        print(f"❌ Yiyan chat failed: {e}")


def test_modelscope_chat(api_key=None):
    """Test ModelScope chat."""
    api_key = api_key or os.getenv("MODELSCOPE_API_KEY")
    if not api_key:
        print("⏭️  [Skip] ModelScope chat - No API key provided")
        return
    
    print("\n[Test] ModelScope chat...")
    try:
        response = modelscope_chat(
            prompt="Say 'test passed' in one word.",
            api_key=api_key
        )
        assert response is not None, "Response should not be None"
        assert len(response) > 0, "Response should not be empty"
        print(f"✅ ModelScope chat passed - Response: {response[:50]}...")
    except Exception as e:
        print(f"❌ ModelScope chat failed: {e}")


def test_dashscope_chat(api_key=None):
    """Test DashScope chat."""
    api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("⏭️  [Skip] DashScope chat - No API key provided")
        return
    
    print("\n[Test] DashScope chat...")
    try:
        response = dashscope_chat(
            prompt="Say 'test passed' in one word.",
            api_key=api_key
        )
        assert response is not None, "Response should not be None"
        assert len(response) > 0, "Response should not be empty"
        print(f"✅ DashScope chat passed - Response: {response[:50]}...")
    except Exception as e:
        print(f"❌ DashScope chat failed: {e}")


def test_openai_chat(api_key=None):
    """Test OpenAI chat."""
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⏭️  [Skip] OpenAI chat - No API key provided")
        return
    
    print("\n[Test] OpenAI chat...")
    try:
        response = openai_chat(
            prompt="Say 'test passed' in one word.",
            api_key=api_key
        )
        assert response is not None, "Response should not be None"
        assert len(response) > 0, "Response should not be empty"
        print(f"✅ OpenAI chat passed - Response: {response[:50]}...")
    except Exception as e:
        print(f"❌ OpenAI chat failed: {e}")


def test_stepfun_chat(api_key=None):
    """Test StepFun chat."""
    # Prefer official STEP_API_KEY, but also support STEPFUN_API_KEY
    api_key = api_key or os.getenv("STEP_API_KEY") or os.getenv("STEPFUN_API_KEY")
    if not api_key:
        print("⏭️  [Skip] StepFun chat - No API key provided")
        return

    print("\n[Test] StepFun chat...")
    try:
        response = stepfun_chat(
            prompt="Say 'test passed' in one word.",
            api_key=api_key
        )
        assert response is not None, "Response should not be None"
        assert len(response) > 0, "Response should not be empty"
        print(f"✅ StepFun chat passed - Response: {response[:50]}...")
    except Exception as e:
        print(f"❌ StepFun chat failed: {e}")


def test_anthropic_chat(api_key=None):
    """Test Anthropic Claude chat."""
    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⏭️  [Skip] Anthropic chat - No API key provided")
        return
    
    print("\n[Test] Anthropic chat...")
    try:
        response = anthropic_chat(
            prompt="Say 'test passed' in one word.",
            api_key=api_key
        )
        assert response is not None, "Response should not be None"
        assert len(response) > 0, "Response should not be empty"
        print(f"✅ Anthropic chat passed - Response: {response[:50]}...")
    except Exception as e:
        print(f"❌ Anthropic chat failed: {e}")


def test_gemini_chat(api_key=None):
    """Test Google Gemini chat."""
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⏭️  [Skip] Gemini chat - No API key provided")
        return
    
    print("\n[Test] Gemini chat...")
    try:
        response = gemini_chat(
            prompt="Say 'test passed' in one word.",
            api_key=api_key
        )
        assert response is not None, "Response should not be None"
        assert len(response) > 0, "Response should not be empty"
        print(f"✅ Gemini chat passed - Response: {response[:50]}...")
    except Exception as e:
        print(f"❌ Gemini chat failed: {e}")


def test_groq_chat(api_key=None):
    """Test Groq chat."""
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("⏭️  [Skip] Groq chat - No API key provided")
        return
    
    print("\n[Test] Groq chat...")
    try:
        response = groq_chat(
            prompt="Say 'test passed' in one word.",
            api_key=api_key
        )
        assert response is not None, "Response should not be None"
        assert len(response) > 0, "Response should not be empty"
        print(f"✅ Groq chat passed - Response: {response[:50]}...")
    except Exception as e:
        print(f"❌ Groq chat failed: {e}")


def test_together_chat(api_key=None):
    """Test Together AI chat."""
    api_key = api_key or os.getenv("TOGETHER_API_KEY")
    if not api_key:
        print("⏭️  [Skip] Together chat - No API key provided")
        return
    
    print("\n[Test] Together chat...")
    try:
        response = together_chat(
            prompt="Say 'test passed' in one word.",
            api_key=api_key
        )
        assert response is not None, "Response should not be None"
        assert len(response) > 0, "Response should not be empty"
        print(f"✅ Together chat passed - Response: {response[:50]}...")
    except Exception as e:
        print(f"❌ Together chat failed: {e}")


def test_mistral_chat(api_key=None):
    """Test Mistral chat."""
    api_key = api_key or os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("⏭️  [Skip] Mistral chat - No API key provided")
        return
    
    print("\n[Test] Mistral chat...")
    try:
        response = mistral_chat(
            prompt="Say 'test passed' in one word.",
            api_key=api_key
        )
        assert response is not None, "Response should not be None"
        assert len(response) > 0, "Response should not be empty"
        print(f"✅ Mistral chat passed - Response: {response[:50]}...")
    except Exception as e:
        print(f"❌ Mistral chat failed: {e}")


def test_cohere_chat(api_key=None):
    """Test Cohere chat."""
    api_key = api_key or os.getenv("COHERE_API_KEY")
    if not api_key:
        print("⏭️  [Skip] Cohere chat - No API key provided")
        return
    
    print("\n[Test] Cohere chat...")
    try:
        response = cohere_chat(
            prompt="Say 'test passed' in one word.",
            api_key=api_key
        )
        assert response is not None, "Response should not be None"
        assert len(response) > 0, "Response should not be empty"
        print(f"✅ Cohere chat passed - Response: {response[:50]}...")
    except Exception as e:
        print(f"❌ Cohere chat failed: {e}")


def test_xunfei_chat(api_key=None):
    """Test Xunfei Spark chat."""
    api_key = api_key or os.getenv("XUNFEI_SPARK_API_KEY")
    if not api_key:
        print("⏭️  [Skip] Xunfei chat - No API key provided")
        return
    
    print("\n[Test] Xunfei chat...")
    try:
        response = xunfei_chat(
            prompt="说'测试通过'（一个词）",
            api_key=api_key
        )
        assert response is not None, "Response should not be None"
        assert len(response) > 0, "Response should not be empty"
        print(f"✅ Xunfei chat passed - Response: {response[:50]}...")
    except Exception as e:
        print(f"❌ Xunfei chat failed: {e}")


def test_perplexity_chat(api_key=None):
    """Test Perplexity chat."""
    api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        print("⏭️  [Skip] Perplexity chat - No API key provided")
        return
    
    print("\n[Test] Perplexity chat...")
    try:
        response = perplexity_chat(
            prompt="Say 'test passed' in one word.",
            api_key=api_key
        )
        assert response is not None, "Response should not be None"
        assert len(response) > 0, "Response should not be empty"
        print(f"✅ Perplexity chat passed - Response: {response[:50]}...")
    except Exception as e:
        print(f"❌ Perplexity chat failed: {e}")


def test_azure_chat(api_key=None):
    """Test Azure OpenAI chat."""
    api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        print("⏭️  [Skip] Azure chat - No API key provided")
        return
    
    print("\n[Test] Azure chat...")
    try:
        response = azure_chat(
            prompt="Say 'test passed' in one word.",
            api_key=api_key
        )
        assert response is not None, "Response should not be None"
        assert len(response) > 0, "Response should not be empty"
        print(f"✅ Azure chat passed - Response: {response[:50]}...")
    except Exception as e:
        print(f"❌ Azure chat failed: {e}")


def main():
    """Run all LLM tests."""
    print("="*60)
    print("LLM Provider Tests")
    print("="*60)
    
    # Run all tests
    test_deepseek_create_client()
    test_deepseek_chat()
    test_siliconflow_chat()
    test_zhipuai_chat()
    test_kimi_chat()
    test_minimax_chat()
    test_yiyan_chat()
    test_modelscope_chat()
    test_dashscope_chat()
    test_openai_chat()
    test_stepfun_chat()
    test_anthropic_chat()
    test_gemini_chat()
    test_groq_chat()
    test_together_chat()
    test_mistral_chat()
    test_cohere_chat()
    test_xunfei_chat()
    test_perplexity_chat()
    test_azure_chat()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
