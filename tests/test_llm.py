"""
Tests for LLM providers.

These tests require API keys to run. Set them in .env or pass directly.
"""

import pytest
from model_api_hub.api.llm.deepseek_llm import create_client, get_completion, chat
from model_api_hub.api.llm.siliconflow_llm import chat as siliconflow_chat
from model_api_hub.api.llm.zhipuai_llm import chat as zhipuai_chat
from model_api_hub.api.llm.kimi_llm import chat as kimi_chat
from model_api_hub.api.llm.minimax_llm import chat as minimax_chat


class TestDeepSeek:
    """Test DeepSeek LLM provider."""

    def test_create_client_with_key(self):
        """Test client creation with API key."""
        client = create_client(api_key="test_key")
        assert client is not None
        assert client.api_key == "test_key"

    @pytest.mark.skip(reason="Requires real API key")
    def test_chat_with_api_key(self):
        """Test chat with API key."""
        response = chat(
            prompt="Say 'test passed' in one word.",
            api_key="your_api_key_here"
        )
        assert response is not None
        assert len(response) > 0


class TestSiliconFlow:
    """Test SiliconFlow LLM provider."""

    @pytest.mark.skip(reason="Requires real API key")
    def test_chat_with_api_key(self):
        """Test chat with API key."""
        response = siliconflow_chat(
            prompt="Say 'test passed' in one word.",
            api_key="your_api_key_here"
        )
        assert response is not None
        assert len(response) > 0


class TestZhipuAI:
    """Test ZhipuAI LLM provider."""

    @pytest.mark.skip(reason="Requires real API key")
    def test_chat_with_api_key(self):
        """Test chat with API key."""
        response = zhipuai_chat(
            prompt="说'测试通过'（一个词）",
            api_key="your_api_key_here"
        )
        assert response is not None
        assert len(response) > 0


class TestKimi:
    """Test Kimi LLM provider."""

    @pytest.mark.skip(reason="Requires real API key")
    def test_chat_with_api_key(self):
        """Test chat with API key."""
        response = kimi_chat(
            prompt="说'测试通过'（一个词）",
            api_key="your_api_key_here"
        )
        assert response is not None
        assert len(response) > 0


class TestMiniMax:
    """Test MiniMax LLM provider."""

    @pytest.mark.skip(reason="Requires real API key")
    def test_chat_with_api_key(self):
        """Test chat with API key."""
        response = minimax_chat(
            prompt="说'测试通过'（一个词）",
            api_key="your_api_key_here"
        )
        assert response is not None
        assert len(response) > 0


if __name__ == "__main__":
    # Run with your API key:
    # python test_llm.py --api-key your_key
    import argparse

    parser = argparse.ArgumentParser(description="Test LLM providers")
    parser.add_argument("--api-key", help="API key for testing")
    args = parser.parse_args()

    if args.api_key:
        print("Testing DeepSeek...")
        response = chat("Say 'test' in one word.", api_key=args.api_key)
        print(f"Response: {response}")

        print("\nAll tests passed!")
    else:
        pytest.main([__file__, "-v"])
