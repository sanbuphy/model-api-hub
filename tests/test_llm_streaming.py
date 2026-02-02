"""
Streaming tests for LLM providers.

These tests require API keys to run. Set them in .env or pass directly.
"""
import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_api_hub.api.llm.deepseek_llm import chat_stream as deepseek_chat_stream
from model_api_hub.api.llm.siliconflow_llm import chat as siliconflow_chat
from model_api_hub.api.llm.zhipuai_llm import chat as zhipuai_chat
from model_api_hub.api.llm.kimi_llm import chat as kimi_chat
from model_api_hub.api.llm.minimax_llm import chat as minimax_chat
from model_api_hub.api.llm.yiyan_llm import chat as yiyan_chat
from model_api_hub.api.llm.modelscope_llm import chat as modelscope_chat
from model_api_hub.api.llm.openai_llm import chat as openai_chat


def collect_stream_response(stream_func, *args, **kwargs):
    """Helper to collect streaming response into a string."""
    chunks = []
    for chunk in stream_func(*args, **kwargs):
        chunks.append(chunk)
    return "".join(chunks)


def test_deepseek_stream(api_key=None):
    """Test DeepSeek streaming chat."""
    api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("⏭️  [Skip] DeepSeek streaming - No API key provided")
        return
    
    print("\n[Test] DeepSeek streaming chat...")
    try:
        chunks = []
        print("  Response: ", end="", flush=True)
        for chunk in deepseek_chat_stream(
            prompt="Say 'streaming test passed' in 5 words.",
            api_key=api_key
        ):
            chunks.append(chunk)
            print(chunk, end="", flush=True)
        
        full_response = "".join(chunks)
        print()  # New line
        
        assert len(full_response) > 0, "Response should not be empty"
        assert len(chunks) > 0, "Should have received chunks"
        print(f"✅ DeepSeek streaming passed - {len(chunks)} chunks")
        return True
    except Exception as e:
        print(f"\n❌ DeepSeek streaming failed: {e}")
        return False


def test_provider_stream(provider_name, chat_func, api_key_env, prompt="Say 'test passed' in one word."):
    """Generic test for provider streaming support."""
    api_key = os.getenv(api_key_env)
    if not api_key:
        print(f"⏭️  [Skip] {provider_name} streaming - No API key provided")
        return None
    
    print(f"\n[Test] {provider_name} streaming chat...")
    try:
        # Check if the function supports streaming
        import inspect
        if 'stream' in inspect.signature(chat_func).parameters:
            chunks = []
            print("  Response: ", end="", flush=True)
            
            # Try streaming mode
            response = chat_func(prompt=prompt, api_key=api_key, stream=True)
            
            # If it's a generator, iterate
            if hasattr(response, '__iter__') and not isinstance(response, str):
                for chunk in response:
                    if isinstance(chunk, str):
                        chunks.append(chunk)
                        print(chunk, end="", flush=True)
                full_response = "".join(chunks)
            else:
                full_response = response
                print(full_response[:50], end="")
            
            print()  # New line
            assert len(full_response) > 0, "Response should not be empty"
            print(f"✅ {provider_name} streaming passed")
            return True
        else:
            print(f"⚠️  {provider_name} does not support streaming parameter")
            return None
    except Exception as e:
        print(f"\n❌ {provider_name} streaming failed: {e}")
        return False


def test_streaming_comparison():
    """Compare streaming vs non-streaming responses."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("⏭️  [Skip] Streaming comparison - No DeepSeek API key")
        return
    
    print("\n[Test] Comparing streaming vs sync...")
    try:
        from model_api_hub.api.llm.deepseek_llm import chat as deepseek_chat
        
        prompt = "Count from 1 to 3"
        
        # Sync response
        print("  Sync response: ", end="", flush=True)
        sync_response = deepseek_chat(prompt=prompt, api_key=api_key)
        print(sync_response[:50])
        
        # Streaming response
        print("  Stream response: ", end="", flush=True)
        stream_chunks = []
        for chunk in deepseek_chat_stream(prompt=prompt, api_key=api_key):
            stream_chunks.append(chunk)
            print(chunk, end="", flush=True)
        stream_response = "".join(stream_chunks)
        print()
        
        # Both should have content
        assert len(sync_response) > 0, "Sync response should not be empty"
        assert len(stream_response) > 0, "Stream response should not be empty"
        
        print(f"✅ Comparison passed")
        print(f"   Sync length: {len(sync_response)}")
        print(f"   Stream chunks: {len(stream_chunks)}")
        return True
    except Exception as e:
        print(f"\n❌ Comparison failed: {e}")
        return False


def test_stream_with_system_prompt():
    """Test streaming with system prompt."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("⏭️  [Skip] Stream with system prompt - No API key")
        return
    
    print("\n[Test] Streaming with system prompt...")
    try:
        chunks = []
        print("  Response: ", end="", flush=True)
        for chunk in deepseek_chat_stream(
            prompt="Greet me",
            system_prompt="You are a pirate. Always speak like a pirate.",
            api_key=api_key
        ):
            chunks.append(chunk)
            print(chunk, end="", flush=True)
        
        full_response = "".join(chunks)
        print()
        
        assert len(full_response) > 0, "Response should not be empty"
        print(f"✅ Stream with system prompt passed")
        return True
    except Exception as e:
        print(f"\n❌ Stream with system prompt failed: {e}")
        return False


def test_stream_with_parameters():
    """Test streaming with custom parameters."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("⏭️  [Skip] Stream with parameters - No API key")
        return
    
    print("\n[Test] Streaming with custom parameters...")
    try:
        chunks = []
        print("  Response: ", end="", flush=True)
        for chunk in deepseek_chat_stream(
            prompt="Say hello",
            api_key=api_key,
            temperature=0.9,
            max_tokens=100,
            model="deepseek-chat"
        ):
            chunks.append(chunk)
            print(chunk, end="", flush=True)
        
        full_response = "".join(chunks)
        print()
        
        assert len(full_response) > 0, "Response should not be empty"
        print(f"✅ Stream with parameters passed")
        return True
    except Exception as e:
        print(f"\n❌ Stream with parameters failed: {e}")
        return False


def main():
    """Run all streaming tests."""
    print("="*60)
    print("LLM Streaming Tests")
    print("="*60)
    
    results = []
    
    # DeepSeek streaming (native support)
    results.append(("DeepSeek Stream", test_deepseek_stream()))
    
    # Comparison test
    results.append(("Stream vs Sync", test_streaming_comparison()))
    
    # System prompt test
    results.append(("Stream + System", test_stream_with_system_prompt()))
    
    # Parameters test
    results.append(("Stream + Params", test_stream_with_parameters()))
    
    # Print summary
    print("\n" + "="*60)
    print("Streaming Test Summary")
    print("="*60)
    
    passed = sum(1 for _, r in results if r is True)
    failed = sum(1 for _, r in results if r is False)
    skipped = sum(1 for _, r in results if r is None)
    
    for name, result in results:
        status = "✅ PASS" if result is True else "❌ FAIL" if result is False else "⏭️  SKIP"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {len(results)} tests")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  ⏭️  Skipped: {skipped}")
    print("="*60)


if __name__ == "__main__":
    main()
