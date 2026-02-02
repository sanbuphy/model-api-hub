"""
Tests for Audio (TTS) providers.

These tests require API keys to run.
"""

import pytest
import os
from model_api_hub.api.audio.elevenlabs_tts import text_to_speech as elevenlabs_tts
from model_api_hub.api.audio.openai_tts import text_to_speech as openai_tts


class TestElevenLabs:
    """Test ElevenLabs TTS provider."""

    @pytest.mark.skip(reason="Requires real API key")
    def test_text_to_speech_with_key(self):
        """Test text to speech with API key."""
        output_path = "test_output.mp3"

        success = elevenlabs_tts(
            text="This is a test.",
            output_path=output_path,
            api_key="your_api_key_here"
        )

        assert success is True
        assert os.path.exists(output_path)

        # Cleanup
        if os.path.exists(output_path):
            os.remove(output_path)


class TestOpenAITTS:
    """Test OpenAI TTS provider."""

    @pytest.mark.skip(reason="Requires real API key")
    def test_text_to_speech_with_key(self):
        """Test text to speech with API key."""
        output_path = "test_output_openai.mp3"

        success = openai_tts(
            text="This is a test.",
            output_path=output_path,
            api_key="your_api_key_here"
        )

        assert success is True
        assert os.path.exists(output_path)

        # Cleanup
        if os.path.exists(output_path):
            os.remove(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Audio (TTS) providers")
    parser.add_argument("--api-key", help="API key for testing")
    args = parser.parse_args()

    if args.api_key:
        print("Testing ElevenLabs TTS...")
        success = elevenlabs_tts(
            text="Hello, this is a test of the text to speech API.",
            output_path="test_output.mp3",
            api_key=args.api_key
        )

        if success:
            print("✓ Audio generated successfully!")
            print("Output: test_output.mp3")
        else:
            print("✗ TTS failed")
    else:
        pytest.main([__file__, "-v"])
