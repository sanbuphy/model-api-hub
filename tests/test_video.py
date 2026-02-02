"""
Tests for Video Generation providers.

These tests require API keys to run and may take several minutes.
"""

import pytest
import os
from model_api_hub.api.video.runway_gen import generate_video as runway_generate
from model_api_hub.api.video.luma_gen import generate_video as luma_generate


class TestRunwayVideo:
    """Test Runway Video Generation provider."""

    @pytest.mark.skip(reason="Requires real API key and takes time")
    def test_generate_video_with_key(self):
        """Test video generation with API key."""
        output_path = "test_output.mp4"

        success = runway_generate(
            prompt="A peaceful sunset over a calm lake",
            output_path=output_path,
            api_key="your_api_key_here",
            duration=5
        )

        assert success is True
        assert os.path.exists(output_path)

        # Cleanup
        if os.path.exists(output_path):
            os.remove(output_path)


class TestLumaVideo:
    """Test Luma Video Generation provider."""

    @pytest.mark.skip(reason="Requires real API key and takes time")
    def test_generate_video_with_key(self):
        """Test video generation with API key."""
        output_path = "test_output_luma.mp4"

        success = luma_generate(
            prompt="A peaceful sunset over a calm lake",
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

    parser = argparse.ArgumentParser(description="Test Video Generation providers")
    parser.add_argument("--api-key", help="API key for testing")
    args = parser.parse_args()

    if args.api_key:
        print("Testing Runway Video Generation...")
        print("This may take several minutes...")
        success = runway_generate(
            prompt="A peaceful sunset over a calm lake",
            output_path="test_video.mp4",
            api_key=args.api_key
        )

        if success:
            print("✓ Video generated successfully!")
            print("Output: test_video.mp4")
        else:
            print("✗ Video generation failed")
    else:
        print("Usage: python test_video.py --api-key your_key")
        print("Note: Video generation tests take several minutes")
        pytest.main([__file__, "-v"])
