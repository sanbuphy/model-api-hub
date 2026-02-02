"""
Tests for Image Generation providers.

These tests require API keys to run.
"""

import pytest
import os
from model_api_hub.api.image.siliconflow_image_gen import text_to_image
from model_api_hub.api.image.recraft_image_gen import text_to_image as recraft_text_to_image


class TestSiliconFlowImage:
    """Test SiliconFlow Image Generation provider."""

    @pytest.mark.skip(reason="Requires real API key")
    def test_text_to_image_with_key(self):
        """Test text to image generation with API key."""
        output_path = "test_output.png"

        success = text_to_image(
            prompt="A simple red circle on white background",
            output_path=output_path,
            api_key="your_api_key_here",
            image_size="1024x1024"
        )

        assert success is True
        assert os.path.exists(output_path)

        # Cleanup
        if os.path.exists(output_path):
            os.remove(output_path)


class TestRecraftImage:
    """Test Recraft Image Generation provider."""

    @pytest.mark.skip(reason="Requires real API key")
    def test_text_to_image_with_key(self):
        """Test text to image generation with API key."""
        output_path = "test_output_recraft.png"

        success = recraft_text_to_image(
            prompt="A simple blue square on white background",
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

    parser = argparse.ArgumentParser(description="Test Image Generation providers")
    parser.add_argument("--api-key", help="API key for testing")
    args = parser.parse_args()

    if args.api_key:
        print("Testing SiliconFlow Image Generation...")
        success = text_to_image(
            prompt="A beautiful sunset",
            output_path="test_sunset.png",
            api_key=args.api_key
        )

        if success:
            print("✓ Image generated successfully!")
            print("Output: test_sunset.png")
        else:
            print("✗ Image generation failed")
    else:
        pytest.main([__file__, "-v"])
