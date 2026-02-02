"""
Tests for VLM providers.

These tests require API keys to run. Set them in .env or pass directly.
"""

import pytest
import os
from model_api_hub.api.vlm.siliconflow_vlm import analyze_image
from model_api_hub.api.vlm.yiyan_vlm import analyze_image as yiyan_analyze_image


class TestSiliconFlowVLM:
    """Test SiliconFlow VLM provider."""

    @pytest.mark.skip(reason="Requires real API key and test image")
    def test_analyze_image_with_key(self):
        """Test image analysis with API key."""
        # Create a simple test image if needed
        test_image = "test_image.jpg"

        if not os.path.exists(test_image):
            pytest.skip(f"Test image not found: {test_image}")

        response = analyze_image(
            image_path=test_image,
            prompt="What color is this image?",
            api_key="your_api_key_here"
        )
        assert response is not None
        assert len(response) > 0


class TestYiyanVLM:
    """Test Yiyan VLM provider."""

    @pytest.mark.skip(reason="Requires real API key and test image")
    def test_analyze_image_with_key(self):
        """Test image analysis with API key."""
        test_image = "test_image.jpg"

        if not os.path.exists(test_image):
            pytest.skip(f"Test image not found: {test_image}")

        response = yiyan_analyze_image(
            image_path=test_image,
            prompt="这张图是什么颜色的？",
            api_key="your_api_key_here"
        )
        assert response is not None
        assert len(response) > 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test VLM providers")
    parser.add_argument("--api-key", help="API key for testing")
    parser.add_argument("--image", default="test_image.jpg", help="Path to test image")
    args = parser.parse_args()

    if args.api_key:
        print(f"Testing SiliconFlow VLM with image: {args.image}")
        response = analyze_image(args.image, "Describe this image", api_key=args.api_key)
        print(f"Response: {response}")

        print("\nTest passed!")
    else:
        print("Usage: python test_vlm.py --api-key your_key --image path/to/image.jpg")
        pytest.main([__file__, "-v"])
