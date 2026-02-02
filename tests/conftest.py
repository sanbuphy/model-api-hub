"""
Pytest configuration and fixtures for Model API Hub tests.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may require API keys)"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (video generation, etc.)"
    )


@pytest.fixture
def test_image_path(tmp_path):
    """Create a simple test image."""
    from PIL import Image

    img_path = tmp_path / "test_image.jpg"
    img = Image.new('RGB', (100, 100), color='red')
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def api_key_monkeypatch(monkeypatch):
    """Fixture to set API keys via environment variables."""
    def set_key(provider, key):
        monkeypatch.setenv(f"{provider.upper()}_API_KEY", key)
    return set_key
