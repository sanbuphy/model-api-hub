"""
Utility module for loading configuration and API keys.

Supports loading from environment variables (.env file) and YAML config files.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
import yaml


def load_env(env_path: Optional[str] = None) -> None:
    """
    Load environment variables from .env file.

    Args:
        env_path: Path to .env file. If None, looks for .env in current directory and parent directories.
    """
    if env_path:
        load_dotenv(env_path)
    else:
        load_dotenv()


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing configuration settings.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_api_key(provider: str, env_path: Optional[str] = None) -> str:
    """
    Get API key from environment variables.

    Args:
        provider: Provider name (e.g., "openai", "anthropic", "deepseek").
        env_path: Optional path to .env file.

    Returns:
        API key string.

    Raises:
        ValueError: If API key is not found in environment variables.
    """
    load_env(env_path)

    # Try different naming conventions
    env_var_names = [
        f"{provider.upper()}_API_KEY",
        f"{provider.upper()}_KEY",
        provider.upper(),
    ]

    for env_var in env_var_names:
        api_key = os.getenv(env_var)
        if api_key:
            return api_key

    raise ValueError(
        f"API key for '{provider}' not found. "
        f"Please set {env_var_names[0]} environment variable."
    )


def get_config_value(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """
    Safely get nested configuration value.

    Args:
        config: Configuration dictionary.
        *keys: Sequence of keys to traverse the nested dictionary.
        default: Default value if key path doesn't exist.

    Returns:
        Configuration value or default.

    Example:
        >>> config = {"llm": {"openai": {"model": "gpt-4"}}}
        >>> get_config_value(config, "llm", "openai", "model")
        "gpt-4"
    """
    current = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


class ConfigManager:
    """
    Configuration manager class for handling multiple configuration sources.
    """

    def __init__(self, config_path: str = "config.yaml", env_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to YAML configuration file.
            env_path: Optional path to .env file.
        """
        self.config_path = config_path
        self.env_path = env_path
        self._config: Optional[Dict[str, Any]] = None
        load_env(env_path)

    @property
    def config(self) -> Dict[str, Any]:
        """Lazy load configuration."""
        if self._config is None:
            self._config = load_config(self.config_path)
        return self._config

    def reload(self) -> None:
        """Reload configuration from files."""
        self._config = load_config(self.config_path)
        load_env(self.env_path)

    def get_api_key(self, provider: str) -> str:
        """Get API key for a provider."""
        return get_api_key(provider, self.env_path)

    def get_setting(self, *keys: str, default: Any = None) -> Any:
        """Get configuration setting by key path."""
        return get_config_value(self.config, *keys, default=default)

    def get_llm_config(self, provider: str) -> Dict[str, Any]:
        """Get LLM configuration for a provider."""
        return get_config_value(self.config, "llm", provider, default={})

    def get_vlm_config(self, provider: str) -> Dict[str, Any]:
        """Get VLM configuration for a provider."""
        return get_config_value(self.config, "vlm", provider, default={})

    def get_image_config(self, provider: str) -> Dict[str, Any]:
        """Get image generation configuration for a provider."""
        return get_config_value(self.config, "image", provider, default={})

    def get_audio_config(self, provider: str) -> Dict[str, Any]:
        """Get audio processing configuration for a provider."""
        return get_config_value(self.config, "audio", provider, default={})

    def get_video_config(self, provider: str) -> Dict[str, Any]:
        """Get video generation configuration for a provider."""
        return get_config_value(self.config, "video", provider, default={})

    def get_aggregator_config(self, provider: str) -> Dict[str, Any]:
        """Get API aggregator configuration for a provider."""
        return get_config_value(self.config, "aggregators", provider, default={})
