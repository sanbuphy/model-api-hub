"""
Model API Hub - Unified Python interface for multiple AI model APIs
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Define requirements directly
requirements = [
    "requests>=2.31.0",
    "openai>=1.0.0",
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0",
    "Pillow>=10.0.0",
    "pydub>=0.25.0",
]

setup(
    name="model-api-hub",
    version="0.1.1",
    author="sanbu",
    author_email="physicoada@gmail.com",
    description="Unified Python interface for multiple AI model APIs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sanbuphy/model-api-hub",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "model-api-hub=model_api_hub.cli:main",
            "modelhub=model_api_hub.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="ai api llm vlm image-generation audio video openai anthropic deepseek",
)
