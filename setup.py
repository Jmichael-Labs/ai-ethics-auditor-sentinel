#!/usr/bin/env python3
"""
AI Ethics Auditor Sentinel - Setup Configuration
Professional-grade package setup for production deployment.
"""

import re
from pathlib import Path
from setuptools import setup, find_packages

# Read version from package
def get_version():
    """Extract version from package __init__.py"""
    init_file = Path(__file__).parent / "sentinel" / "__init__.py"
    if init_file.exists():
        with open(init_file, "r", encoding="utf-8") as f:
            content = f.read()
            match = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", content)
            if match:
                return match.group(1)
    return "0.1.0"

# Read long description from README
def get_long_description():
    """Get long description from README.md"""
    readme_path = Path(__file__).parent / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "AI Ethics Auditor Sentinel - Professional ethics auditing framework for AI systems"

# Read requirements
def get_requirements():
    """Parse requirements.txt"""
    requirements_path = Path(__file__).parent / "requirements.txt"
    requirements = []
    if requirements_path.exists():
        with open(requirements_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
    return requirements

setup(
    name="ai-ethics-auditor-sentinel",
    version=get_version(),
    author="Dev-Accelerator",
    author_email="dev@agi-ecosystem.org",
    description="Professional-grade AI ethics auditing framework with comprehensive bias detection and safety analysis",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/agi-ecosystem/ai-ethics-auditor-sentinel",
    project_urls={
        "Bug Reports": "https://github.com/agi-ecosystem/ai-ethics-auditor-sentinel/issues",
        "Source": "https://github.com/agi-ecosystem/ai-ethics-auditor-sentinel",
        "Documentation": "https://ai-ethics-auditor-sentinel.readthedocs.io/",
    },
    packages=find_packages(exclude=["tests*", "docs*"]),
    include_package_data=True,
    package_data={
        "sentinel": [
            "configs/*.yaml",
            "templates/*.html",
            "templates/*.css",
            "resources/*.json"
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Security",
        "Typing :: Typed",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.4.0",
            "flake8>=6.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
        "api": [
            "fastapi>=0.100.0",
            "uvicorn>=0.22.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "sentinel-audit=sentinel.cli:main",
            "sentinel-server=sentinel.api:serve",
        ],
    },
    keywords=[
        "ai", "ethics", "bias", "fairness", "machine-learning", 
        "auditing", "security", "safety", "responsible-ai"
    ],
    zip_safe=False,
)