"""
AI Ethics Auditor Sentinel - Professional Ethics Auditing Framework
====================================================================

A comprehensive, production-ready framework for auditing AI systems for ethical
violations, bias detection, and safety vulnerabilities.

Key Features:
- Comprehensive bias detection across multiple protected attributes
- Safety vulnerability scanning for AI models
- Extensible taxonomy-based ethics framework
- Professional HTML reporting with visualizations
- Modular architecture for easy integration
- Production-grade error handling and logging

Example Usage:
    >>> from sentinel import EthicsAuditor
    >>> auditor = EthicsAuditor(config_path="configs/comprehensive.yaml")
    >>> results = auditor.audit_model(model, dataset)
    >>> auditor.generate_report(results, "audit_report.html")

Author: Dev-Accelerator (AGI Ecosystem)
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Dev-Accelerator"
__email__ = "dev@agi-ecosystem.org"
__license__ = "MIT"

# Core imports for public API
from sentinel.core.auditor import EthicsAuditor
from sentinel.core.taxonomy_loader import EthicsTaxonomyLoader
from sentinel.auditors.bias_detector import BiasDetector
from sentinel.auditors.safety_scanner import SafetyScanner
from sentinel.reporters.html_reporter import HTMLReporter

# Configuration classes
from sentinel.core.config import AuditConfig, BiasConfig, SafetyConfig

# Result classes
from sentinel.core.results import (
    AuditResult,
    BiasResult,
    SafetyResult,
    EthicsViolation,
    RiskLevel
)

# Exceptions
from sentinel.core.exceptions import (
    SentinelError,
    ConfigurationError,
    AuditError,
    ModelCompatibilityError
)

__all__ = [
    # Core classes
    "EthicsAuditor",
    "EthicsTaxonomyLoader",
    "BiasDetector",
    "SafetyScanner", 
    "HTMLReporter",
    
    # Configuration
    "AuditConfig",
    "BiasConfig",
    "SafetyConfig",
    
    # Results
    "AuditResult",
    "BiasResult",
    "SafetyResult",
    "EthicsViolation",
    "RiskLevel",
    
    # Exceptions
    "SentinelError",
    "ConfigurationError",
    "AuditError",
    "ModelCompatibilityError",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__"
]

# Package-level configuration
import logging
from pathlib import Path

# Set up package-level logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Package paths
PACKAGE_ROOT = Path(__file__).parent
CONFIG_DIR = PACKAGE_ROOT / "configs"
TEMPLATES_DIR = PACKAGE_ROOT / "templates"
RESOURCES_DIR = PACKAGE_ROOT / "resources"

# Ensure directories exist
CONFIG_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)
RESOURCES_DIR.mkdir(exist_ok=True)