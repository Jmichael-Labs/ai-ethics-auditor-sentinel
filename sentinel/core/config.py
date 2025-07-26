"""
Configuration management for AI Ethics Auditor Sentinel.
Type-safe configuration classes with validation and YAML support.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import yaml
from sentinel.core.exceptions import ConfigurationError


@dataclass
class BiasConfig:
    """Configuration for bias detection."""
    
    # Protected attributes to analyze
    protected_attributes: List[str] = field(default_factory=lambda: [
        "race", "gender", "age", "religion", "disability", "sexual_orientation"
    ])
    
    # Fairness metrics to calculate
    fairness_metrics: List[str] = field(default_factory=lambda: [
        "demographic_parity", "equalized_odds", "calibration", "individual_fairness"
    ])
    
    # Statistical significance thresholds
    significance_threshold: float = 0.05
    bias_threshold: float = 0.1  # Acceptable bias level
    
    # Sample sizes for statistical tests
    min_sample_size: int = 100
    bootstrap_samples: int = 1000
    
    # Advanced options
    intersectional_analysis: bool = True
    temporal_analysis: bool = False
    
    def __post_init__(self):
        """Validate configuration values."""
        if not 0.0 <= self.significance_threshold <= 1.0:
            raise ConfigurationError("significance_threshold must be between 0.0 and 1.0")
        
        if not 0.0 <= self.bias_threshold <= 1.0:
            raise ConfigurationError("bias_threshold must be between 0.0 and 1.0")
        
        if self.min_sample_size < 10:
            raise ConfigurationError("min_sample_size must be at least 10")


@dataclass
class SafetyConfig:
    """Configuration for safety vulnerability scanning."""
    
    # Types of safety tests to run
    vulnerability_tests: List[str] = field(default_factory=lambda: [
        "adversarial_examples", "data_poisoning", "model_inversion", 
        "membership_inference", "prompt_injection", "output_manipulation"
    ])
    
    # Adversarial attack parameters
    adversarial_epsilon: float = 0.1  # Perturbation magnitude
    adversarial_iterations: int = 40
    adversarial_step_size: float = 0.01
    
    # Robustness testing
    noise_levels: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.2])
    robustness_samples: int = 1000
    
    # Safety thresholds
    safety_threshold: float = 0.7  # Minimum safety score
    max_attack_success_rate: float = 0.1
    
    # Privacy testing
    privacy_budget: float = 1.0  # Differential privacy budget
    k_anonymity: int = 5
    
    def __post_init__(self):
        """Validate configuration values."""
        if not 0.0 <= self.adversarial_epsilon <= 1.0:
            raise ConfigurationError("adversarial_epsilon must be between 0.0 and 1.0")
        
        if not 0.0 <= self.safety_threshold <= 1.0:
            raise ConfigurationError("safety_threshold must be between 0.0 and 1.0")
        
        if self.k_anonymity < 2:
            raise ConfigurationError("k_anonymity must be at least 2")


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    
    # Output formats
    formats: List[str] = field(default_factory=lambda: ["html", "json"])
    
    # Report content options
    include_visualizations: bool = True
    include_raw_data: bool = False
    include_recommendations: bool = True
    detailed_explanations: bool = True
    
    # Styling options
    theme: str = "professional"  # professional, academic, minimal
    color_scheme: str = "default"  # default, colorblind, high_contrast
    
    # Output settings
    output_directory: str = "audit_reports"
    filename_template: str = "ethics_audit_{timestamp}_{model_name}"
    
    def __post_init__(self):
        """Validate configuration values."""
        valid_themes = ["professional", "academic", "minimal"]
        if self.theme not in valid_themes:
            raise ConfigurationError(f"theme must be one of {valid_themes}")
        
        valid_color_schemes = ["default", "colorblind", "high_contrast"]
        if self.color_scheme not in valid_color_schemes:
            raise ConfigurationError(f"color_scheme must be one of {valid_color_schemes}")


@dataclass
class AuditConfig:
    """Main configuration class for ethics auditing."""
    
    # Component configurations
    bias: BiasConfig = field(default_factory=BiasConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    
    # General audit settings
    audit_components: List[str] = field(default_factory=lambda: ["bias", "safety"])
    parallel_execution: bool = True
    max_workers: int = 4
    
    # Logging and monitoring
    log_level: str = "INFO"
    enable_progress_tracking: bool = True
    save_intermediate_results: bool = True
    
    # Ethics taxonomy
    taxonomy_path: Optional[str] = None
    custom_taxonomy: Dict[str, Any] = field(default_factory=dict)
    
    # Model compatibility
    supported_model_types: List[str] = field(default_factory=lambda: [
        "sklearn", "pytorch", "tensorflow", "huggingface", "custom"
    ])
    
    def __post_init__(self):
        """Validate configuration values."""
        valid_components = ["bias", "safety"]
        invalid_components = set(self.audit_components) - set(valid_components)
        if invalid_components:
            raise ConfigurationError(f"Invalid audit components: {invalid_components}")
        
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            raise ConfigurationError(f"log_level must be one of {valid_log_levels}")
        
        if self.max_workers < 1:
            raise ConfigurationError("max_workers must be at least 1")
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'AuditConfig':
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {config_path}",
                config_path=str(config_path)
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML format in configuration file: {e}",
                config_path=str(config_path)
            )
        
        return cls.from_dict(config_data, config_path=str(config_path))
    
    @classmethod
    def from_dict(cls, config_data: Dict[str, Any], config_path: Optional[str] = None) -> 'AuditConfig':
        """Create configuration from dictionary."""
        try:
            # Extract nested configurations
            bias_config = BiasConfig(**config_data.get("bias", {}))
            safety_config = SafetyConfig(**config_data.get("safety", {}))
            report_config = ReportConfig(**config_data.get("report", {}))
            
            # Remove nested configs from main data
            main_config_data = {k: v for k, v in config_data.items() 
                              if k not in ["bias", "safety", "report"]}
            
            return cls(
                bias=bias_config,
                safety=safety_config,
                report=report_config,
                **main_config_data
            )
        
        except TypeError as e:
            raise ConfigurationError(
                f"Invalid configuration structure: {e}",
                config_path=config_path
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "bias": {
                "protected_attributes": self.bias.protected_attributes,
                "fairness_metrics": self.bias.fairness_metrics,
                "significance_threshold": self.bias.significance_threshold,
                "bias_threshold": self.bias.bias_threshold,
                "min_sample_size": self.bias.min_sample_size,
                "bootstrap_samples": self.bias.bootstrap_samples,
                "intersectional_analysis": self.bias.intersectional_analysis,
                "temporal_analysis": self.bias.temporal_analysis
            },
            "safety": {
                "vulnerability_tests": self.safety.vulnerability_tests,
                "adversarial_epsilon": self.safety.adversarial_epsilon,
                "adversarial_iterations": self.safety.adversarial_iterations,
                "adversarial_step_size": self.safety.adversarial_step_size,
                "noise_levels": self.safety.noise_levels,
                "robustness_samples": self.safety.robustness_samples,
                "safety_threshold": self.safety.safety_threshold,
                "max_attack_success_rate": self.safety.max_attack_success_rate,
                "privacy_budget": self.safety.privacy_budget,
                "k_anonymity": self.safety.k_anonymity
            },
            "report": {
                "formats": self.report.formats,
                "include_visualizations": self.report.include_visualizations,
                "include_raw_data": self.report.include_raw_data,
                "include_recommendations": self.report.include_recommendations,
                "detailed_explanations": self.report.detailed_explanations,
                "theme": self.report.theme,
                "color_scheme": self.report.color_scheme,
                "output_directory": self.report.output_directory,
                "filename_template": self.report.filename_template
            },
            "audit_components": self.audit_components,
            "parallel_execution": self.parallel_execution,
            "max_workers": self.max_workers,
            "log_level": self.log_level,
            "enable_progress_tracking": self.enable_progress_tracking,
            "save_intermediate_results": self.save_intermediate_results,
            "taxonomy_path": self.taxonomy_path,
            "custom_taxonomy": self.custom_taxonomy,
            "supported_model_types": self.supported_model_types
        }
    
    def save_yaml(self, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)