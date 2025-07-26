"""
Ethics Taxonomy Loader - Comprehensive ethics framework integration.
Supports multiple ethics frameworks and custom taxonomies.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from loguru import logger

from sentinel.core.exceptions import ConfigurationError


@dataclass
class EthicsCategory:
    """Represents a single ethics category in the taxonomy."""
    
    name: str
    description: str
    severity_level: str  # low, medium, high, critical
    subcategories: List[str] = field(default_factory=list)
    detection_methods: List[str] = field(default_factory=list)
    remediation_strategies: List[str] = field(default_factory=list)
    regulatory_references: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate severity level."""
        valid_levels = ["low", "medium", "high", "critical"]
        if self.severity_level not in valid_levels:
            raise ValueError(f"Invalid severity level: {self.severity_level}. Must be one of {valid_levels}")


@dataclass
class EthicsTaxonomy:
    """Complete ethics taxonomy structure."""
    
    name: str
    version: str
    description: str
    categories: Dict[str, EthicsCategory] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_category(self, category_name: str) -> Optional[EthicsCategory]:
        """Get a specific ethics category."""
        return self.categories.get(category_name)
    
    def get_categories_by_severity(self, severity: str) -> List[EthicsCategory]:
        """Get all categories matching a specific severity level."""
        return [cat for cat in self.categories.values() if cat.severity_level == severity]
    
    def list_all_categories(self) -> List[str]:
        """Get list of all category names."""
        return list(self.categories.keys())


class EthicsTaxonomyLoader:
    """
    Loads and manages ethics taxonomies from various sources.
    
    Supports multiple ethics frameworks including:
    - IEEE Standards for AI Ethics
    - ACM Code of Ethics
    - Partnership on AI Principles
    - Custom organizational taxonomies
    """
    
    def __init__(self):
        """Initialize the taxonomy loader."""
        self.taxonomies: Dict[str, EthicsTaxonomy] = {}
        self.active_taxonomy: Optional[EthicsTaxonomy] = None
        
        # Load default taxonomy
        self._load_default_taxonomy()
    
    def _load_default_taxonomy(self) -> None:
        """Load the default comprehensive ethics taxonomy."""
        default_taxonomy = self._create_default_taxonomy()
        self.taxonomies["default"] = default_taxonomy
        self.active_taxonomy = default_taxonomy
        logger.info("Loaded default ethics taxonomy")
    
    def _create_default_taxonomy(self) -> EthicsTaxonomy:
        """Create a comprehensive default ethics taxonomy."""
        categories = {
            # Bias and Fairness
            "demographic_bias": EthicsCategory(
                name="Demographic Bias",
                description="Systematic unfair treatment based on demographic characteristics",
                severity_level="high",
                subcategories=[
                    "racial_bias", "gender_bias", "age_bias", "religious_bias",
                    "disability_bias", "sexual_orientation_bias"
                ],
                detection_methods=[
                    "statistical_parity_test", "equalized_odds_test", 
                    "demographic_parity_test", "individual_fairness_test"
                ],
                remediation_strategies=[
                    "data_augmentation", "algorithmic_debiasing", 
                    "post_processing_calibration", "adversarial_debiasing"
                ],
                regulatory_references=[
                    "EU_AI_Act", "GDPR_Article_22", "US_Equal_Credit_Opportunity_Act"
                ]
            ),
            
            "algorithmic_bias": EthicsCategory(
                name="Algorithmic Bias",
                description="Bias introduced by algorithmic design choices",
                severity_level="high",
                subcategories=[
                    "selection_bias", "confirmation_bias", "automation_bias",
                    "representation_bias", "measurement_bias"
                ],
                detection_methods=[
                    "bias_variance_analysis", "feature_importance_analysis",
                    "model_interpretation", "counterfactual_analysis"
                ],
                remediation_strategies=[
                    "algorithm_redesign", "ensemble_methods", 
                    "regularization_techniques", "cross_validation"
                ],
                regulatory_references=["IEEE_2857", "ISO_23053"]
            ),
            
            # Privacy and Data Protection
            "privacy_violation": EthicsCategory(
                name="Privacy Violation",
                description="Unauthorized access to or misuse of personal information",
                severity_level="critical",
                subcategories=[
                    "data_leakage", "re_identification", "inference_attacks",
                    "membership_inference", "model_inversion"
                ],
                detection_methods=[
                    "differential_privacy_audit", "k_anonymity_check",
                    "membership_inference_test", "model_inversion_test"
                ],
                remediation_strategies=[
                    "differential_privacy", "federated_learning",
                    "homomorphic_encryption", "secure_multiparty_computation"
                ],
                regulatory_references=["GDPR", "CCPA", "PIPEDA", "LGPD"]
            ),
            
            # Safety and Security
            "adversarial_vulnerability": EthicsCategory(
                name="Adversarial Vulnerability",
                description="Susceptibility to adversarial attacks and manipulation",
                severity_level="high",
                subcategories=[
                    "evasion_attacks", "poisoning_attacks", "model_extraction",
                    "backdoor_attacks", "prompt_injection"
                ],
                detection_methods=[
                    "adversarial_example_generation", "robustness_testing",
                    "attack_simulation", "gradient_analysis"
                ],
                remediation_strategies=[
                    "adversarial_training", "input_validation",
                    "model_hardening", "ensemble_defense"
                ],
                regulatory_references=["NIST_AI_RMF", "ISO_27001"]
            ),
            
            "safety_risk": EthicsCategory(
                name="Safety Risk",
                description="Potential for physical or psychological harm",
                severity_level="critical",
                subcategories=[
                    "physical_harm", "psychological_harm", "economic_harm",
                    "social_harm", "environmental_harm"
                ],
                detection_methods=[
                    "hazard_analysis", "failure_mode_analysis",
                    "risk_assessment", "safety_testing"
                ],
                remediation_strategies=[
                    "safety_constraints", "fail_safe_mechanisms",
                    "human_oversight", "gradual_deployment"
                ],
                regulatory_references=[
                    "ISO_26262", "IEC_61508", "FDA_Software_Guidance"
                ]
            ),
            
            # Transparency and Explainability  
            "lack_of_transparency": EthicsCategory(
                name="Lack of Transparency",
                description="Insufficient disclosure of system capabilities and limitations",
                severity_level="medium",
                subcategories=[
                    "black_box_decisions", "undisclosed_capabilities",
                    "hidden_biases", "unclear_decision_boundaries"
                ],
                detection_methods=[
                    "interpretability_analysis", "feature_attribution",
                    "model_explanation", "transparency_audit"
                ],
                remediation_strategies=[
                    "explainable_AI", "model_documentation",
                    "decision_logs", "user_education"
                ],
                regulatory_references=["EU_AI_Act_Article_13", "GDPR_Article_22"]
            ),
            
            # Accountability and Governance
            "accountability_gap": EthicsCategory(
                name="Accountability Gap",
                description="Unclear responsibility for AI system decisions and outcomes",
                severity_level="medium",
                subcategories=[
                    "unclear_responsibility", "inadequate_oversight",
                    "insufficient_documentation", "lack_of_recourse"
                ],
                detection_methods=[
                    "governance_audit", "responsibility_mapping",
                    "documentation_review", "oversight_analysis"
                ],
                remediation_strategies=[
                    "clear_governance_structure", "audit_trails",
                    "human_in_the_loop", "appeals_process"
                ],
                regulatory_references=["EU_AI_Act", "UNESCO_AI_Ethics"]
            ),
            
            # Environmental Impact
            "environmental_harm": EthicsCategory(
                name="Environmental Harm",
                description="Negative environmental impact from AI systems",
                severity_level="medium",
                subcategories=[
                    "excessive_energy_consumption", "carbon_footprint",
                    "resource_waste", "electronic_waste"
                ],
                detection_methods=[
                    "energy_consumption_audit", "carbon_footprint_analysis",
                    "lifecycle_assessment", "efficiency_metrics"
                ],
                remediation_strategies=[
                    "model_compression", "efficient_architectures",
                    "green_computing", "renewable_energy"
                ],
                regulatory_references=["EU_Green_Deal", "UN_SDG"]
            )
        }
        
        return EthicsTaxonomy(
            name="Comprehensive AI Ethics Taxonomy",
            version="1.0.0",
            description="Professional-grade taxonomy covering all major AI ethics concerns",
            categories=categories,
            metadata={
                "created_by": "AI Ethics Auditor Sentinel",
                "standards_compliance": [
                    "IEEE Standards", "ISO Standards", "EU AI Act",
                    "ACM Code of Ethics", "Partnership on AI"
                ],
                "last_updated": "2024-01-01",
                "coverage_areas": [
                    "Bias and Fairness", "Privacy and Data Protection",
                    "Safety and Security", "Transparency and Explainability",
                    "Accountability and Governance", "Environmental Impact"
                ]
            }
        )
    
    def load_taxonomy(
        self,
        taxonomy_path: Union[str, Path],
        taxonomy_name: Optional[str] = None
    ) -> EthicsTaxonomy:
        """
        Load taxonomy from file.
        
        Args:
            taxonomy_path: Path to taxonomy file (JSON or YAML)
            taxonomy_name: Optional name for the taxonomy
            
        Returns:
            EthicsTaxonomy: Loaded taxonomy
        """
        taxonomy_path = Path(taxonomy_path)
        
        if not taxonomy_path.exists():
            raise ConfigurationError(f"Taxonomy file not found: {taxonomy_path}")
        
        try:
            with open(taxonomy_path, 'r', encoding='utf-8') as f:
                if taxonomy_path.suffix.lower() == '.json':
                    data = json.load(f)
                elif taxonomy_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    raise ConfigurationError(f"Unsupported file format: {taxonomy_path.suffix}")
            
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ConfigurationError(f"Invalid file format: {e}")
        
        # Parse taxonomy data
        taxonomy = self._parse_taxonomy_data(data)
        
        # Store taxonomy
        name = taxonomy_name or taxonomy.name or taxonomy_path.stem
        self.taxonomies[name] = taxonomy
        
        logger.info(f"Loaded taxonomy '{name}' with {len(taxonomy.categories)} categories")
        return taxonomy
    
    def _parse_taxonomy_data(self, data: Dict[str, Any]) -> EthicsTaxonomy:
        """Parse taxonomy data from dictionary."""
        try:
            # Parse categories
            categories = {}
            for cat_name, cat_data in data.get("categories", {}).items():
                categories[cat_name] = EthicsCategory(
                    name=cat_data.get("name", cat_name),
                    description=cat_data["description"],
                    severity_level=cat_data["severity_level"],
                    subcategories=cat_data.get("subcategories", []),
                    detection_methods=cat_data.get("detection_methods", []),
                    remediation_strategies=cat_data.get("remediation_strategies", []),
                    regulatory_references=cat_data.get("regulatory_references", [])
                )
            
            return EthicsTaxonomy(
                name=data.get("name", "Custom Taxonomy"),
                version=data.get("version", "1.0.0"),
                description=data.get("description", ""),
                categories=categories,
                metadata=data.get("metadata", {})
            )
            
        except KeyError as e:
            raise ConfigurationError(f"Missing required field in taxonomy: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error parsing taxonomy: {e}")
    
    def set_active_taxonomy(self, taxonomy_name: str) -> None:
        """Set the active taxonomy by name."""
        if taxonomy_name not in self.taxonomies:
            raise ConfigurationError(f"Taxonomy '{taxonomy_name}' not found")
        
        self.active_taxonomy = self.taxonomies[taxonomy_name]
        logger.info(f"Set active taxonomy to '{taxonomy_name}'")
    
    def get_active_taxonomy(self) -> Optional[EthicsTaxonomy]:
        """Get the currently active taxonomy."""
        return self.active_taxonomy
    
    def list_taxonomies(self) -> List[str]:
        """List all loaded taxonomy names."""
        return list(self.taxonomies.keys())
    
    def get_violation_severity(self, violation_type: str) -> Optional[str]:
        """Get severity level for a specific violation type."""
        if not self.active_taxonomy:
            return None
        
        category = self.active_taxonomy.get_category(violation_type)
        return category.severity_level if category else None
    
    def get_detection_methods(self, violation_type: str) -> List[str]:
        """Get detection methods for a specific violation type."""
        if not self.active_taxonomy:
            return []
        
        category = self.active_taxonomy.get_category(violation_type)
        return category.detection_methods if category else []
    
    def get_remediation_strategies(self, violation_type: str) -> List[str]:
        """Get remediation strategies for a specific violation type."""
        if not self.active_taxonomy:
            return []
        
        category = self.active_taxonomy.get_category(violation_type)
        return category.remediation_strategies if category else []
    
    def export_taxonomy(
        self,
        taxonomy_name: str,
        output_path: Union[str, Path],
        format_type: str = "yaml"
    ) -> None:
        """
        Export taxonomy to file.
        
        Args:
            taxonomy_name: Name of taxonomy to export
            output_path: Output file path
            format_type: Export format ("json" or "yaml")
        """
        if taxonomy_name not in self.taxonomies:
            raise ConfigurationError(f"Taxonomy '{taxonomy_name}' not found")
        
        taxonomy = self.taxonomies[taxonomy_name]
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to exportable format
        export_data = {
            "name": taxonomy.name,
            "version": taxonomy.version,
            "description": taxonomy.description,
            "metadata": taxonomy.metadata,
            "categories": {}
        }
        
        for cat_name, category in taxonomy.categories.items():
            export_data["categories"][cat_name] = {
                "name": category.name,
                "description": category.description,
                "severity_level": category.severity_level,
                "subcategories": category.subcategories,
                "detection_methods": category.detection_methods,
                "remediation_strategies": category.remediation_strategies,
                "regulatory_references": category.regulatory_references
            }
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            if format_type.lower() == "json":
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            else:  # yaml
                yaml.dump(export_data, f, default_flow_style=False, indent=2)
        
        logger.info(f"Exported taxonomy '{taxonomy_name}' to {output_path}")
    
    def merge_taxonomies(
        self,
        taxonomy_names: List[str],
        merged_name: str,
        conflict_resolution: str = "highest_severity"
    ) -> EthicsTaxonomy:
        """
        Merge multiple taxonomies into one.
        
        Args:
            taxonomy_names: List of taxonomy names to merge
            merged_name: Name for the merged taxonomy
            conflict_resolution: How to resolve conflicts ("highest_severity", "latest", "custom")
            
        Returns:
            EthicsTaxonomy: Merged taxonomy
        """
        if not taxonomy_names:
            raise ConfigurationError("No taxonomies specified for merging")
        
        # Check all taxonomies exist
        for name in taxonomy_names:
            if name not in self.taxonomies:
                raise ConfigurationError(f"Taxonomy '{name}' not found")
        
        # Initialize merged taxonomy
        base_taxonomy = self.taxonomies[taxonomy_names[0]]
        merged_categories = dict(base_taxonomy.categories)
        
        # Merge categories from other taxonomies
        for taxonomy_name in taxonomy_names[1:]:
            taxonomy = self.taxonomies[taxonomy_name]
            
            for cat_name, category in taxonomy.categories.items():
                if cat_name in merged_categories:
                    # Handle conflict
                    if conflict_resolution == "highest_severity":
                        existing_severity = merged_categories[cat_name].severity_level
                        new_severity = category.severity_level
                        
                        severity_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
                        if severity_order.get(new_severity, 0) > severity_order.get(existing_severity, 0):
                            merged_categories[cat_name] = category
                    
                    elif conflict_resolution == "latest":
                        merged_categories[cat_name] = category
                
                else:
                    merged_categories[cat_name] = category
        
        # Create merged taxonomy
        merged_taxonomy = EthicsTaxonomy(
            name=merged_name,
            version="1.0.0",
            description=f"Merged taxonomy from: {', '.join(taxonomy_names)}",
            categories=merged_categories,
            metadata={
                "merged_from": taxonomy_names,
                "conflict_resolution": conflict_resolution,
                "created_by": "EthicsTaxonomyLoader"
            }
        )
        
        # Store merged taxonomy
        self.taxonomies[merged_name] = merged_taxonomy
        
        logger.info(f"Merged {len(taxonomy_names)} taxonomies into '{merged_name}' with {len(merged_categories)} categories")
        return merged_taxonomy