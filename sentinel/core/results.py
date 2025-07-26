"""
Result classes for AI Ethics Auditor Sentinel.
Comprehensive data structures for audit results with typing and validation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import json
from pathlib import Path


class RiskLevel(Enum):
    """Risk level enumeration for ethics violations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    
    def __lt__(self, other):
        """Enable comparison of risk levels."""
        if not isinstance(other, RiskLevel):
            return NotImplemented
        
        level_order = {
            RiskLevel.LOW: 0,
            RiskLevel.MEDIUM: 1,
            RiskLevel.HIGH: 2,
            RiskLevel.CRITICAL: 3
        }
        return level_order[self] < level_order[other]


@dataclass
class EthicsViolation:
    """Represents a single ethics violation found during audit."""
    
    violation_type: str
    description: str
    risk_level: RiskLevel
    confidence: float  # 0.0 to 1.0
    affected_groups: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    taxonomy_reference: Optional[str] = None
    
    def __post_init__(self):
        """Validate fields after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        
        if not self.description.strip():
            raise ValueError("Description cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "violation_type": self.violation_type,
            "description": self.description,
            "risk_level": self.risk_level.value,
            "confidence": self.confidence,
            "affected_groups": self.affected_groups,
            "evidence": self.evidence,
            "recommendations": self.recommendations,
            "taxonomy_reference": self.taxonomy_reference
        }


@dataclass
class BiasResult:
    """Results from bias detection analysis."""
    
    overall_bias_score: float  # 0.0 (no bias) to 1.0 (maximum bias)
    protected_attributes: List[str]
    violations: List[EthicsViolation] = field(default_factory=list)
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    statistical_tests: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    fairness_metrics: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate bias score range."""
        if not 0.0 <= self.overall_bias_score <= 1.0:
            raise ValueError("Bias score must be between 0.0 and 1.0")
    
    @property
    def has_significant_bias(self) -> bool:
        """Check if significant bias was detected."""
        return self.overall_bias_score > 0.5 or any(
            v.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] 
            for v in self.violations
        )
    
    def get_violations_by_risk(self, risk_level: RiskLevel) -> List[EthicsViolation]:
        """Get violations filtered by risk level."""
        return [v for v in self.violations if v.risk_level == risk_level]


@dataclass 
class SafetyResult:
    """Results from safety vulnerability scanning."""
    
    overall_safety_score: float  # 0.0 (unsafe) to 1.0 (safe)
    vulnerabilities_found: int
    violations: List[EthicsViolation] = field(default_factory=list)
    attack_vectors: List[str] = field(default_factory=list)
    robustness_metrics: Dict[str, float] = field(default_factory=dict)
    adversarial_examples: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate safety score range."""
        if not 0.0 <= self.overall_safety_score <= 1.0:
            raise ValueError("Safety score must be between 0.0 and 1.0")
    
    @property
    def is_safe(self) -> bool:
        """Check if model passes safety requirements."""
        return (self.overall_safety_score > 0.7 and 
                not any(v.risk_level == RiskLevel.CRITICAL for v in self.violations))
    
    @property
    def critical_vulnerabilities(self) -> List[EthicsViolation]:
        """Get only critical vulnerabilities."""
        return [v for v in self.violations if v.risk_level == RiskLevel.CRITICAL]


@dataclass
class AuditResult:
    """Comprehensive audit results combining all analysis components."""
    
    audit_id: str
    timestamp: datetime
    model_info: Dict[str, Any]
    dataset_info: Dict[str, Any]
    config_used: Dict[str, Any]
    
    # Component results
    bias_result: Optional[BiasResult] = None
    safety_result: Optional[SafetyResult] = None
    
    # Overall metrics
    overall_ethics_score: float = 0.0  # Combined score 0.0 to 1.0
    total_violations: int = 0
    
    # Metadata
    execution_time: float = 0.0  # seconds
    warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        self._calculate_overall_metrics()
    
    def _calculate_overall_metrics(self):
        """Calculate overall ethics score and violation count."""
        scores = []
        violations = []
        
        if self.bias_result:
            # Invert bias score (lower bias = higher ethics score)
            scores.append(1.0 - self.bias_result.overall_bias_score)
            violations.extend(self.bias_result.violations)
        
        if self.safety_result:
            scores.append(self.safety_result.overall_safety_score)
            violations.extend(self.safety_result.violations)
        
        # Calculate weighted average if we have scores
        if scores:
            self.overall_ethics_score = sum(scores) / len(scores)
        
        self.total_violations = len(violations)
    
    @property
    def risk_summary(self) -> Dict[str, int]:
        """Get count of violations by risk level."""
        violations = []
        if self.bias_result:
            violations.extend(self.bias_result.violations)
        if self.safety_result:
            violations.extend(self.safety_result.violations)
        
        summary = {level.value: 0 for level in RiskLevel}
        for violation in violations:
            summary[violation.risk_level.value] += 1
        
        return summary
    
    @property
    def passes_audit(self) -> bool:
        """Check if model passes overall ethics audit."""
        risk_counts = self.risk_summary
        has_critical_risks = risk_counts.get(RiskLevel.CRITICAL.value, 0) > 0
        return self.overall_ethics_score > 0.6 and not has_critical_risks
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result_dict = {
            "audit_id": self.audit_id,
            "timestamp": self.timestamp.isoformat(),
            "model_info": self.model_info,
            "dataset_info": self.dataset_info,
            "config_used": self.config_used,
            "overall_ethics_score": self.overall_ethics_score,
            "total_violations": self.total_violations,
            "execution_time": self.execution_time,
            "warnings": self.warnings,
            "risk_summary": self.risk_summary,
            "passes_audit": self.passes_audit
        }
        
        if self.bias_result:
            result_dict["bias_result"] = {
                "overall_bias_score": self.bias_result.overall_bias_score,
                "protected_attributes": self.bias_result.protected_attributes,
                "violations": [v.to_dict() for v in self.bias_result.violations],
                "metrics": self.bias_result.metrics,
                "statistical_tests": self.bias_result.statistical_tests,
                "fairness_metrics": self.bias_result.fairness_metrics,
                "has_significant_bias": self.bias_result.has_significant_bias
            }
        
        if self.safety_result:
            result_dict["safety_result"] = {
                "overall_safety_score": self.safety_result.overall_safety_score,
                "vulnerabilities_found": self.safety_result.vulnerabilities_found,
                "violations": [v.to_dict() for v in self.safety_result.violations],
                "attack_vectors": self.safety_result.attack_vectors,
                "robustness_metrics": self.safety_result.robustness_metrics,
                "adversarial_examples": self.safety_result.adversarial_examples,
                "is_safe": self.safety_result.is_safe,
                "critical_vulnerabilities": [v.to_dict() for v in self.safety_result.critical_vulnerabilities]
            }
        
        return result_dict
    
    def save_json(self, filepath: Union[str, Path]) -> None:
        """Save results to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_json(cls, filepath: Union[str, Path]) -> 'AuditResult':
        """Load results from JSON file."""
        filepath = Path(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Reconstruct the object (simplified version)
        # In a full implementation, you'd want complete deserialization
        return cls(
            audit_id=data["audit_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            model_info=data["model_info"],
            dataset_info=data["dataset_info"],
            config_used=data["config_used"],
            overall_ethics_score=data["overall_ethics_score"],
            total_violations=data["total_violations"],
            execution_time=data["execution_time"],
            warnings=data["warnings"]
        )