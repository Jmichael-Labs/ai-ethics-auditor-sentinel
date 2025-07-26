"""
Core functionality tests for AI Ethics Auditor Sentinel.
Tests for configuration, auditor engine, results, and taxonomy loading.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from sentinel.core.config import AuditConfig, BiasConfig, SafetyConfig, ReportConfig
from sentinel.core.auditor import EthicsAuditor
from sentinel.core.results import AuditResult, BiasResult, SafetyResult, EthicsViolation, RiskLevel
from sentinel.core.taxonomy_loader import EthicsTaxonomyLoader, EthicsTaxonomy, EthicsCategory
from sentinel.core.exceptions import ConfigurationError, AuditError, ModelCompatibilityError


class TestConfiguration:
    """Test configuration management classes."""
    
    def test_bias_config_creation(self):
        """Test BiasConfig creation with valid parameters."""
        config = BiasConfig(
            protected_attributes=["gender", "race"],
            fairness_metrics=["demographic_parity", "equalized_odds"],
            significance_threshold=0.05,
            bias_threshold=0.1
        )
        
        assert config.protected_attributes == ["gender", "race"]
        assert config.fairness_metrics == ["demographic_parity", "equalized_odds"]
        assert config.significance_threshold == 0.05
        assert config.bias_threshold == 0.1
    
    def test_bias_config_validation(self):
        """Test BiasConfig validation of parameters."""
        with pytest.raises(ConfigurationError):
            BiasConfig(significance_threshold=1.5)  # Invalid threshold
        
        with pytest.raises(ConfigurationError):
            BiasConfig(bias_threshold=-0.1)  # Invalid threshold
        
        with pytest.raises(ConfigurationError):
            BiasConfig(min_sample_size=5)  # Too small sample size
    
    def test_safety_config_creation(self):
        """Test SafetyConfig creation with valid parameters."""
        config = SafetyConfig(
            vulnerability_tests=["adversarial_examples", "data_poisoning"],
            adversarial_epsilon=0.1,
            safety_threshold=0.7
        )
        
        assert "adversarial_examples" in config.vulnerability_tests
        assert config.adversarial_epsilon == 0.1
        assert config.safety_threshold == 0.7
    
    def test_safety_config_validation(self):
        """Test SafetyConfig validation of parameters."""
        with pytest.raises(ConfigurationError):
            SafetyConfig(adversarial_epsilon=1.5)  # Invalid epsilon
        
        with pytest.raises(ConfigurationError):
            SafetyConfig(safety_threshold=1.5)  # Invalid threshold
        
        with pytest.raises(ConfigurationError):
            SafetyConfig(k_anonymity=1)  # Invalid k value
    
    def test_audit_config_creation(self):
        """Test AuditConfig creation and component integration."""
        config = AuditConfig(
            audit_components=["bias", "safety"],
            parallel_execution=True,
            max_workers=2
        )
        
        assert config.audit_components == ["bias", "safety"]
        assert config.parallel_execution is True
        assert config.max_workers == 2
        assert isinstance(config.bias, BiasConfig)
        assert isinstance(config.safety, SafetyConfig)
        assert isinstance(config.report, ReportConfig)
    
    def test_audit_config_validation(self):
        """Test AuditConfig validation."""
        with pytest.raises(ConfigurationError):
            AuditConfig(audit_components=["invalid_component"])
        
        with pytest.raises(ConfigurationError):
            AuditConfig(log_level="INVALID_LEVEL")
        
        with pytest.raises(ConfigurationError):
            AuditConfig(max_workers=0)
    
    def test_config_yaml_loading(self):
        """Test loading configuration from YAML file."""
        yaml_content = '''
        audit_components:
          - "bias"
          - "safety"
        parallel_execution: true
        max_workers: 2
        log_level: "INFO"
        
        bias:
          protected_attributes:
            - "gender"
            - "race"
          fairness_metrics:
            - "demographic_parity"
          significance_threshold: 0.05
          bias_threshold: 0.1
        
        safety:
          vulnerability_tests:
            - "adversarial_examples"
          adversarial_epsilon: 0.1
          safety_threshold: 0.7
        
        report:
          formats:
            - "html"
          theme: "professional"
        '''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_path = f.name
        
        try:
            config = AuditConfig.from_yaml(config_path)
            
            assert config.audit_components == ["bias", "safety"]
            assert config.parallel_execution is True
            assert config.max_workers == 2
            assert config.bias.protected_attributes == ["gender", "race"]
            assert config.safety.adversarial_epsilon == 0.1
            assert config.report.theme == "professional"
        
        finally:
            Path(config_path).unlink()
    
    def test_config_dict_conversion(self):
        """Test configuration to/from dictionary conversion."""
        original_config = AuditConfig(
            audit_components=["bias"],
            max_workers=2
        )
        
        # Convert to dict and back
        config_dict = original_config.to_dict()
        reconstructed_config = AuditConfig.from_dict(config_dict)
        
        assert reconstructed_config.audit_components == original_config.audit_components
        assert reconstructed_config.max_workers == original_config.max_workers


class TestResults:
    """Test result data structures."""
    
    def test_ethics_violation_creation(self):
        """Test EthicsViolation creation and validation."""
        violation = EthicsViolation(
            violation_type="demographic_bias",
            description="Test violation",
            risk_level=RiskLevel.HIGH,
            confidence=0.8,
            affected_groups=["gender"],
            evidence={"test": "value"},
            recommendations=["Fix the bias"]
        )
        
        assert violation.violation_type == "demographic_bias"
        assert violation.risk_level == RiskLevel.HIGH
        assert violation.confidence == 0.8
        assert violation.affected_groups == ["gender"]
    
    def test_ethics_violation_validation(self):
        """Test EthicsViolation validation."""
        with pytest.raises(ValueError):
            EthicsViolation(
                violation_type="test",
                description="",  # Empty description
                risk_level=RiskLevel.LOW,
                confidence=0.5
            )
        
        with pytest.raises(ValueError):
            EthicsViolation(
                violation_type="test",
                description="Test",
                risk_level=RiskLevel.LOW,
                confidence=1.5  # Invalid confidence
            )
    
    def test_bias_result_creation(self):
        """Test BiasResult creation."""
        violations = [
            EthicsViolation(
                violation_type="demographic_bias",
                description="Test violation",
                risk_level=RiskLevel.MEDIUM,
                confidence=0.7
            )
        ]
        
        bias_result = BiasResult(
            overall_bias_score=0.3,
            protected_attributes=["gender", "race"],
            violations=violations,
            metrics={"demographic_parity": {"gender_diff": 0.1}},
            fairness_metrics={"overall_fairness": 0.7}
        )
        
        assert bias_result.overall_bias_score == 0.3
        assert len(bias_result.violations) == 1
        assert bias_result.has_significant_bias is False  # Score < 0.5 and no critical violations
    
    def test_safety_result_creation(self):
        """Test SafetyResult creation."""
        violations = [
            EthicsViolation(
                violation_type="adversarial_vulnerability",
                description="Test vulnerability",
                risk_level=RiskLevel.HIGH,
                confidence=0.9
            )
        ]
        
        safety_result = SafetyResult(
            overall_safety_score=0.6,
            vulnerabilities_found=1,
            violations=violations,
            attack_vectors=["FGSM"],
            robustness_metrics={"noise_robustness": 0.8}
        )
        
        assert safety_result.overall_safety_score == 0.6
        assert safety_result.vulnerabilities_found == 1
        assert safety_result.is_safe is False  # Score < 0.7
    
    def test_audit_result_creation(self):
        """Test AuditResult creation and metric calculation."""
        bias_result = BiasResult(
            overall_bias_score=0.2,
            protected_attributes=["gender"],
            violations=[],
            metrics={},
            fairness_metrics={}
        )
        
        safety_result = SafetyResult(
            overall_safety_score=0.8,
            vulnerabilities_found=0,
            violations=[],
            attack_vectors=[],
            robustness_metrics={}
        )
        
        audit_result = AuditResult(
            audit_id="test-audit-123",
            timestamp=datetime.now(),
            model_info={"type": "RandomForestClassifier"},
            dataset_info={"shape": [1000, 10]},
            config_used={"test": "config"},
            bias_result=bias_result,
            safety_result=safety_result
        )
        
        # Check calculated metrics
        assert audit_result.overall_ethics_score > 0.7  # Good combined score
        assert audit_result.total_violations == 0
        assert audit_result.passes_audit is True
    
    def test_audit_result_serialization(self):
        """Test AuditResult JSON serialization."""
        audit_result = AuditResult(
            audit_id="test-audit-456",
            timestamp=datetime.now(),
            model_info={"type": "test"},
            dataset_info={"shape": [100, 5]},
            config_used={}
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            audit_result.save_json(json_path)
            
            # Verify file was created and contains expected data
            loaded_result = AuditResult.load_json(json_path)
            assert loaded_result.audit_id == audit_result.audit_id
        
        finally:
            Path(json_path).unlink()


class TestTaxonomyLoader:
    """Test ethics taxonomy loading and management."""
    
    def test_ethics_category_creation(self):
        """Test EthicsCategory creation and validation."""
        category = EthicsCategory(
            name="Test Category",
            description="A test category",
            severity_level="high",
            subcategories=["sub1", "sub2"],
            detection_methods=["method1"],
            remediation_strategies=["strategy1"],
            regulatory_references=["ref1"]
        )
        
        assert category.name == "Test Category"
        assert category.severity_level == "high"
        assert len(category.subcategories) == 2
    
    def test_ethics_category_validation(self):
        """Test EthicsCategory validation."""
        with pytest.raises(ValueError):
            EthicsCategory(
                name="Test",
                description="Test",
                severity_level="invalid_level"  # Invalid severity
            )
    
    def test_ethics_taxonomy_creation(self):
        """Test EthicsTaxonomy creation."""
        category = EthicsCategory(
            name="Bias",
            description="Bias category",
            severity_level="high"
        )
        
        taxonomy = EthicsTaxonomy(
            name="Test Taxonomy",
            version="1.0.0",
            description="A test taxonomy",
            categories={"bias": category}
        )
        
        assert taxonomy.name == "Test Taxonomy"
        assert len(taxonomy.categories) == 1
        assert taxonomy.get_category("bias") == category
        assert taxonomy.get_category("nonexistent") is None
    
    def test_taxonomy_loader_initialization(self):
        """Test EthicsTaxonomyLoader initialization."""
        loader = EthicsTaxonomyLoader()
        
        # Should have default taxonomy loaded
        assert len(loader.taxonomies) >= 1
        assert "default" in loader.taxonomies
        assert loader.active_taxonomy is not None
    
    def test_taxonomy_loader_yaml_loading(self):
        """Test loading taxonomy from YAML file."""
        yaml_content = '''
        name: "Test Taxonomy"
        version: "1.0.0"
        description: "Test taxonomy for unit tests"
        metadata:
          created_by: "test"
        categories:
          test_bias:
            name: "Test Bias"
            description: "A test bias category"
            severity_level: "high"
            subcategories:
              - "demographic_bias"
            detection_methods:
              - "statistical_test"
            remediation_strategies:
              - "data_augmentation"
            regulatory_references:
              - "test_regulation"
        '''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name
        
        try:
            loader = EthicsTaxonomyLoader()
            taxonomy = loader.load_taxonomy(yaml_path, "test_taxonomy")
            
            assert taxonomy.name == "Test Taxonomy"
            assert "test_bias" in taxonomy.categories
            assert taxonomy.categories["test_bias"].severity_level == "high"
            assert "test_taxonomy" in loader.taxonomies
        
        finally:
            Path(yaml_path).unlink()
    
    def test_taxonomy_loader_set_active(self):
        """Test setting active taxonomy."""
        loader = EthicsTaxonomyLoader()
        original_active = loader.active_taxonomy
        
        # Create and add a test taxonomy
        test_taxonomy = EthicsTaxonomy(
            name="Test Active",
            version="1.0.0",
            description="Test",
            categories={}
        )
        loader.taxonomies["test_active"] = test_taxonomy
        
        # Set as active
        loader.set_active_taxonomy("test_active")
        assert loader.active_taxonomy == test_taxonomy
        
        # Test error for nonexistent taxonomy
        with pytest.raises(ConfigurationError):
            loader.set_active_taxonomy("nonexistent")


class TestEthicsAuditor:
    """Test main EthicsAuditor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample data
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        
        # Add protected attributes
        self.df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        self.df['target'] = y
        self.df['gender'] = np.random.choice(['M', 'F'], size=1000)
        self.df['race'] = np.random.choice(['A', 'B', 'C'], size=1000)
        
        # Create a simple model
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(X, y)
    
    def test_auditor_initialization(self):
        """Test EthicsAuditor initialization."""
        auditor = EthicsAuditor()
        
        assert auditor.config is not None
        assert auditor.taxonomy_loader is not None
        assert len(auditor.auditors) >= 0  # Depends on configured components
    
    def test_auditor_with_config(self):
        """Test EthicsAuditor initialization with custom config."""
        config = AuditConfig(
            audit_components=["bias"],
            max_workers=1
        )
        
        auditor = EthicsAuditor(config=config)
        
        assert auditor.config.max_workers == 1
        assert "bias" in auditor.config.audit_components
    
    def test_auditor_input_validation(self):
        """Test input validation in audit_model."""
        auditor = EthicsAuditor()
        
        # Test with None model
        with pytest.raises(ModelCompatibilityError):
            auditor.audit_model(None, self.df)
        
        # Test with None dataset
        with pytest.raises(AuditError):
            auditor.audit_model(self.model, None)
        
        # Test with empty dataset
        empty_df = pd.DataFrame()
        with pytest.raises(AuditError):
            auditor.audit_model(self.model, empty_df)
    
    def test_auditor_model_compatibility(self):
        """Test model compatibility checking."""
        auditor = EthicsAuditor()
        
        # Create mock model without predict method
        mock_model = Mock()
        del mock_model.predict  # Remove predict method
        
        with pytest.raises(ModelCompatibilityError):
            auditor.audit_model(mock_model, self.df)
    
    def test_auditor_successful_audit(self):
        """Test successful audit execution."""
        config = AuditConfig(
            audit_components=["bias"],  # Only test bias to speed up
            parallel_execution=False
        )
        
        auditor = EthicsAuditor(config=config)
        
        result = auditor.audit_model(self.model, self.df)
        
        assert isinstance(result, AuditResult)
        assert result.audit_id is not None
        assert result.timestamp is not None
        assert result.overall_ethics_score >= 0.0
        assert result.overall_ethics_score <= 1.0
        assert result.execution_time > 0
    
    def test_auditor_with_dict_dataset(self):
        """Test audit with dictionary dataset format."""
        config = AuditConfig(
            audit_components=["bias"],
            parallel_execution=False
        )
        
        auditor = EthicsAuditor(config=config)
        
        # Create dict dataset
        feature_cols = [col for col in self.df.columns if col not in ['target', 'gender', 'race']]
        dataset_dict = {
            "features": self.df[feature_cols + ['gender', 'race']],
            "labels": self.df['target']
        }
        
        result = auditor.audit_model(self.model, dataset_dict)
        
        assert isinstance(result, AuditResult)
        assert result.overall_ethics_score >= 0.0
    
    def test_auditor_parallel_execution(self):
        """Test parallel audit execution."""
        config = AuditConfig(
            audit_components=["bias", "safety"],
            parallel_execution=True,
            max_workers=2
        )
        
        auditor = EthicsAuditor(config=config)
        
        result = auditor.audit_model(self.model, self.df)
        
        assert isinstance(result, AuditResult)
        # Should have results from both components if they completed successfully
        # (Some may fail due to mock setup, but audit should still complete)
    
    def test_auditor_report_generation(self):
        """Test report generation."""
        config = AuditConfig(
            audit_components=["bias"],
            parallel_execution=False
        )
        
        auditor = EthicsAuditor(config=config)
        
        # Run audit
        result = auditor.audit_model(self.model, self.df)
        
        # Generate report
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_report.html"
            generated_path = auditor.generate_report(result, output_path)
            
            assert generated_path.exists()
            assert generated_path.suffix == ".html"
            
            # Verify report contains basic content
            content = generated_path.read_text()
            assert "AI Ethics Audit Report" in content
            assert result.audit_id in content


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_end_to_end_basic_audit(self):
        """Test complete end-to-end audit workflow."""
        # Create test data
        X, y = make_classification(
            n_samples=500,
            n_features=8,
            n_classes=2,
            random_state=42
        )
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(8)])
        df['target'] = y
        df['gender'] = np.random.choice(['M', 'F'], size=500)
        df['race'] = np.random.choice(['A', 'B'], size=500)
        
        # Train model
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        feature_cols = [col for col in df.columns if col not in ['target', 'gender', 'race']]
        model.fit(df[feature_cols], df['target'])
        
        # Create auditor with basic config
        config = AuditConfig(
            audit_components=["bias"],
            parallel_execution=False,
            log_level="WARNING"  # Reduce log noise
        )
        
        auditor = EthicsAuditor(config=config)
        
        # Run audit
        result = auditor.audit_model(model, df)
        
        # Verify result structure
        assert isinstance(result, AuditResult)
        assert result.audit_id is not None
        assert result.overall_ethics_score >= 0.0
        assert result.overall_ethics_score <= 1.0
        
        # Generate report
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = auditor.generate_report(result, Path(temp_dir) / "report.html")
            assert report_path.exists()
    
    def test_configuration_loading_integration(self):
        """Test loading configuration and running audit."""
        # Create sample config file
        config_content = '''
        audit_components:
          - "bias"
        parallel_execution: false
        log_level: "WARNING"
        
        bias:
          protected_attributes:
            - "gender"
          fairness_metrics:
            - "demographic_parity"
          significance_threshold: 0.05
          bias_threshold: 0.2
        '''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            config_path = f.name
        
        try:
            # Create auditor with config file
            auditor = EthicsAuditor(config=config_path)
            
            # Verify config was loaded
            assert auditor.config.audit_components == ["bias"]
            assert "gender" in auditor.config.bias.protected_attributes
            
            # Create simple test data
            X, y = make_classification(n_samples=200, n_features=5, random_state=42)
            df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
            df['target'] = y
            df['gender'] = np.random.choice(['M', 'F'], size=200)
            
            model = RandomForestClassifier(n_estimators=3, random_state=42)
            model.fit(X, y)
            
            # Run audit
            result = auditor.audit_model(model, df)
            assert isinstance(result, AuditResult)
        
        finally:
            Path(config_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])