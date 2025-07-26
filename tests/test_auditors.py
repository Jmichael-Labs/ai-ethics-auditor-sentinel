"""
Auditor module tests for AI Ethics Auditor Sentinel.
Tests for bias detection and safety scanning functionality.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from sentinel.core.config import BiasConfig, SafetyConfig
from sentinel.core.results import BiasResult, SafetyResult, EthicsViolation, RiskLevel
from sentinel.core.exceptions import BiasDetectionError, SafetyScanError, DataValidationError
from sentinel.core.taxonomy_loader import EthicsTaxonomyLoader
from sentinel.auditors.bias_detector import BiasDetector
from sentinel.auditors.safety_scanner import SafetyScanner


class TestBiasDetector:
    """Test BiasDetector functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample data with bias
        np.random.seed(42)
        n_samples = 1000
        
        # Generate features
        X = np.random.randn(n_samples, 5)
        
        # Create protected attributes
        gender = np.random.choice(['M', 'F'], size=n_samples)
        race = np.random.choice(['A', 'B', 'C'], size=n_samples)
        age = np.random.choice(['Young', 'Old'], size=n_samples)
        
        # Create biased target (slightly favor certain groups)
        y = np.random.binomial(1, 0.5, n_samples)
        
        # Introduce some bias - males slightly more likely to be positive
        bias_mask = (gender == 'M')
        y[bias_mask] = np.random.binomial(1, 0.6, bias_mask.sum())
        
        # Create DataFrame
        self.df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        self.df['target'] = y
        self.df['gender'] = gender
        self.df['race'] = race
        self.df['age'] = age
        
        # Train a simple model
        self.model = LogisticRegression(random_state=42)
        self.model.fit(X, y)
        
        # Create bias detector
        self.config = BiasConfig(
            protected_attributes=['gender', 'race'],
            fairness_metrics=['demographic_parity', 'equalized_odds'],
            significance_threshold=0.05,
            bias_threshold=0.1
        )
        
        self.detector = BiasDetector(self.config)
    
    def test_bias_detector_initialization(self):
        """Test BiasDetector initialization."""
        assert self.detector.config == self.config
        assert self.detector.taxonomy_loader is None
        assert len(self.detector.metric_calculators) > 0
    
    def test_bias_detector_with_taxonomy(self):
        """Test BiasDetector with taxonomy loader."""
        taxonomy_loader = EthicsTaxonomyLoader()
        detector = BiasDetector(self.config, taxonomy_loader)
        
        assert detector.taxonomy_loader == taxonomy_loader
    
    def test_data_preparation_dataframe(self):
        """Test data preparation with DataFrame input."""
        df, y_true, y_pred, y_prob = self.detector._prepare_data(
            self.model, self.df, 'target'
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(y_true) == len(self.df)
        assert len(y_pred) == len(self.df)
        assert y_prob is not None  # LogisticRegression has predict_proba
        assert len(y_prob) == len(self.df)
    
    def test_data_preparation_dict(self):
        """Test data preparation with dictionary input."""
        feature_cols = [col for col in self.df.columns if col not in ['target']]
        dataset_dict = {
            'features': self.df[feature_cols],
            'labels': self.df['target']
        }
        
        df, y_true, y_pred, y_prob = self.detector._prepare_data(
            self.model, dataset_dict, 'target'
        )
        
        assert isinstance(df, pd.DataFrame)
        assert len(y_true) == len(self.df)
        assert len(y_pred) == len(self.df)
    
    def test_data_validation_errors(self):
        """Test data validation error handling."""
        # Test with missing target column
        df_missing_target = self.df.drop('target', axis=1)
        
        with pytest.raises(DataValidationError):
            self.detector._prepare_data(self.model, df_missing_target, 'target')
        
        # Test with invalid dataset type
        with pytest.raises(DataValidationError):
            self.detector._prepare_data(self.model, "invalid_data", 'target')
        
        # Test with too small dataset
        small_df = self.df.head(5)  # Smaller than min_sample_size
        
        with pytest.raises(DataValidationError):
            self.detector._prepare_data(self.model, small_df, 'target')
    
    def test_protected_attribute_identification(self):
        """Test automatic identification of protected attributes."""
        identified = self.detector._identify_protected_attributes(self.df)
        
        # Should find gender and race (if they're in config.protected_attributes)
        assert 'gender' in identified
        # May or may not find 'race' depending on exact implementation
    
    def test_demographic_parity_calculation(self):
        """Test demographic parity metric calculation."""
        df, y_true, y_pred, y_prob = self.detector._prepare_data(
            self.model, self.df, 'target'
        )
        
        metrics = self.detector._calculate_demographic_parity(
            df, y_true, y_pred, y_prob, ['gender']
        )
        
        assert isinstance(metrics, dict)
        # Should have metrics for gender
        gender_metrics = [k for k in metrics.keys() if 'gender' in k]
        assert len(gender_metrics) > 0
    
    def test_equalized_odds_calculation(self):
        """Test equalized odds metric calculation."""
        df, y_true, y_pred, y_prob = self.detector._prepare_data(
            self.model, self.df, 'target'
        )
        
        metrics = self.detector._calculate_equalized_odds(
            df, y_true, y_pred, y_prob, ['gender']
        )
        
        assert isinstance(metrics, dict)
        # May have TPR and FPR difference metrics
        if metrics:  # Only check if metrics were calculated
            assert all(isinstance(v, (int, float)) for v in metrics.values())
    
    def test_statistical_tests(self):
        """Test statistical significance testing."""
        df, y_true, y_pred, y_prob = self.detector._prepare_data(
            self.model, self.df, 'target'
        )
        
        tests = self.detector._perform_statistical_tests(
            df, y_true, y_pred, ['gender']
        )
        
        assert isinstance(tests, dict)
        # Should have some test results for gender
        gender_tests = [k for k in tests.keys() if 'gender' in k]
        if gender_tests:  # Only check if tests were performed
            for test_name in gender_tests:
                test_result = tests[test_name]
                assert 'p_value' in test_result
                assert 'significant' in test_result
                assert isinstance(test_result['significant'], bool)
    
    def test_violation_detection(self):
        """Test ethics violation detection."""
        # Create mock metrics with high bias
        metrics = {
            'demographic_parity': {
                'gender_demographic_parity_diff': 0.25  # Above threshold
            }
        }
        
        statistical_tests = {
            'gender_chi2_test': {
                'p_value': 0.01,  # Significant
                'significant': True
            }
        }
        
        violations = self.detector._detect_violations(
            metrics, statistical_tests, ['gender']
        )
        
        assert isinstance(violations, list)
        assert len(violations) > 0
        
        # Check violation structure
        for violation in violations:
            assert isinstance(violation, EthicsViolation)
            assert violation.violation_type in ['demographic_bias', 'algorithmic_bias']
            assert isinstance(violation.risk_level, RiskLevel)
    
    def test_bias_detection_full(self):
        """Test complete bias detection workflow."""
        result = self.detector.detect_bias(self.model, self.df)
        
        assert isinstance(result, BiasResult)
        assert 0.0 <= result.overall_bias_score <= 1.0
        assert isinstance(result.protected_attributes, list)
        assert isinstance(result.violations, list)
        assert isinstance(result.metrics, dict)
        assert isinstance(result.fairness_metrics, dict)
    
    def test_bias_detection_with_specified_attributes(self):
        """Test bias detection with specific protected attributes."""
        result = self.detector.detect_bias(
            self.model, self.df, protected_attributes=['gender']
        )
        
        assert isinstance(result, BiasResult)
        assert 'gender' in result.protected_attributes
    
    def test_bias_detection_error_handling(self):
        """Test error handling in bias detection."""
        # Test with incompatible model
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Model error")
        
        with pytest.raises(BiasDetectionError):
            self.detector.detect_bias(mock_model, self.df)


class TestSafetyScanner:
    """Test SafetyScanner functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create sample data
        X, y = make_classification(
            n_samples=500,
            n_features=8,
            n_classes=2,
            random_state=42
        )
        
        self.df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(8)])
        self.df['target'] = y
        
        # Train a simple model
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(X, y)
        
        # Create safety scanner
        self.config = SafetyConfig(
            vulnerability_tests=['adversarial_examples', 'data_poisoning'],
            adversarial_epsilon=0.1,
            robustness_samples=100,  # Small for testing
            safety_threshold=0.7
        )
        
        self.scanner = SafetyScanner(self.config)
    
    def test_safety_scanner_initialization(self):
        """Test SafetyScanner initialization."""
        assert self.scanner.config == self.config
        assert self.scanner.taxonomy_loader is None
        assert len(self.scanner.vulnerability_tests) > 0
    
    def test_safety_scanner_with_taxonomy(self):
        """Test SafetyScanner with taxonomy loader."""
        taxonomy_loader = EthicsTaxonomyLoader()
        scanner = SafetyScanner(self.config, taxonomy_loader)
        
        assert scanner.taxonomy_loader == taxonomy_loader
    
    def test_data_preparation(self):
        """Test data preparation for safety analysis."""
        df, X, y_true, y_pred = self.scanner._prepare_data(
            self.model, self.df, 'target'
        )
        
        assert isinstance(df, pd.DataFrame)
        assert isinstance(X, np.ndarray)
        assert len(y_true) == len(self.df)
        assert len(y_pred) == len(self.df)
        assert X.shape[0] == len(self.df)
    
    def test_fgsm_attack_implementation(self):
        """Test FGSM attack implementation."""
        X = self.df[[col for col in self.df.columns if col != 'target']].values
        y = self.df['target'].values
        
        success_rate = self.scanner._fgsm_attack(self.model, X, y)
        
        assert isinstance(success_rate, float)
        assert 0.0 <= success_rate <= 1.0
    
    def test_noise_robustness_testing(self):
        """Test noise robustness evaluation."""
        X = self.df[[col for col in self.df.columns if col != 'target']].values
        y = self.df['target'].values
        
        robustness_score = self.scanner._test_noise_robustness(self.model, X, y)
        
        assert isinstance(robustness_score, float)
        assert 0.0 <= robustness_score <= 1.0
    
    def test_adversarial_examples_test(self):
        """Test adversarial examples vulnerability test."""
        df, X, y_true, y_pred = self.scanner._prepare_data(
            self.model, self.df, 'target'
        )
        
        results = self.scanner._test_adversarial_examples(
            self.model, df, X, y_true, y_pred
        )
        
        assert isinstance(results, dict)
        assert 'vulnerabilities' in results
        assert 'attack_vectors' in results
        assert 'metrics' in results
        assert isinstance(results['vulnerabilities'], list)
    
    def test_data_poisoning_resistance_test(self):
        """Test data poisoning resistance evaluation."""
        df, X, y_true, y_pred = self.scanner._prepare_data(
            self.model, self.df, 'target'
        )
        
        results = self.scanner._test_data_poisoning_resistance(
            self.model, df, X, y_true, y_pred
        )
        
        assert isinstance(results, dict)
        assert 'vulnerabilities' in results
        assert 'metrics' in results
        assert isinstance(results['vulnerabilities'], list)
    
    def test_model_inversion_test(self):
        """Test model inversion vulnerability assessment."""
        df, X, y_true, y_pred = self.scanner._prepare_data(
            self.model, self.df, 'target'
        )
        
        results = self.scanner._test_model_inversion(
            self.model, df, X, y_true, y_pred
        )
        
        assert isinstance(results, dict)
        assert 'vulnerabilities' in results
        assert 'metrics' in results
        assert isinstance(results['vulnerabilities'], list)
    
    def test_membership_inference_test(self):
        """Test membership inference vulnerability assessment."""
        df, X, y_true, y_pred = self.scanner._prepare_data(
            self.model, self.df, 'target'
        )
        
        results = self.scanner._test_membership_inference(
            self.model, df, X, y_true, y_pred
        )
        
        assert isinstance(results, dict)
        assert 'vulnerabilities' in results
        assert 'metrics' in results
        assert isinstance(results['vulnerabilities'], list)
    
    def test_feature_anomaly_detection(self):
        """Test feature anomaly detection."""
        X = self.df[[col for col in self.df.columns if col != 'target']].values
        
        anomaly_score = self.scanner._detect_feature_anomalies(X)
        
        assert isinstance(anomaly_score, float)
        assert 0.0 <= anomaly_score <= 1.0
    
    def test_reconstruction_risk_assessment(self):
        """Test feature reconstruction risk assessment."""
        X = self.df[[col for col in self.df.columns if col != 'target']].values
        
        risk_score = self.scanner._assess_reconstruction_risk(self.model, X)
        
        assert isinstance(risk_score, float)
        assert 0.0 <= risk_score <= 1.0
    
    def test_input_sensitivity_testing(self):
        """Test input sensitivity analysis."""
        X = self.df[[col for col in self.df.columns if col != 'target']].values
        
        sensitivity_score = self.scanner._test_input_sensitivity(self.model, X)
        
        assert isinstance(sensitivity_score, float)
        assert 0.0 <= sensitivity_score <= 1.0
    
    def test_prediction_consistency_testing(self):
        """Test prediction consistency evaluation."""
        X = self.df[[col for col in self.df.columns if col != 'target']].values
        
        consistency_score = self.scanner._test_prediction_consistency(self.model, X)
        
        assert isinstance(consistency_score, float)
        assert 0.0 <= consistency_score <= 1.0
    
    def test_safety_score_calculation(self):
        """Test overall safety score calculation."""
        # Create sample vulnerabilities
        violations = [
            EthicsViolation(
                violation_type="adversarial_vulnerability",
                description="Test vulnerability",
                risk_level=RiskLevel.MEDIUM,
                confidence=0.7
            )
        ]
        
        metrics = {
            'noise_robustness': 0.8,
            'adversarial_success_rate': 0.2
        }
        
        safety_score = self.scanner._calculate_safety_score(violations, metrics)
        
        assert isinstance(safety_score, float)
        assert 0.0 <= safety_score <= 1.0
    
    def test_safety_scanning_full(self):
        """Test complete safety scanning workflow."""
        result = self.scanner.scan_vulnerabilities(self.model, self.df)
        
        assert isinstance(result, SafetyResult)
        assert 0.0 <= result.overall_safety_score <= 1.0
        assert isinstance(result.vulnerabilities_found, int)
        assert isinstance(result.violations, list)
        assert isinstance(result.attack_vectors, list)
        assert isinstance(result.robustness_metrics, dict)
    
    def test_safety_scanning_with_dict_dataset(self):
        """Test safety scanning with dictionary dataset."""
        feature_cols = [col for col in self.df.columns if col != 'target']
        dataset_dict = {
            'features': self.df[feature_cols],
            'labels': self.df['target']
        }
        
        result = self.scanner.scan_vulnerabilities(self.model, dataset_dict)
        
        assert isinstance(result, SafetyResult)
        assert result.overall_safety_score >= 0.0
    
    def test_safety_scanning_error_handling(self):
        """Test error handling in safety scanning."""
        # Test with incompatible model
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Model error")
        
        with pytest.raises(SafetyScanError):
            self.scanner.scan_vulnerabilities(mock_model, self.df)


class TestAuditorIntegration:
    """Integration tests for auditor components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test data with some bias
        np.random.seed(42)
        X, y = make_classification(
            n_samples=300,
            n_features=6,
            n_classes=2,
            random_state=42
        )
        
        self.df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(6)])
        self.df['target'] = y
        self.df['gender'] = np.random.choice(['M', 'F'], size=300)
        self.df['race'] = np.random.choice(['A', 'B'], size=300)
        
        # Train model
        self.model = LogisticRegression(random_state=42)
        self.model.fit(X, y)
    
    def test_bias_and_safety_together(self):
        """Test running both bias detection and safety scanning."""
        # Configure both auditors
        bias_config = BiasConfig(
            protected_attributes=['gender'],
            fairness_metrics=['demographic_parity'],
            bias_threshold=0.2  # Lenient for testing
        )
        
        safety_config = SafetyConfig(
            vulnerability_tests=['adversarial_examples'],
            robustness_samples=50,  # Small for testing
            safety_threshold=0.5
        )
        
        # Create auditors
        bias_detector = BiasDetector(bias_config)
        safety_scanner = SafetyScanner(safety_config)
        
        # Run both audits
        bias_result = bias_detector.detect_bias(self.model, self.df)
        safety_result = safety_scanner.scan_vulnerabilities(self.model, self.df)
        
        # Verify both results
        assert isinstance(bias_result, BiasResult)
        assert isinstance(safety_result, SafetyResult)
        
        assert bias_result.overall_bias_score >= 0.0
        assert safety_result.overall_safety_score >= 0.0
    
    def test_auditor_with_taxonomy_integration(self):
        """Test auditors with taxonomy loader integration."""
        taxonomy_loader = EthicsTaxonomyLoader()
        
        bias_config = BiasConfig(protected_attributes=['gender'])
        bias_detector = BiasDetector(bias_config, taxonomy_loader)
        
        result = bias_detector.detect_bias(self.model, self.df)
        
        # Verify taxonomy integration
        assert bias_detector.taxonomy_loader == taxonomy_loader
        assert isinstance(result, BiasResult)
        
        # Check if violations have taxonomy-based recommendations
        if result.violations:
            for violation in result.violations:
                assert isinstance(violation.recommendations, list)
    
    def test_auditor_recommendation_generation(self):
        """Test recommendation generation from auditors."""
        bias_config = BiasConfig(
            protected_attributes=['gender'],
            bias_threshold=0.01  # Very strict to trigger violations
        )
        
        bias_detector = BiasDetector(bias_config)
        
        # Create biased predictions to trigger violations
        with patch.object(self.model, 'predict') as mock_predict:
            # Create biased predictions
            predictions = np.zeros(len(self.df))
            gender_mask = self.df['gender'] == 'M'
            predictions[gender_mask] = 1  # All males predicted positive
            predictions[~gender_mask] = 0  # All females predicted negative
            mock_predict.return_value = predictions
            
            result = bias_detector.detect_bias(self.model, self.df)
            
            # Should detect violations and provide recommendations
            if result.violations:
                for violation in result.violations:
                    assert len(violation.recommendations) > 0
                    assert all(isinstance(rec, str) for rec in violation.recommendations)
    
    def test_auditor_performance_with_large_dataset(self):
        """Test auditor performance with larger dataset."""
        # Create larger dataset
        X, y = make_classification(
            n_samples=2000,
            n_features=10,
            n_classes=2,
            random_state=42
        )
        
        large_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        large_df['target'] = y
        large_df['gender'] = np.random.choice(['M', 'F'], size=2000)
        
        # Train model
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        # Configure for faster execution
        bias_config = BiasConfig(
            protected_attributes=['gender'],
            fairness_metrics=['demographic_parity'],
            bootstrap_samples=100  # Reduced for speed
        )
        
        safety_config = SafetyConfig(
            vulnerability_tests=['adversarial_examples'],
            robustness_samples=100,  # Reduced for speed
            adversarial_iterations=10  # Reduced for speed
        )
        
        # Run audits and measure basic performance
        bias_detector = BiasDetector(bias_config)
        safety_scanner = SafetyScanner(safety_config)
        
        import time
        
        start_time = time.time()
        bias_result = bias_detector.detect_bias(model, large_df)
        bias_time = time.time() - start_time
        
        start_time = time.time()
        safety_result = safety_scanner.scan_vulnerabilities(model, large_df)
        safety_time = time.time() - start_time
        
        # Verify results are valid
        assert isinstance(bias_result, BiasResult)
        assert isinstance(safety_result, SafetyResult)
        
        # Basic performance check (should complete in reasonable time)
        assert bias_time < 30.0  # Should complete within 30 seconds
        assert safety_time < 30.0  # Should complete within 30 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])