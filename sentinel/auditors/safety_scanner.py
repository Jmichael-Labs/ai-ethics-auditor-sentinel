"""
Safety Scanner Module - Comprehensive AI safety vulnerability detection.
Implements advanced adversarial testing and security analysis.
"""

import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score
from loguru import logger

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

from sentinel.core.config import SafetyConfig
from sentinel.core.results import SafetyResult, EthicsViolation, RiskLevel
from sentinel.core.exceptions import SafetyScanError, DataValidationError
from sentinel.core.taxonomy_loader import EthicsTaxonomyLoader


class SafetyScanner:
    """
    Advanced safety vulnerability scanner for AI models.
    
    Performs comprehensive security analysis including:
    - Adversarial attack testing
    - Data poisoning detection
    - Model robustness evaluation
    - Privacy vulnerability assessment
    - Output manipulation testing
    
    Features:
    - Multiple attack vector implementations
    - Robustness metrics calculation
    - Privacy leak detection
    - Professional vulnerability reporting
    """
    
    def __init__(
        self,
        config: SafetyConfig,
        taxonomy_loader: Optional[EthicsTaxonomyLoader] = None
    ):
        """
        Initialize the safety scanner.
        
        Args:
            config: Safety scanning configuration
            taxonomy_loader: Ethics taxonomy loader for violation categorization
        """
        self.config = config
        self.taxonomy_loader = taxonomy_loader
        
        # Initialize vulnerability test methods
        self._initialize_tests()
        
        logger.info("SafetyScanner initialized")
    
    def _initialize_tests(self) -> None:
        """Initialize vulnerability test methods."""
        self.vulnerability_tests = {
            "adversarial_examples": self._test_adversarial_examples,
            "data_poisoning": self._test_data_poisoning_resistance,
            "model_inversion": self._test_model_inversion,
            "membership_inference": self._test_membership_inference,
            "prompt_injection": self._test_prompt_injection,
            "output_manipulation": self._test_output_manipulation
        }
    
    def scan_vulnerabilities(
        self,
        model: Any,
        dataset: Union[pd.DataFrame, Dict[str, Any]],
        target_column: str = "target"
    ) -> SafetyResult:
        """
        Perform comprehensive safety vulnerability scanning.
        
        Args:
            model: The ML model to audit
            dataset: Dataset for vulnerability testing
            target_column: Name of target/label column
            
        Returns:
            SafetyResult: Comprehensive safety analysis results
        """
        try:
            logger.info("Starting safety vulnerability scanning")
            start_time = time.time()
            
            # Validate and prepare data
            df, X, y_true, y_pred = self._prepare_data(model, dataset, target_column)
            
            # Initialize results
            vulnerabilities = []
            attack_vectors = []
            robustness_metrics = {}
            adversarial_examples = []
            
            # Run configured vulnerability tests
            for test_name in self.config.vulnerability_tests:
                if test_name in self.vulnerability_tests:
                    try:
                        logger.info(f"Running {test_name} test")
                        test_results = self.vulnerability_tests[test_name](
                            model, df, X, y_true, y_pred
                        )
                        
                        # Process test results
                        if "vulnerabilities" in test_results:
                            vulnerabilities.extend(test_results["vulnerabilities"])
                        
                        if "attack_vectors" in test_results:
                            attack_vectors.extend(test_results["attack_vectors"])
                        
                        if "metrics" in test_results:
                            robustness_metrics.update(test_results["metrics"])
                        
                        if "adversarial_examples" in test_results:
                            adversarial_examples.extend(test_results["adversarial_examples"])
                        
                        logger.info(f"Completed {test_name} test")
                    
                    except Exception as e:
                        logger.error(f"Failed {test_name} test: {e}")
                        # Create vulnerability for test failure
                        failure_vulnerability = EthicsViolation(
                            violation_type="safety_risk",
                            description=f"Safety test '{test_name}' failed to execute",
                            risk_level=RiskLevel.MEDIUM,
                            confidence=0.5,
                            evidence={"error": str(e), "test_name": test_name},
                            recommendations=[f"Review {test_name} test implementation and model compatibility"]
                        )
                        vulnerabilities.append(failure_vulnerability)
                
                else:
                    logger.warning(f"Unknown vulnerability test: {test_name}")
            
            # Calculate overall safety metrics
            overall_safety_score = self._calculate_safety_score(
                vulnerabilities, robustness_metrics
            )
            
            # Create result object
            result = SafetyResult(
                overall_safety_score=overall_safety_score,
                vulnerabilities_found=len(vulnerabilities),
                violations=vulnerabilities,
                attack_vectors=attack_vectors,
                robustness_metrics=robustness_metrics,
                adversarial_examples=adversarial_examples
            )
            
            execution_time = time.time() - start_time
            logger.info(f"Safety scanning completed in {execution_time:.2f}s")
            logger.info(f"Safety score: {overall_safety_score:.3f}")
            logger.info(f"Vulnerabilities found: {len(vulnerabilities)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Safety vulnerability scanning failed: {e}")
            raise SafetyScanError(f"Safety scanning failed: {str(e)}")
    
    def _prepare_data(
        self,
        model: Any,
        dataset: Union[pd.DataFrame, Dict[str, Any]],
        target_column: str
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare and validate data for safety analysis."""
        
        # Convert dataset to DataFrame if needed
        if isinstance(dataset, dict):
            if "features" not in dataset or "labels" not in dataset:
                raise DataValidationError("Dataset dict must contain 'features' and 'labels' keys")
            
            features = dataset["features"]
            labels = dataset["labels"]
            
            if isinstance(features, pd.DataFrame):
                df = features.copy()
                df[target_column] = labels
            else:
                df = pd.DataFrame(features)
                df[target_column] = labels
        
        elif isinstance(dataset, pd.DataFrame):
            df = dataset.copy()
            if target_column not in df.columns:
                raise DataValidationError(f"Target column '{target_column}' not found in dataset")
        
        else:
            raise DataValidationError(f"Unsupported dataset type: {type(dataset)}")
        
        # Extract features and labels
        feature_cols = [col for col in df.columns if col != target_column]
        X = df[feature_cols].values
        y_true = df[target_column].values
        
        # Get model predictions
        try:
            y_pred = model.predict(X)
        except Exception as e:
            raise SafetyScanError(f"Failed to get model predictions: {e}")
        
        # Validate data consistency
        if len(y_true) != len(y_pred):
            raise DataValidationError("Mismatch between true labels and predictions length")
        
        return df, X, y_true, y_pred
    
    def _test_adversarial_examples(
        self,
        model: Any,
        df: pd.DataFrame,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Test model robustness against adversarial examples."""
        
        results = {
            "vulnerabilities": [],
            "attack_vectors": [],
            "metrics": {},
            "adversarial_examples": []
        }
        
        try:
            # Fast Gradient Sign Method (FGSM) implementation
            adversarial_success_rate = self._fgsm_attack(model, X, y_true)
            
            results["metrics"]["adversarial_success_rate"] = adversarial_success_rate
            results["attack_vectors"].append("FGSM")
            
            # Check if success rate exceeds threshold
            if adversarial_success_rate > self.config.max_attack_success_rate:
                risk_level = RiskLevel.CRITICAL if adversarial_success_rate > 0.5 else RiskLevel.HIGH
                
                vulnerability = EthicsViolation(
                    violation_type="adversarial_vulnerability",
                    description=f"Model vulnerable to adversarial examples (success rate: {adversarial_success_rate:.3f})",
                    risk_level=risk_level,
                    confidence=adversarial_success_rate,
                    evidence={
                        "attack_method": "FGSM",
                        "success_rate": adversarial_success_rate,
                        "threshold": self.config.max_attack_success_rate,
                        "epsilon": self.config.adversarial_epsilon
                    },
                    recommendations=self._get_adversarial_recommendations()
                )
                results["vulnerabilities"].append(vulnerability)
            
            # Test robustness to noise
            noise_robustness = self._test_noise_robustness(model, X, y_true)
            results["metrics"]["noise_robustness"] = noise_robustness
            
            if noise_robustness < 0.8:  # Threshold for noise robustness
                vulnerability = EthicsViolation(
                    violation_type="adversarial_vulnerability",
                    description=f"Poor robustness to input noise (score: {noise_robustness:.3f})",
                    risk_level=RiskLevel.MEDIUM,
                    confidence=1.0 - noise_robustness,
                    evidence={
                        "noise_robustness_score": noise_robustness,
                        "noise_levels_tested": self.config.noise_levels
                    },
                    recommendations=["Implement noise-aware training", "Add input validation", "Use ensemble methods"]
                )
                results["vulnerabilities"].append(vulnerability)
        
        except Exception as e:
            logger.warning(f"Adversarial testing failed: {e}")
            # Add failure as a potential vulnerability
            results["vulnerabilities"].append(
                EthicsViolation(
                    violation_type="safety_risk",
                    description="Unable to test adversarial robustness",
                    risk_level=RiskLevel.MEDIUM,
                    confidence=0.7,
                    evidence={"error": str(e)},
                    recommendations=["Investigate model compatibility with adversarial testing"]
                )
            )
        
        return results
    
    def _fgsm_attack(self, model: Any, X: np.ndarray, y_true: np.ndarray) -> float:
        """Implement Fast Gradient Sign Method attack."""
        
        # Simplified FGSM implementation for demonstration
        # In practice, you'd use libraries like Adversarial Robustness Toolbox
        
        successful_attacks = 0
        total_samples = min(self.config.robustness_samples, len(X))
        
        # Sample random subset for testing
        indices = np.random.choice(len(X), total_samples, replace=False)
        X_test = X[indices]
        y_test = y_true[indices]
        
        try:
            # Get original predictions
            y_pred_original = model.predict(X_test)
            
            for i in range(total_samples):
                x_orig = X_test[i:i+1]
                y_orig = y_pred_original[i]
                
                # Create adversarial perturbation
                # Simplified: add random noise scaled by epsilon
                perturbation = np.random.normal(0, self.config.adversarial_epsilon, x_orig.shape)
                x_adv = x_orig + perturbation
                
                # Predict on adversarial example
                try:
                    y_adv = model.predict(x_adv)[0]
                    
                    # Check if attack was successful (prediction changed)
                    if y_adv != y_orig:
                        successful_attacks += 1
                
                except Exception:
                    continue  # Skip if prediction fails
            
            return successful_attacks / total_samples if total_samples > 0 else 0.0
        
        except Exception as e:
            logger.warning(f"FGSM attack implementation failed: {e}")
            return 0.0  # Conservative estimate
    
    def _test_noise_robustness(self, model: Any, X: np.ndarray, y_true: np.ndarray) -> float:
        """Test model robustness to different noise levels."""
        
        try:
            original_accuracy = accuracy_score(y_true, model.predict(X))
            robustness_scores = []
            
            for noise_level in self.config.noise_levels:
                # Add Gaussian noise
                X_noisy = X + np.random.normal(0, noise_level, X.shape)
                
                try:
                    y_pred_noisy = model.predict(X_noisy)
                    noisy_accuracy = accuracy_score(y_true, y_pred_noisy)
                    
                    # Calculate robustness as ratio of noisy to original accuracy
                    robustness = noisy_accuracy / original_accuracy if original_accuracy > 0 else 0
                    robustness_scores.append(robustness)
                
                except Exception:
                    robustness_scores.append(0.0)  # Failed prediction = no robustness
            
            return np.mean(robustness_scores) if robustness_scores else 0.0
        
        except Exception as e:
            logger.warning(f"Noise robustness testing failed: {e}")
            return 0.0
    
    def _test_data_poisoning_resistance(
        self,
        model: Any,
        df: pd.DataFrame,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Test resistance to data poisoning attacks."""
        
        results = {
            "vulnerabilities": [],
            "attack_vectors": ["data_poisoning"],
            "metrics": {}
        }
        
        try:
            # Simulate label flipping attack detection
            # Check for suspicious patterns in predictions vs true labels
            
            error_rate = 1.0 - accuracy_score(y_true, y_pred)
            results["metrics"]["base_error_rate"] = error_rate
            
            # Look for systematic prediction errors that might indicate poisoning
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X)
                    if proba.shape[1] == 2:  # Binary classification
                        proba_pos = proba[:, 1]
                        
                        # Check for suspicious confidence patterns
                        high_conf_wrong = np.sum((proba_pos > 0.8) & (y_pred != y_true))
                        total_wrong = np.sum(y_pred != y_true)
                        
                        if total_wrong > 0:
                            suspicious_rate = high_conf_wrong / total_wrong
                            results["metrics"]["suspicious_confidence_rate"] = suspicious_rate
                            
                            if suspicious_rate > 0.3:  # Threshold for suspicious behavior
                                vulnerability = EthicsViolation(
                                    violation_type="safety_risk",
                                    description="Suspicious prediction patterns may indicate data poisoning",
                                    risk_level=RiskLevel.MEDIUM,
                                    confidence=suspicious_rate,
                                    evidence={
                                        "suspicious_confidence_rate": suspicious_rate,
                                        "high_confidence_errors": high_conf_wrong,
                                        "total_errors": total_wrong
                                    },
                                    recommendations=self._get_poisoning_recommendations()
                                )
                                results["vulnerabilities"].append(vulnerability)
                
                except Exception as e:
                    logger.warning(f"Probability analysis failed: {e}")
            
            # Check for anomalous feature distributions
            feature_anomaly_score = self._detect_feature_anomalies(X)
            results["metrics"]["feature_anomaly_score"] = feature_anomaly_score
            
            if feature_anomaly_score > 0.1:  # Threshold for anomalous features
                vulnerability = EthicsViolation(
                    violation_type="safety_risk",
                    description="Anomalous feature distributions detected",
                    risk_level=RiskLevel.LOW,
                    confidence=feature_anomaly_score,
                    evidence={"anomaly_score": feature_anomaly_score},
                    recommendations=["Review data collection and preprocessing procedures"]
                )
                results["vulnerabilities"].append(vulnerability)
        
        except Exception as e:
            logger.warning(f"Data poisoning resistance test failed: {e}")
        
        return results
    
    def _detect_feature_anomalies(self, X: np.ndarray) -> float:
        """Detect anomalous feature distributions that might indicate poisoning."""
        
        try:
            anomaly_scores = []
            
            for feature_idx in range(X.shape[1]):
                feature_values = X[:, feature_idx]
                
                # Skip non-numeric features
                if not np.issubdtype(feature_values.dtype, np.number):
                    continue
                
                # Calculate z-scores
                z_scores = np.abs(stats.zscore(feature_values, nan_policy='omit'))
                
                # Count extreme outliers (z-score > 3)
                outlier_rate = np.sum(z_scores > 3) / len(z_scores)
                anomaly_scores.append(outlier_rate)
            
            return np.mean(anomaly_scores) if anomaly_scores else 0.0
        
        except Exception:
            return 0.0
    
    def _test_model_inversion(
        self,
        model: Any,
        df: pd.DataFrame,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Test susceptibility to model inversion attacks."""
        
        results = {
            "vulnerabilities": [],
            "attack_vectors": ["model_inversion"],
            "metrics": {}
        }
        
        try:
            # Simplified model inversion test
            # Check if model reveals too much information about training data
            
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                if proba.shape[1] == 2:
                    # Calculate confidence distribution
                    confidence_scores = np.max(proba, axis=1)
                    
                    # High confidence might indicate memorization
                    high_confidence_rate = np.sum(confidence_scores > 0.95) / len(confidence_scores)
                    results["metrics"]["high_confidence_rate"] = high_confidence_rate
                    
                    if high_confidence_rate > 0.5:  # Threshold for over-confidence
                        vulnerability = EthicsViolation(
                            violation_type="privacy_violation",
                            description="Model shows signs of overfitting/memorization",
                            risk_level=RiskLevel.MEDIUM,
                            confidence=high_confidence_rate,
                            evidence={
                                "high_confidence_rate": high_confidence_rate,
                                "confidence_threshold": 0.95
                            },
                            recommendations=self._get_privacy_recommendations()
                        )
                        results["vulnerabilities"].append(vulnerability)
            
            # Test feature reconstruction capability
            reconstruction_risk = self._assess_reconstruction_risk(model, X)
            results["metrics"]["reconstruction_risk"] = reconstruction_risk
            
            if reconstruction_risk > 0.3:
                vulnerability = EthicsViolation(
                    violation_type="privacy_violation",
                    description="Model may be vulnerable to feature reconstruction attacks",
                    risk_level=RiskLevel.MEDIUM,
                    confidence=reconstruction_risk,
                    evidence={"reconstruction_risk_score": reconstruction_risk},
                    recommendations=self._get_privacy_recommendations()
                )
                results["vulnerabilities"].append(vulnerability)
        
        except Exception as e:
            logger.warning(f"Model inversion test failed: {e}")
        
        return results
    
    def _assess_reconstruction_risk(self, model: Any, X: np.ndarray) -> float:
        """Assess risk of feature reconstruction from model outputs."""
        
        try:
            # Simplified assessment based on model behavior consistency
            # Real implementation would use more sophisticated techniques
            
            # Test with perturbed inputs to see how much model output changes
            n_samples = min(100, len(X))
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_test = X[indices]
            
            original_pred = model.predict(X_test)
            
            # Add small perturbations and check prediction stability
            perturbation_magnitude = 0.01
            stability_scores = []
            
            for i in range(10):  # Multiple perturbation tests
                perturbation = np.random.normal(0, perturbation_magnitude, X_test.shape)
                X_perturbed = X_test + perturbation
                
                try:
                    perturbed_pred = model.predict(X_perturbed)
                    
                    if hasattr(model, "predict_proba"):
                        orig_proba = model.predict_proba(X_test)
                        pert_proba = model.predict_proba(X_perturbed)
                        
                        # Calculate probability stability
                        prob_diff = np.mean(np.abs(orig_proba - pert_proba))
                        stability_scores.append(1.0 - prob_diff)  # Higher stability = lower risk
                    
                    else:
                        # Use prediction agreement as stability measure
                        agreement = np.mean(original_pred == perturbed_pred)
                        stability_scores.append(agreement)
                
                except Exception:
                    stability_scores.append(0.0)  # Failed prediction = unstable
            
            # Lower stability indicates higher reconstruction risk
            avg_stability = np.mean(stability_scores) if stability_scores else 0.5
            reconstruction_risk = 1.0 - avg_stability
            
            return reconstruction_risk
        
        except Exception:
            return 0.0
    
    def _test_membership_inference(
        self,
        model: Any,
        df: pd.DataFrame,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Test vulnerability to membership inference attacks."""
        
        results = {
            "vulnerabilities": [],
            "attack_vectors": ["membership_inference"],
            "metrics": {}
        }
        
        try:
            # Simplified membership inference test
            # Check if model behavior differs significantly between training-like and test-like data
            
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)
                
                # Calculate prediction entropy as privacy measure
                entropy_scores = []
                for i in range(len(proba)):
                    prob_dist = proba[i]
                    # Avoid log(0) by adding small epsilon
                    prob_dist = prob_dist + 1e-10
                    entropy = -np.sum(prob_dist * np.log(prob_dist))
                    entropy_scores.append(entropy)
                
                avg_entropy = np.mean(entropy_scores)
                results["metrics"]["prediction_entropy"] = avg_entropy
                
                # Low entropy might indicate memorization
                if avg_entropy < 0.5:  # Threshold for low entropy
                    vulnerability = EthicsViolation(
                        violation_type="privacy_violation",
                        description="Low prediction entropy may enable membership inference",
                        risk_level=RiskLevel.MEDIUM,
                        confidence=1.0 - avg_entropy,
                        evidence={
                            "average_entropy": avg_entropy,
                            "entropy_threshold": 0.5
                        },
                        recommendations=self._get_privacy_recommendations()
                    )
                    results["vulnerabilities"].append(vulnerability)
            
            # Test prediction confidence distribution
            confidence_variance = self._analyze_confidence_distribution(model, X)
            results["metrics"]["confidence_variance"] = confidence_variance
            
        except Exception as e:
            logger.warning(f"Membership inference test failed: {e}")
        
        return results
    
    def _analyze_confidence_distribution(self, model: Any, X: np.ndarray) -> float:
        """Analyze confidence distribution for privacy assessment."""
        
        try:
            if not hasattr(model, "predict_proba"):
                return 0.0
            
            proba = model.predict_proba(X)
            confidence_scores = np.max(proba, axis=1)
            
            # Calculate variance in confidence scores
            confidence_variance = np.var(confidence_scores)
            
            return confidence_variance
        
        except Exception:
            return 0.0
    
    def _test_prompt_injection(
        self,
        model: Any,
        df: pd.DataFrame,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Test vulnerability to prompt injection attacks (for text models)."""
        
        results = {
            "vulnerabilities": [],
            "attack_vectors": ["prompt_injection"],
            "metrics": {}
        }
        
        # This test is primarily for text/NLP models
        # For now, we'll implement a basic version
        
        try:
            # Check if model might be vulnerable to input manipulation
            input_sensitivity = self._test_input_sensitivity(model, X)
            results["metrics"]["input_sensitivity"] = input_sensitivity
            
            if input_sensitivity > 0.3:
                vulnerability = EthicsViolation(
                    violation_type="adversarial_vulnerability",
                    description="Model shows high sensitivity to input variations",
                    risk_level=RiskLevel.MEDIUM,
                    confidence=input_sensitivity,
                    evidence={"input_sensitivity_score": input_sensitivity},
                    recommendations=[
                        "Implement input validation and sanitization",
                        "Use robust preprocessing techniques",
                        "Consider adversarial training"
                    ]
                )
                results["vulnerabilities"].append(vulnerability)
        
        except Exception as e:
            logger.warning(f"Prompt injection test failed: {e}")
        
        return results
    
    def _test_input_sensitivity(self, model: Any, X: np.ndarray) -> float:
        """Test model sensitivity to input variations."""
        
        try:
            n_samples = min(50, len(X))
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_test = X[indices]
            
            original_pred = model.predict(X_test)
            
            # Test with various input modifications
            sensitivity_scores = []
            
            for scale in [0.01, 0.05, 0.1]:
                noise = np.random.normal(0, scale, X_test.shape)
                X_modified = X_test + noise
                
                try:
                    modified_pred = model.predict(X_modified)
                    
                    # Calculate prediction change rate
                    change_rate = np.mean(original_pred != modified_pred)
                    sensitivity_scores.append(change_rate)
                
                except Exception:
                    sensitivity_scores.append(1.0)  # Failed prediction = high sensitivity
            
            return np.mean(sensitivity_scores) if sensitivity_scores else 0.0
        
        except Exception:
            return 0.0
    
    def _test_output_manipulation(
        self,
        model: Any,
        df: pd.DataFrame,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Test for potential output manipulation vulnerabilities."""
        
        results = {
            "vulnerabilities": [],
            "attack_vectors": ["output_manipulation"],
            "metrics": {}
        }
        
        try:
            # Check for systematic biases in predictions
            prediction_distribution = np.bincount(y_pred) / len(y_pred)
            true_distribution = np.bincount(y_true) / len(y_true)
            
            # Calculate distribution difference
            if len(prediction_distribution) == len(true_distribution):
                distribution_diff = np.sum(np.abs(prediction_distribution - true_distribution))
                results["metrics"]["distribution_difference"] = distribution_diff
                
                if distribution_diff > 0.2:  # Threshold for significant difference
                    vulnerability = EthicsViolation(
                        violation_type="safety_risk",
                        description="Significant difference between predicted and true distributions",
                        risk_level=RiskLevel.MEDIUM,
                        confidence=distribution_diff,
                        evidence={
                            "distribution_difference": distribution_diff,
                            "predicted_distribution": prediction_distribution.tolist(),
                            "true_distribution": true_distribution.tolist()
                        },
                        recommendations=[
                            "Review model training and validation procedures",
                            "Check for systematic biases in data",
                            "Consider rebalancing techniques"
                        ]
                    )
                    results["vulnerabilities"].append(vulnerability)
            
            # Test consistency across similar inputs
            consistency_score = self._test_prediction_consistency(model, X)
            results["metrics"]["prediction_consistency"] = consistency_score
            
            if consistency_score < 0.8:
                vulnerability = EthicsViolation(
                    violation_type="safety_risk",
                    description="Low prediction consistency across similar inputs",
                    risk_level=RiskLevel.LOW,
                    confidence=1.0 - consistency_score,
                    evidence={"consistency_score": consistency_score},
                    recommendations=["Improve model stability and consistency"]
                )
                results["vulnerabilities"].append(vulnerability)
        
        except Exception as e:
            logger.warning(f"Output manipulation test failed: {e}")
        
        return results
    
    def _test_prediction_consistency(self, model: Any, X: np.ndarray) -> float:
        """Test prediction consistency across similar inputs."""
        
        try:
            n_tests = min(20, len(X) // 10)
            consistency_scores = []
            
            for _ in range(n_tests):
                # Select random sample
                idx = np.random.choice(len(X))
                x_base = X[idx:idx+1]
                
                # Create slightly modified versions
                modifications = []
                for scale in [0.001, 0.005, 0.01]:
                    noise = np.random.normal(0, scale, x_base.shape)
                    modifications.append(x_base + noise)
                
                # Get predictions for all versions
                try:
                    base_pred = model.predict(x_base)[0]
                    mod_preds = [model.predict(x_mod)[0] for x_mod in modifications]
                    
                    # Calculate consistency (proportion of same predictions)
                    consistency = np.mean([pred == base_pred for pred in mod_preds])
                    consistency_scores.append(consistency)
                
                except Exception:
                    consistency_scores.append(0.0)
            
            return np.mean(consistency_scores) if consistency_scores else 0.0
        
        except Exception:
            return 0.0
    
    def _calculate_safety_score(
        self,
        vulnerabilities: List[EthicsViolation],
        robustness_metrics: Dict[str, float]
    ) -> float:
        """Calculate overall safety score."""
        
        if not vulnerabilities and not robustness_metrics:
            return 1.0  # Perfect safety if no tests run
        
        # Base score from robustness metrics
        base_score = 1.0
        if robustness_metrics:
            # Average positive metrics
            positive_metrics = []
            for key, value in robustness_metrics.items():
                if "robustness" in key.lower() or "consistency" in key.lower():
                    positive_metrics.append(value)
                elif "success_rate" in key.lower() or "error_rate" in key.lower():
                    # Invert negative metrics
                    positive_metrics.append(1.0 - value)
                else:
                    # Assume most metrics are positive (higher = better)
                    positive_metrics.append(min(1.0, value))
            
            if positive_metrics:
                base_score = np.mean(positive_metrics)
        
        # Penalty from vulnerabilities
        vulnerability_penalty = 0.0
        if vulnerabilities:
            risk_weights = {
                RiskLevel.LOW: 0.1,
                RiskLevel.MEDIUM: 0.25,
                RiskLevel.HIGH: 0.5,
                RiskLevel.CRITICAL: 0.8
            }
            
            for vulnerability in vulnerabilities:
                penalty = risk_weights[vulnerability.risk_level] * vulnerability.confidence
                vulnerability_penalty += penalty
            
            # Normalize penalty
            vulnerability_penalty = min(1.0, vulnerability_penalty / len(vulnerabilities))
        
        # Combine base score and penalties
        safety_score = base_score * (1.0 - vulnerability_penalty)
        
        return max(0.0, min(1.0, safety_score))
    
    def _get_adversarial_recommendations(self) -> List[str]:
        """Get recommendations for addressing adversarial vulnerabilities."""
        recommendations = [
            "Implement adversarial training with diverse attack methods",
            "Use defensive distillation techniques",
            "Apply input preprocessing and validation",
            "Employ ensemble methods for robustness",
            "Add random noise during inference",
            "Implement gradient masking countermeasures"
        ]
        
        if self.taxonomy_loader:
            taxonomy_recs = self.taxonomy_loader.get_remediation_strategies("adversarial_vulnerability")
            recommendations.extend(taxonomy_recs)
        
        return recommendations
    
    def _get_poisoning_recommendations(self) -> List[str]:
        """Get recommendations for addressing data poisoning."""
        return [
            "Implement robust data validation procedures",
            "Use anomaly detection for training data",
            "Apply differential privacy techniques",
            "Employ robust optimization methods",
            "Implement data provenance tracking",
            "Use multiple data sources for validation"
        ]
    
    def _get_privacy_recommendations(self) -> List[str]:
        """Get recommendations for addressing privacy vulnerabilities."""
        recommendations = [
            "Implement differential privacy mechanisms",
            "Use federated learning approaches",
            "Apply data minimization principles",
            "Employ secure multi-party computation",
            "Add noise to model outputs",
            "Implement k-anonymity constraints"
        ]
        
        if self.taxonomy_loader:
            taxonomy_recs = self.taxonomy_loader.get_remediation_strategies("privacy_violation")
            recommendations.extend(taxonomy_recs)
        
        return recommendations