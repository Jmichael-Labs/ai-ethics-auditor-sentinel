"""
Bias Detection Module - Advanced bias detection and fairness analysis.
Implements state-of-the-art fairness metrics and statistical tests.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score
from loguru import logger

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

from sentinel.core.config import BiasConfig
from sentinel.core.results import BiasResult, EthicsViolation, RiskLevel
from sentinel.core.exceptions import BiasDetectionError, DataValidationError
from sentinel.core.taxonomy_loader import EthicsTaxonomyLoader


class BiasDetector:
    """
    Advanced bias detection system for AI models.
    
    Implements comprehensive fairness metrics and statistical tests
    to detect various forms of bias across protected attributes.
    
    Features:
    - Multiple fairness metrics (demographic parity, equalized odds, etc.)
    - Statistical significance testing
    - Intersectional bias analysis
    - Bootstrap confidence intervals
    - Professional-grade reporting
    """
    
    def __init__(
        self,
        config: BiasConfig,
        taxonomy_loader: Optional[EthicsTaxonomyLoader] = None
    ):
        """
        Initialize the bias detector.
        
        Args:
            config: Bias detection configuration
            taxonomy_loader: Ethics taxonomy loader for violation categorization
        """
        self.config = config
        self.taxonomy_loader = taxonomy_loader
        
        # Initialize fairness metric calculators
        self._initialize_metrics()
        
        logger.info("BiasDetector initialized")
    
    def _initialize_metrics(self) -> None:
        """Initialize fairness metric calculation methods."""
        self.metric_calculators = {
            "demographic_parity": self._calculate_demographic_parity,
            "equalized_odds": self._calculate_equalized_odds,
            "calibration": self._calculate_calibration,
            "individual_fairness": self._calculate_individual_fairness,
            "predictive_parity": self._calculate_predictive_parity,
            "treatment_equality": self._calculate_treatment_equality
        }
    
    def detect_bias(
        self,
        model: Any,
        dataset: Union[pd.DataFrame, Dict[str, Any]],
        protected_attributes: Optional[List[str]] = None,
        target_column: str = "target"
    ) -> BiasResult:
        """
        Perform comprehensive bias detection analysis.
        
        Args:
            model: The ML model to audit
            dataset: Dataset with features, labels, and protected attributes
            protected_attributes: List of protected attribute column names
            target_column: Name of target/label column
            
        Returns:
            BiasResult: Comprehensive bias analysis results
        """
        try:
            logger.info("Starting bias detection analysis")
            
            # Validate and prepare data
            df, y_true, y_pred, y_prob = self._prepare_data(
                model, dataset, target_column
            )
            
            # Use configured protected attributes if not specified
            if protected_attributes is None:
                protected_attributes = self._identify_protected_attributes(df)
            
            # Validate protected attributes exist in dataset
            missing_attrs = [attr for attr in protected_attributes if attr not in df.columns]
            if missing_attrs:
                raise DataValidationError(f"Protected attributes not found in dataset: {missing_attrs}")
            
            # Calculate fairness metrics
            metrics = self._calculate_all_metrics(
                df, y_true, y_pred, y_prob, protected_attributes
            )
            
            # Perform statistical tests
            statistical_tests = self._perform_statistical_tests(
                df, y_true, y_pred, protected_attributes
            )
            
            # Detect violations
            violations = self._detect_violations(
                metrics, statistical_tests, protected_attributes
            )
            
            # Calculate overall bias score
            overall_bias_score = self._calculate_overall_bias_score(metrics, violations)
            
            # Calculate fairness metrics summary
            fairness_summary = self._summarize_fairness_metrics(metrics)
            
            result = BiasResult(
                overall_bias_score=overall_bias_score,
                protected_attributes=protected_attributes,
                violations=violations,
                metrics=metrics,
                statistical_tests=statistical_tests,
                fairness_metrics=fairness_summary
            )
            
            logger.info(f"Bias detection completed. Bias score: {overall_bias_score:.3f}")
            logger.info(f"Found {len(violations)} violations")
            
            return result
            
        except Exception as e:
            logger.error(f"Bias detection failed: {e}")
            raise BiasDetectionError(f"Bias detection analysis failed: {str(e)}")
    
    def _prepare_data(
        self,
        model: Any,
        dataset: Union[pd.DataFrame, Dict[str, Any]],
        target_column: str
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Prepare and validate data for bias analysis."""
        
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
                # Convert numpy arrays to DataFrame
                df = pd.DataFrame(features)
                df[target_column] = labels
        
        elif isinstance(dataset, pd.DataFrame):
            df = dataset.copy()
            if target_column not in df.columns:
                raise DataValidationError(f"Target column '{target_column}' not found in dataset")
        
        else:
            raise DataValidationError(f"Unsupported dataset type: {type(dataset)}")
        
        # Extract true labels
        y_true = df[target_column].values
        
        # Get model predictions - exclude protected attributes and target from features
        all_protected_attrs = set(self.config.protected_attributes)
        
        # Get feature columns (exclude target and any protected attributes)
        feature_cols = [col for col in df.columns 
                       if col != target_column and col not in all_protected_attrs]
        
        # If no feature columns found, try to use numeric columns only
        if not feature_cols:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col != target_column]
        
        if not feature_cols:
            raise DataValidationError("No suitable feature columns found for model prediction")
        
        X = df[feature_cols]
        
        try:
            # Ensure X is numeric
            X_numeric = X.select_dtypes(include=[np.number])
            if X_numeric.shape[1] != X.shape[1]:
                logger.warning(f"Dropping non-numeric columns from features: {set(X.columns) - set(X_numeric.columns)}")
                X = X_numeric
            
            if X.empty:
                raise DataValidationError("No numeric features available for model prediction")
            
            y_pred = model.predict(X)
            
            # Get prediction probabilities if available
            y_prob = None
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X)
                if y_prob.shape[1] == 2:  # Binary classification
                    y_prob = y_prob[:, 1]  # Use positive class probability
        
        except Exception as e:
            raise BiasDetectionError(f"Failed to get model predictions: {e}")
        
        # Validate data consistency
        if len(y_true) != len(y_pred):
            raise DataValidationError("Mismatch between true labels and predictions length")
        
        if len(df) < self.config.min_sample_size:
            raise DataValidationError(
                f"Dataset too small for reliable bias analysis. "
                f"Need at least {self.config.min_sample_size} samples, got {len(df)}"
            )
        
        return df, y_true, y_pred, y_prob
    
    def _identify_protected_attributes(self, df: pd.DataFrame) -> List[str]:
        """Automatically identify protected attributes in the dataset."""
        identified_attrs = []
        
        for attr in self.config.protected_attributes:
            # Check for exact match
            if attr in df.columns:
                identified_attrs.append(attr)
                continue
            
            # Check for partial matches (e.g., "gender" matches "gender_encoded")
            matches = [col for col in df.columns if attr.lower() in col.lower()]
            if matches:
                identified_attrs.extend(matches)
        
        if not identified_attrs:
            logger.warning("No protected attributes found in dataset columns")
            # Fallback: look for categorical columns that might be protected attributes
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if categorical_cols:
                logger.info(f"Using categorical columns as potential protected attributes: {categorical_cols}")
                identified_attrs = categorical_cols[:3]  # Limit to first 3
        
        return identified_attrs
    
    def _calculate_all_metrics(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        protected_attributes: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate all configured fairness metrics."""
        
        all_metrics = {}
        
        for metric_name in self.config.fairness_metrics:
            if metric_name in self.metric_calculators:
                try:
                    calculator = self.metric_calculators[metric_name]
                    metric_results = calculator(
                        df, y_true, y_pred, y_prob, protected_attributes
                    )
                    all_metrics[metric_name] = metric_results
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate {metric_name}: {e}")
                    all_metrics[metric_name] = {}
            
            else:
                logger.warning(f"Unknown fairness metric: {metric_name}")
        
        return all_metrics
    
    def _calculate_demographic_parity(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        protected_attributes: List[str]
    ) -> Dict[str, float]:
        """Calculate demographic parity metrics."""
        
        results = {}
        
        for attr in protected_attributes:
            try:
                # Get unique groups
                groups = df[attr].unique()
                if len(groups) < 2:
                    continue
                
                group_rates = {}
                for group in groups:
                    mask = df[attr] == group
                    if mask.sum() == 0:
                        continue
                    
                    positive_rate = y_pred[mask].mean()
                    group_rates[str(group)] = positive_rate
                
                if len(group_rates) >= 2:
                    # Calculate parity difference (max - min)
                    rates = list(group_rates.values())
                    parity_diff = max(rates) - min(rates)
                    results[f"{attr}_demographic_parity_diff"] = parity_diff
                    
                    # Calculate ratio (min / max)
                    if max(rates) > 0:
                        parity_ratio = min(rates) / max(rates)
                        results[f"{attr}_demographic_parity_ratio"] = parity_ratio
                
            except Exception as e:
                logger.warning(f"Error calculating demographic parity for {attr}: {e}")
        
        return results
    
    def _calculate_equalized_odds(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        protected_attributes: List[str]
    ) -> Dict[str, float]:
        """Calculate equalized odds metrics."""
        
        results = {}
        
        for attr in protected_attributes:
            try:
                groups = df[attr].unique()
                if len(groups) < 2:
                    continue
                
                tpr_by_group = {}  # True Positive Rate
                fpr_by_group = {}  # False Positive Rate
                
                for group in groups:
                    mask = df[attr] == group
                    if mask.sum() == 0:
                        continue
                    
                    y_true_group = y_true[mask]
                    y_pred_group = y_pred[mask]
                    
                    # Calculate confusion matrix
                    try:
                        cm = confusion_matrix(y_true_group, y_pred_group)
                        if cm.shape == (2, 2):
                            tn, fp, fn, tp = cm.ravel()
                            
                            # True Positive Rate (Sensitivity)
                            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                            tpr_by_group[str(group)] = tpr
                            
                            # False Positive Rate
                            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                            fpr_by_group[str(group)] = fpr
                    
                    except ValueError:
                        # Handle case where only one class is present
                        continue
                
                # Calculate equalized odds difference
                if len(tpr_by_group) >= 2:
                    tpr_values = list(tpr_by_group.values())
                    tpr_diff = max(tpr_values) - min(tpr_values)
                    results[f"{attr}_tpr_diff"] = tpr_diff
                
                if len(fpr_by_group) >= 2:
                    fpr_values = list(fpr_by_group.values())
                    fpr_diff = max(fpr_values) - min(fpr_values)
                    results[f"{attr}_fpr_diff"] = fpr_diff
                    
                    # Overall equalized odds (average of TPR and FPR differences)
                    if f"{attr}_tpr_diff" in results:
                        eq_odds = (results[f"{attr}_tpr_diff"] + fpr_diff) / 2
                        results[f"{attr}_equalized_odds"] = eq_odds
            
            except Exception as e:
                logger.warning(f"Error calculating equalized odds for {attr}: {e}")
        
        return results
    
    def _calculate_calibration(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        protected_attributes: List[str]
    ) -> Dict[str, float]:
        """Calculate calibration metrics."""
        
        results = {}
        
        if y_prob is None:
            logger.warning("Prediction probabilities not available for calibration analysis")
            return results
        
        for attr in protected_attributes:
            try:
                groups = df[attr].unique()
                if len(groups) < 2:
                    continue
                
                calibration_by_group = {}
                
                for group in groups:
                    mask = df[attr] == group
                    if mask.sum() < 10:  # Need minimum samples for calibration
                        continue
                    
                    y_true_group = y_true[mask]
                    y_prob_group = y_prob[mask] if y_prob.ndim == 1 else y_prob[mask]
                    
                    # Bin probabilities and calculate calibration
                    n_bins = min(10, len(y_true_group) // 5)  # Adaptive number of bins
                    if n_bins < 2:
                        continue
                    
                    bin_boundaries = np.linspace(0, 1, n_bins + 1)
                    bin_lowers = bin_boundaries[:-1]
                    bin_uppers = bin_boundaries[1:]
                    
                    calibration_error = 0
                    total_samples = 0
                    
                    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                        in_bin = (y_prob_group > bin_lower) & (y_prob_group <= bin_upper)
                        prop_in_bin = in_bin.mean()
                        
                        if prop_in_bin > 0:
                            accuracy_in_bin = y_true_group[in_bin].mean()
                            avg_confidence_in_bin = y_prob_group[in_bin].mean()
                            
                            calibration_error += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                            total_samples += in_bin.sum()
                    
                    if total_samples > 0:
                        calibration_by_group[str(group)] = calibration_error
                
                # Calculate calibration difference between groups
                if len(calibration_by_group) >= 2:
                    cal_values = list(calibration_by_group.values())
                    cal_diff = max(cal_values) - min(cal_values)
                    results[f"{attr}_calibration_diff"] = cal_diff
            
            except Exception as e:
                logger.warning(f"Error calculating calibration for {attr}: {e}")
        
        return results
    
    def _calculate_individual_fairness(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        protected_attributes: List[str]
    ) -> Dict[str, float]:
        """Calculate individual fairness metrics."""
        
        results = {}
        
        # Individual fairness is complex to calculate without distance metrics
        # Here we implement a simplified version based on prediction consistency
        # for similar individuals (within each protected group)
        
        for attr in protected_attributes:
            try:
                groups = df[attr].unique()
                if len(groups) < 2:
                    continue
                
                # Calculate prediction variance within each group
                variance_by_group = {}
                
                for group in groups:
                    mask = df[attr] == group
                    if mask.sum() < 10:
                        continue
                    
                    y_pred_group = y_pred[mask]
                    if y_prob is not None:
                        y_prob_group = y_prob[mask] if y_prob.ndim == 1 else y_prob[mask]
                        # Use probability variance as individual fairness measure
                        variance = np.var(y_prob_group)
                    else:
                        # Use prediction variance for discrete predictions
                        variance = np.var(y_pred_group.astype(float))
                    
                    variance_by_group[str(group)] = variance
                
                # Calculate individual fairness as inverse of variance difference
                if len(variance_by_group) >= 2:
                    var_values = list(variance_by_group.values())
                    var_diff = max(var_values) - min(var_values)
                    # Higher variance difference = lower individual fairness
                    individual_fairness = 1.0 / (1.0 + var_diff)
                    results[f"{attr}_individual_fairness"] = individual_fairness
            
            except Exception as e:
                logger.warning(f"Error calculating individual fairness for {attr}: {e}")
        
        return results
    
    def _calculate_predictive_parity(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        protected_attributes: List[str]
    ) -> Dict[str, float]:
        """Calculate predictive parity metrics."""
        
        results = {}
        
        for attr in protected_attributes:
            try:
                groups = df[attr].unique()
                if len(groups) < 2:
                    continue
                
                ppv_by_group = {}  # Positive Predictive Value (Precision)
                
                for group in groups:
                    mask = df[attr] == group
                    if mask.sum() == 0:
                        continue
                    
                    y_true_group = y_true[mask]
                    y_pred_group = y_pred[mask]
                    
                    # Calculate positive predictive value
                    try:
                        cm = confusion_matrix(y_true_group, y_pred_group)
                        if cm.shape == (2, 2):
                            tn, fp, fn, tp = cm.ravel()
                            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                            ppv_by_group[str(group)] = ppv
                    
                    except ValueError:
                        continue
                
                # Calculate predictive parity difference
                if len(ppv_by_group) >= 2:
                    ppv_values = list(ppv_by_group.values())
                    ppv_diff = max(ppv_values) - min(ppv_values)
                    results[f"{attr}_predictive_parity_diff"] = ppv_diff
            
            except Exception as e:
                logger.warning(f"Error calculating predictive parity for {attr}: {e}")
        
        return results
    
    def _calculate_treatment_equality(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray],
        protected_attributes: List[str]
    ) -> Dict[str, float]:
        """Calculate treatment equality metrics."""
        
        results = {}
        
        for attr in protected_attributes:
            try:
                groups = df[attr].unique()
                if len(groups) < 2:
                    continue
                
                fn_fp_ratio_by_group = {}
                
                for group in groups:
                    mask = df[attr] == group
                    if mask.sum() == 0:
                        continue
                    
                    y_true_group = y_true[mask]
                    y_pred_group = y_pred[mask]
                    
                    try:
                        cm = confusion_matrix(y_true_group, y_pred_group)
                        if cm.shape == (2, 2):
                            tn, fp, fn, tp = cm.ravel()
                            
                            # Treatment equality: FN/FP ratio
                            if fp > 0:
                                fn_fp_ratio = fn / fp
                                fn_fp_ratio_by_group[str(group)] = fn_fp_ratio
                    
                    except ValueError:
                        continue
                
                # Calculate treatment equality difference
                if len(fn_fp_ratio_by_group) >= 2:
                    ratio_values = list(fn_fp_ratio_by_group.values())
                    ratio_diff = max(ratio_values) - min(ratio_values)
                    results[f"{attr}_treatment_equality_diff"] = ratio_diff
            
            except Exception as e:
                logger.warning(f"Error calculating treatment equality for {attr}: {e}")
        
        return results
    
    def _perform_statistical_tests(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attributes: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Perform statistical significance tests with bootstrap confidence intervals."""
        
        statistical_tests = {}
        
        for attr in protected_attributes:
            try:
                groups = df[attr].unique()
                if len(groups) < 2:  # Need at least 2 groups
                    continue
                
                # For binary comparisons
                if len(groups) == 2:
                    group1_mask = df[attr] == groups[0]
                    group2_mask = df[attr] == groups[1]
                    
                    if group1_mask.sum() < 5 or group2_mask.sum() < 5:
                        continue  # Need minimum samples
                    
                    # Chi-square test for independence
                    contingency_table = pd.crosstab(df[attr], y_pred)
                    try:
                        chi2, p_value_chi2, dof, expected = stats.chi2_contingency(contingency_table)
                        
                        # Bootstrap confidence interval for chi-square
                        bootstrap_chi2 = self._bootstrap_chi2(df[attr], y_pred)
                        
                        statistical_tests[f"{attr}_chi2_test"] = {
                            "test_statistic": float(chi2),
                            "p_value": float(p_value_chi2),
                            "degrees_of_freedom": int(dof),
                            "significant": bool(p_value_chi2 < self.config.significance_threshold),
                            "bootstrap_ci": bootstrap_chi2
                        }
                    
                    except ValueError as e:
                        logger.warning(f"Chi-square test failed for {attr}: {e}")
                    
                    # Two-sample t-test for prediction differences
                    group1_pred = y_pred[group1_mask].astype(float)
                    group2_pred = y_pred[group2_mask].astype(float)
                    
                    try:
                        t_stat, p_value_t = stats.ttest_ind(group1_pred, group2_pred)
                        
                        # Bootstrap confidence interval for mean difference
                        mean_diff_ci = self._bootstrap_mean_difference(group1_pred, group2_pred)
                        
                        statistical_tests[f"{attr}_ttest"] = {
                            "test_statistic": float(t_stat),
                            "p_value": float(p_value_t),
                            "significant": bool(p_value_t < self.config.significance_threshold),
                            "mean_difference": float(np.mean(group1_pred) - np.mean(group2_pred)),
                            "bootstrap_ci": mean_diff_ci
                        }
                    
                    except Exception as e:
                        logger.warning(f"T-test failed for {attr}: {e}")
                    
                    # Mann-Whitney U test (non-parametric)
                    try:
                        u_stat, p_value_u = stats.mannwhitneyu(
                            group1_pred, group2_pred, alternative='two-sided'
                        )
                        
                        statistical_tests[f"{attr}_mannwhitney"] = {
                            "test_statistic": float(u_stat),
                            "p_value": float(p_value_u),
                            "significant": bool(p_value_u < self.config.significance_threshold)
                        }
                    
                    except Exception as e:
                        logger.warning(f"Mann-Whitney U test failed for {attr}: {e}")
                
                # For multi-group comparisons (ANOVA)
                elif len(groups) > 2:
                    group_preds = []
                    for group in groups:
                        group_mask = df[attr] == group
                        if group_mask.sum() >= 5:  # Minimum samples
                            group_preds.append(y_pred[group_mask].astype(float))
                    
                    if len(group_preds) >= 2:
                        try:
                            f_stat, p_value_f = stats.f_oneway(*group_preds)
                            
                            statistical_tests[f"{attr}_anova"] = {
                                "test_statistic": float(f_stat),
                                "p_value": float(p_value_f),
                                "significant": bool(p_value_f < self.config.significance_threshold),
                                "groups_tested": len(group_preds)
                            }
                        
                        except Exception as e:
                            logger.warning(f"ANOVA failed for {attr}: {e}")
            
            except Exception as e:
                logger.warning(f"Statistical tests failed for {attr}: {e}")
        
        return statistical_tests
    
    def _bootstrap_chi2(self, group_data: pd.Series, predictions: np.ndarray, n_bootstrap: int = 500) -> Dict[str, float]:
        """Bootstrap confidence interval for chi-square statistic."""
        bootstrap_stats = []
        n_samples = len(group_data)
        
        for _ in range(min(n_bootstrap, self.config.bootstrap_samples // 2)):
            # Bootstrap resample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            boot_groups = group_data.iloc[indices]
            boot_preds = predictions[indices]
            
            try:
                contingency = pd.crosstab(boot_groups, boot_preds)
                if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
                    chi2, _, _, _ = stats.chi2_contingency(contingency)
                    bootstrap_stats.append(chi2)
            except:
                continue
        
        if bootstrap_stats:
            return {
                "lower": float(np.percentile(bootstrap_stats, 2.5)),
                "upper": float(np.percentile(bootstrap_stats, 97.5)),
                "mean": float(np.mean(bootstrap_stats))
            }
        else:
            return {"lower": 0.0, "upper": 0.0, "mean": 0.0}
    
    def _bootstrap_mean_difference(self, group1: np.ndarray, group2: np.ndarray, n_bootstrap: int = 500) -> Dict[str, float]:
        """Bootstrap confidence interval for mean difference between groups."""
        bootstrap_diffs = []
        
        for _ in range(min(n_bootstrap, self.config.bootstrap_samples // 2)):
            # Bootstrap resample both groups
            boot_group1 = np.random.choice(group1, len(group1), replace=True)
            boot_group2 = np.random.choice(group2, len(group2), replace=True)
            
            diff = np.mean(boot_group1) - np.mean(boot_group2)
            bootstrap_diffs.append(diff)
        
        return {
            "lower": float(np.percentile(bootstrap_diffs, 2.5)),
            "upper": float(np.percentile(bootstrap_diffs, 97.5)),
            "mean": float(np.mean(bootstrap_diffs))
        }
    
    def _detect_violations(
        self,
        metrics: Dict[str, Dict[str, float]],
        statistical_tests: Dict[str, Dict[str, Any]],
        protected_attributes: List[str]
    ) -> List[EthicsViolation]:
        """Detect ethics violations based on metrics and tests."""
        
        violations = []
        
        # Check demographic parity violations
        for attr in protected_attributes:
            demo_parity_key = f"{attr}_demographic_parity_diff"
            
            for metric_type, metric_results in metrics.items():
                if demo_parity_key in metric_results:
                    parity_diff = metric_results[demo_parity_key]
                    
                    if parity_diff > self.config.bias_threshold:
                        # Determine risk level based on severity
                        if parity_diff > 0.3:
                            risk = RiskLevel.CRITICAL
                        elif parity_diff > 0.2:
                            risk = RiskLevel.HIGH
                        elif parity_diff > 0.1:
                            risk = RiskLevel.MEDIUM
                        else:
                            risk = RiskLevel.LOW
                        
                        violation = EthicsViolation(
                            violation_type="demographic_bias",
                            description=f"Significant demographic parity difference detected for {attr}",
                            risk_level=risk,
                            confidence=min(1.0, parity_diff / self.config.bias_threshold),
                            affected_groups=[attr],
                            evidence={
                                "parity_difference": parity_diff,
                                "threshold": self.config.bias_threshold,
                                "metric_type": metric_type
                            },
                            recommendations=self._get_bias_recommendations(attr, "demographic_parity")
                        )
                        violations.append(violation)
        
        # Check equalized odds violations
        for attr in protected_attributes:
            eq_odds_key = f"{attr}_equalized_odds"
            
            for metric_type, metric_results in metrics.items():
                if eq_odds_key in metric_results:
                    eq_odds_diff = metric_results[eq_odds_key]
                    
                    if eq_odds_diff > self.config.bias_threshold:
                        risk = RiskLevel.HIGH if eq_odds_diff > 0.2 else RiskLevel.MEDIUM
                        
                        violation = EthicsViolation(
                            violation_type="algorithmic_bias",
                            description=f"Equalized odds violation detected for {attr}",
                            risk_level=risk,
                            confidence=min(1.0, eq_odds_diff / self.config.bias_threshold),
                            affected_groups=[attr],
                            evidence={
                                "equalized_odds_difference": eq_odds_diff,
                                "threshold": self.config.bias_threshold
                            },
                            recommendations=self._get_bias_recommendations(attr, "equalized_odds")
                        )
                        violations.append(violation)
        
        # Check statistical significance violations
        for test_name, test_results in statistical_tests.items():
            if test_results.get("significant", False):
                attr = test_name.split("_")[0]  # Extract attribute name
                
                evidence = {
                    "test_name": test_name,
                    "p_value": test_results["p_value"]
                }
                
                # Add test statistic if available
                if "test_statistic" in test_results:
                    evidence["test_statistic"] = test_results["test_statistic"]
                
                violation = EthicsViolation(
                    violation_type="algorithmic_bias",
                    description=f"Statistically significant bias detected in {test_name}",
                    risk_level=RiskLevel.MEDIUM,
                    confidence=1.0 - test_results["p_value"],
                    affected_groups=[attr],
                    evidence=evidence,
                    recommendations=self._get_bias_recommendations(attr, "statistical_significance")
                )
                violations.append(violation)
        
        return violations
    
    def _get_bias_recommendations(self, attribute: str, violation_type: str) -> List[str]:
        """Get recommendations for addressing bias violations."""
        
        base_recommendations = [
            "Review data collection and preprocessing procedures",
            "Implement bias-aware machine learning algorithms",
            "Consider using fairness constraints during model training",
            "Perform regular bias audits and monitoring"
        ]
        
        specific_recommendations = {
            "demographic_parity": [
                "Apply demographic parity post-processing techniques",
                "Use adversarial debiasing methods",
                "Implement equalized outcome constraints"
            ],
            "equalized_odds": [
                "Apply equalized odds post-processing",
                "Use threshold optimization techniques",
                "Implement calibration methods"
            ],
            "statistical_significance": [
                "Increase sample sizes for underrepresented groups",
                "Apply statistical bias correction methods",
                "Use stratified sampling techniques"
            ]
        }
        
        recommendations = base_recommendations.copy()
        if violation_type in specific_recommendations:
            recommendations.extend(specific_recommendations[violation_type])
        
        # Add taxonomy-based recommendations if available
        if self.taxonomy_loader:
            taxonomy_recs = self.taxonomy_loader.get_remediation_strategies("demographic_bias")
            recommendations.extend(taxonomy_recs)
        
        return recommendations
    
    def _calculate_overall_bias_score(
        self,
        metrics: Dict[str, Dict[str, float]],
        violations: List[EthicsViolation]
    ) -> float:
        """Calculate overall bias score (0.0 = no bias, 1.0 = maximum bias)."""
        
        if not metrics and not violations:
            return 0.0
        
        # Aggregate metric-based score
        all_metric_values = []
        for metric_results in metrics.values():
            all_metric_values.extend(metric_results.values())
        
        metric_score = 0.0
        if all_metric_values:
            # Normalize and average all metrics
            normalized_values = [min(1.0, abs(v) / 0.5) for v in all_metric_values]
            metric_score = np.mean(normalized_values)
        
        # Violation-based score
        violation_score = 0.0
        if violations:
            risk_weights = {
                RiskLevel.LOW: 0.25,
                RiskLevel.MEDIUM: 0.5,
                RiskLevel.HIGH: 0.75,
                RiskLevel.CRITICAL: 1.0
            }
            
            violation_scores = [risk_weights[v.risk_level] for v in violations]
            violation_score = min(1.0, np.mean(violation_scores))
        
        # Combine scores with weighting
        overall_score = 0.6 * metric_score + 0.4 * violation_score
        
        return min(1.0, overall_score)
    
    def _summarize_fairness_metrics(self, metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Create summary of fairness metrics."""
        
        summary = {}
        
        for metric_name, metric_results in metrics.items():
            if not metric_results:
                continue
            
            values = list(metric_results.values())
            summary[f"{metric_name}_mean"] = np.mean(values)
            summary[f"{metric_name}_max"] = np.max(values)
            summary[f"{metric_name}_min"] = np.min(values)
            summary[f"{metric_name}_std"] = np.std(values)
        
        return summary