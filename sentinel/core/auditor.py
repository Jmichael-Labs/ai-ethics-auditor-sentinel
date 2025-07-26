"""
Main Ethics Auditor Engine - Core auditing functionality.
Professional-grade implementation with comprehensive error handling and logging.
"""

import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

import numpy as np
import pandas as pd
from loguru import logger

from sentinel.core.config import AuditConfig
from sentinel.core.results import AuditResult, BiasResult, SafetyResult
from sentinel.core.exceptions import (
    AuditError, ConfigurationError, ModelCompatibilityError, 
    DataValidationError
)
from sentinel.core.taxonomy_loader import EthicsTaxonomyLoader
from sentinel.auditors.bias_detector import BiasDetector
from sentinel.auditors.safety_scanner import SafetyScanner
from sentinel.reporters.html_reporter import HTMLReporter


class EthicsAuditor:
    """
    Main ethics auditing engine for AI systems.
    
    Provides comprehensive auditing capabilities including bias detection,
    safety vulnerability scanning, and professional reporting.
    
    Example:
        >>> auditor = EthicsAuditor(config_path="configs/comprehensive.yaml")
        >>> results = auditor.audit_model(model, dataset)
        >>> auditor.generate_report(results, "audit_report.html")
    """
    
    def __init__(
        self,
        config: Optional[Union[AuditConfig, str, Path]] = None,
        taxonomy_loader: Optional[EthicsTaxonomyLoader] = None,
        custom_auditors: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Ethics Auditor.
        
        Args:
            config: Configuration object or path to YAML config file
            taxonomy_loader: Custom taxonomy loader instance
            custom_auditors: Dictionary of custom auditor implementations
        """
        # Initialize configuration
        if config is None:
            self.config = AuditConfig()
        elif isinstance(config, (str, Path)):
            self.config = AuditConfig.from_yaml(config)
        else:
            self.config = config
        
        # Set up logging
        self._setup_logging()
        
        # Initialize audit ID
        self.audit_id = str(uuid.uuid4())
        
        # Initialize taxonomy loader
        self.taxonomy_loader = taxonomy_loader or EthicsTaxonomyLoader()
        if self.config.taxonomy_path:
            self.taxonomy_loader.load_taxonomy(self.config.taxonomy_path)
        
        # Initialize component auditors
        self._initialize_auditors(custom_auditors or {})
        
        # Initialize reporter
        self.reporter = HTMLReporter(self.config.report)
        
        logger.info(f"EthicsAuditor initialized with ID: {self.audit_id}")
    
    def _setup_logging(self) -> None:
        """Configure logging system."""
        # Remove default handler and add custom one
        logger.remove()
        logger.add(
            sink=lambda msg: logging.getLogger("sentinel").info(msg.rstrip()),
            level=self.config.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
    
    def _initialize_auditors(self, custom_auditors: Dict[str, Any]) -> None:
        """Initialize component auditors."""
        self.auditors = {}
        
        # Initialize bias detector if enabled
        if "bias" in self.config.audit_components:
            if "bias" in custom_auditors:
                self.auditors["bias"] = custom_auditors["bias"]
            else:
                self.auditors["bias"] = BiasDetector(
                    config=self.config.bias,
                    taxonomy_loader=self.taxonomy_loader
                )
        
        # Initialize safety scanner if enabled
        if "safety" in self.config.audit_components:
            if "safety" in custom_auditors:
                self.auditors["safety"] = custom_auditors["safety"]
            else:
                self.auditors["safety"] = SafetyScanner(
                    config=self.config.safety,
                    taxonomy_loader=self.taxonomy_loader
                )
        
        logger.info(f"Initialized auditors: {list(self.auditors.keys())}")
    
    def audit_model(
        self,
        model: Any,
        dataset: Union[pd.DataFrame, np.ndarray, Dict[str, Any]],
        model_info: Optional[Dict[str, Any]] = None,
        dataset_info: Optional[Dict[str, Any]] = None
    ) -> AuditResult:
        """
        Perform comprehensive ethics audit on a model.
        
        Args:
            model: The ML model to audit (supports various frameworks)
            dataset: Dataset for auditing (with features and labels)
            model_info: Optional metadata about the model
            dataset_info: Optional metadata about the dataset
            
        Returns:
            AuditResult: Comprehensive audit results
            
        Raises:
            AuditError: If auditing process fails
            ModelCompatibilityError: If model is not compatible
            DataValidationError: If dataset is invalid
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting ethics audit for model (ID: {self.audit_id})")
            
            # Validate inputs
            self._validate_inputs(model, dataset)
            
            # Prepare model and dataset info
            model_info = model_info or self._extract_model_info(model)
            dataset_info = dataset_info or self._extract_dataset_info(dataset)
            
            # Initialize result object
            result = AuditResult(
                audit_id=self.audit_id,
                timestamp=datetime.now(),
                model_info=model_info,
                dataset_info=dataset_info,
                config_used=self.config.to_dict()
            )
            
            # Run audits (parallel or sequential)
            if self.config.parallel_execution and len(self.auditors) > 1:
                self._run_parallel_audits(model, dataset, result)
            else:
                self._run_sequential_audits(model, dataset, result)
            
            # Calculate execution time
            result.execution_time = time.time() - start_time
            
            logger.info(f"Ethics audit completed in {result.execution_time:.2f}s")
            logger.info(f"Overall ethics score: {result.overall_ethics_score:.3f}")
            logger.info(f"Total violations: {result.total_violations}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Ethics audit failed after {execution_time:.2f}s: {e}")
            raise AuditError(f"Audit failed: {str(e)}", context={"execution_time": execution_time})
    
    def _validate_inputs(self, model: Any, dataset: Union[pd.DataFrame, np.ndarray, Dict]) -> None:
        """Validate model and dataset inputs."""
        # Model validation
        if model is None:
            raise ModelCompatibilityError("Model cannot be None")
        
        # Check if model has required methods (basic compatibility check)
        required_methods = ["predict"]
        if hasattr(model, "predict_proba"):
            required_methods.append("predict_proba")
        
        missing_methods = [method for method in required_methods 
                          if not hasattr(model, method)]
        if missing_methods:
            raise ModelCompatibilityError(
                f"Model missing required methods: {missing_methods}",
                suggestion="Implement missing methods or use a compatible wrapper"
            )
        
        # Dataset validation
        if dataset is None:
            raise DataValidationError("Dataset cannot be None")
        
        if isinstance(dataset, pd.DataFrame):
            if dataset.empty:
                raise DataValidationError("Dataset cannot be empty")
            if len(dataset) < 10:
                raise DataValidationError("Dataset too small for reliable analysis (minimum 10 samples)")
        
        elif isinstance(dataset, np.ndarray):
            if dataset.size == 0:
                raise DataValidationError("Dataset array cannot be empty")
            if len(dataset) < 10:
                raise DataValidationError("Dataset too small for reliable analysis (minimum 10 samples)")
        
        elif isinstance(dataset, dict):
            if not dataset:
                raise DataValidationError("Dataset dictionary cannot be empty")
            # Additional validation for dict-based datasets
            required_keys = ["features", "labels"]
            missing_keys = [key for key in required_keys if key not in dataset]
            if missing_keys:
                raise DataValidationError(f"Dataset missing required keys: {missing_keys}")
        
        else:
            raise DataValidationError(
                f"Unsupported dataset type: {type(dataset)}",
                suggestion="Use pandas DataFrame, numpy array, or dictionary with 'features' and 'labels' keys"
            )
    
    def _extract_model_info(self, model: Any) -> Dict[str, Any]:
        """Extract metadata from model object."""
        model_info = {
            "type": str(type(model).__name__),
            "module": str(type(model).__module__),
        }
        
        # Try to extract additional info based on model type
        try:
            if hasattr(model, "get_params"):  # sklearn-style
                model_info["parameters"] = model.get_params()
            
            if hasattr(model, "__dict__"):
                # Extract safe attributes
                safe_attrs = ["n_features_", "classes_", "feature_names_in_"]
                for attr in safe_attrs:
                    if hasattr(model, attr):
                        value = getattr(model, attr)
                        if isinstance(value, (int, float, str, list)):
                            model_info[attr] = value
        
        except Exception as e:
            logger.warning(f"Could not extract full model info: {e}")
            model_info["extraction_warning"] = str(e)
        
        return model_info
    
    def _extract_dataset_info(self, dataset: Union[pd.DataFrame, np.ndarray, Dict]) -> Dict[str, Any]:
        """Extract metadata from dataset."""
        dataset_info = {"type": str(type(dataset).__name__)}
        
        try:
            if isinstance(dataset, pd.DataFrame):
                dataset_info.update({
                    "shape": list(dataset.shape),
                    "columns": list(dataset.columns),
                    "dtypes": {col: str(dtype) for col, dtype in dataset.dtypes.items()},
                    "memory_usage": dataset.memory_usage().sum(),
                    "null_counts": dataset.isnull().sum().to_dict()
                })
            
            elif isinstance(dataset, np.ndarray):
                dataset_info.update({
                    "shape": list(dataset.shape),
                    "dtype": str(dataset.dtype),
                    "memory_usage": dataset.nbytes
                })
            
            elif isinstance(dataset, dict):
                dataset_info["keys"] = list(dataset.keys())
                if "features" in dataset:
                    features = dataset["features"]
                    if hasattr(features, "shape"):
                        dataset_info["features_shape"] = list(features.shape)
                
        except Exception as e:
            logger.warning(f"Could not extract full dataset info: {e}")
            dataset_info["extraction_warning"] = str(e)
        
        return dataset_info
    
    def _run_parallel_audits(self, model: Any, dataset: Any, result: AuditResult) -> None:
        """Run audits in parallel using ThreadPoolExecutor."""
        logger.info("Running audits in parallel")
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit audit tasks
            future_to_auditor = {}
            
            for name, auditor in self.auditors.items():
                future = executor.submit(self._run_single_audit, name, auditor, model, dataset)
                future_to_auditor[future] = name
            
            # Collect results as they complete
            for future in as_completed(future_to_auditor):
                auditor_name = future_to_auditor[future]
                try:
                    audit_result = future.result()
                    self._store_audit_result(result, auditor_name, audit_result)
                    logger.info(f"Completed {auditor_name} audit")
                
                except Exception as e:
                    logger.error(f"Failed {auditor_name} audit: {e}")
                    result.warnings.append(f"{auditor_name} audit failed: {str(e)}")
    
    def _run_sequential_audits(self, model: Any, dataset: Any, result: AuditResult) -> None:
        """Run audits sequentially."""
        logger.info("Running audits sequentially")
        
        for name, auditor in self.auditors.items():
            try:
                logger.info(f"Starting {name} audit")
                audit_result = self._run_single_audit(name, auditor, model, dataset)
                self._store_audit_result(result, name, audit_result)
                logger.info(f"Completed {name} audit")
            
            except Exception as e:
                logger.error(f"Failed {name} audit: {e}")
                result.warnings.append(f"{name} audit failed: {str(e)}")
    
    def _run_single_audit(self, name: str, auditor: Any, model: Any, dataset: Any) -> Any:
        """Run a single audit component."""
        if name == "bias":
            return auditor.detect_bias(model, dataset)
        elif name == "safety":
            return auditor.scan_vulnerabilities(model, dataset)
        else:
            # For custom auditors
            if hasattr(auditor, "audit"):
                return auditor.audit(model, dataset)
            else:
                raise AuditError(f"Unknown audit method for {name}")
    
    def _store_audit_result(self, result: AuditResult, auditor_name: str, audit_result: Any) -> None:
        """Store individual audit result in main result object."""
        if auditor_name == "bias":
            result.bias_result = audit_result
        elif auditor_name == "safety":
            result.safety_result = audit_result
        # Handle custom auditors by storing in a generic way
        else:
            if not hasattr(result, "custom_results"):
                result.custom_results = {}
            result.custom_results[auditor_name] = audit_result
    
    def generate_report(
        self,
        audit_result: AuditResult,
        output_path: Optional[Union[str, Path]] = None,
        format_type: str = "html"
    ) -> Path:
        """
        Generate audit report.
        
        Args:
            audit_result: Results from audit_model()
            output_path: Path for output file (optional)
            format_type: Report format ("html", "json", "pdf")
            
        Returns:
            Path: Path to generated report file
        """
        logger.info(f"Generating {format_type} report")
        
        if format_type == "html":
            return self.reporter.generate_report(audit_result, output_path)
        elif format_type == "json":
            if output_path is None:
                output_path = Path(self.config.report.output_directory) / f"audit_{audit_result.audit_id}.json"
            audit_result.save_json(output_path)
            return Path(output_path)
        else:
            raise AuditError(f"Unsupported report format: {format_type}")
    
    def audit_multiple_models(
        self,
        models: Dict[str, Any],
        dataset: Union[pd.DataFrame, np.ndarray, Dict[str, Any]],
        comparison_report: bool = True
    ) -> Dict[str, AuditResult]:
        """
        Audit multiple models for comparison.
        
        Args:
            models: Dictionary of model_name -> model_object
            dataset: Shared dataset for all models
            comparison_report: Whether to generate comparison report
            
        Returns:
            Dictionary of model_name -> AuditResult
        """
        results = {}
        
        logger.info(f"Starting multi-model audit for {len(models)} models")
        
        for model_name, model in models.items():
            logger.info(f"Auditing model: {model_name}")
            try:
                # Create new audit ID for each model
                original_id = self.audit_id
                self.audit_id = f"{original_id}_{model_name}"
                
                result = self.audit_model(
                    model=model,
                    dataset=dataset,
                    model_info={"name": model_name}
                )
                results[model_name] = result
                
            except Exception as e:
                logger.error(f"Failed to audit model {model_name}: {e}")
                results[model_name] = None
            
            finally:
                self.audit_id = original_id
        
        logger.info(f"Multi-model audit completed. Success rate: {sum(1 for r in results.values() if r is not None)}/{len(models)}")
        
        return results