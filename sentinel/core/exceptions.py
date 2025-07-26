"""
Exception classes for AI Ethics Auditor Sentinel.
Professional error handling with detailed context and recovery suggestions.
"""

from typing import Optional, Dict, Any


class SentinelError(Exception):
    """Base exception class for all Sentinel errors."""
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "SENTINEL_ERROR"
        self.context = context or {}
        self.suggestion = suggestion
    
    def __str__(self) -> str:
        base_msg = f"[{self.error_code}] {self.message}"
        if self.suggestion:
            base_msg += f"\nSuggestion: {self.suggestion}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_msg += f"\nContext: {context_str}"
        return base_msg


class ConfigurationError(SentinelError):
    """Raised when there are configuration-related errors."""
    
    def __init__(self, message: str, config_path: Optional[str] = None, **kwargs):
        context = kwargs.get("context", {})
        if config_path:
            context["config_path"] = config_path
        
        suggestion = kwargs.get("suggestion", 
            "Please check the configuration file format and required fields.")
        
        super().__init__(
            message=message,
            error_code="CONFIG_ERROR",
            context=context,
            suggestion=suggestion
        )


class AuditError(SentinelError):
    """Raised when auditing process encounters errors."""
    
    def __init__(self, message: str, audit_stage: Optional[str] = None, **kwargs):
        context = kwargs.get("context", {})
        if audit_stage:
            context["audit_stage"] = audit_stage
        
        suggestion = kwargs.get("suggestion",
            "Please check input data format and model compatibility.")
        
        super().__init__(
            message=message,
            error_code="AUDIT_ERROR", 
            context=context,
            suggestion=suggestion
        )


class ModelCompatibilityError(SentinelError):
    """Raised when model is not compatible with auditing framework."""
    
    def __init__(self, message: str, model_type: Optional[str] = None, **kwargs):
        context = kwargs.get("context", {})
        if model_type:
            context["model_type"] = model_type
        
        suggestion = kwargs.get("suggestion",
            "Please ensure your model implements the required interface or use a compatible wrapper.")
        
        super().__init__(
            message=message,
            error_code="MODEL_COMPAT_ERROR",
            context=context,
            suggestion=suggestion
        )


class DataValidationError(SentinelError):
    """Raised when input data fails validation."""
    
    def __init__(self, message: str, data_field: Optional[str] = None, **kwargs):
        context = kwargs.get("context", {})
        if data_field:
            context["data_field"] = data_field
        
        suggestion = kwargs.get("suggestion",
            "Please check data format, missing values, and column names.")
        
        super().__init__(
            message=message,
            error_code="DATA_VALIDATION_ERROR",
            context=context,
            suggestion=suggestion
        )


class BiasDetectionError(AuditError):
    """Raised when bias detection encounters specific errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("audit_stage", "bias_detection")
        kwargs.setdefault("suggestion", 
            "Check protected attributes and model predictions format.")
        super().__init__(message, **kwargs)


class SafetyScanError(AuditError):
    """Raised when safety scanning encounters specific errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("audit_stage", "safety_scan")
        kwargs.setdefault("suggestion",
            "Verify model input handling and vulnerability test parameters.")
        super().__init__(message, **kwargs)


class ReportGenerationError(SentinelError):
    """Raised when report generation fails."""
    
    def __init__(self, message: str, report_format: Optional[str] = None, **kwargs):
        context = kwargs.get("context", {})
        if report_format:
            context["report_format"] = report_format
        
        suggestion = kwargs.get("suggestion",
            "Check template files and output directory permissions.")
        
        super().__init__(
            message=message,
            error_code="REPORT_ERROR",
            context=context,
            suggestion=suggestion
        )