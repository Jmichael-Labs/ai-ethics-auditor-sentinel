"""
HTML Reporter Module - Professional ethics audit report generation.
Creates comprehensive, interactive HTML reports with visualizations.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import base64
from io import BytesIO

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from jinja2 import Template
from loguru import logger

from sentinel.core.config import ReportConfig
from sentinel.core.results import AuditResult, RiskLevel
from sentinel.core.exceptions import ReportGenerationError


class HTMLReporter:
    """
    Professional HTML report generator for ethics audit results.
    
    Creates comprehensive, interactive reports with:
    - Executive summary
    - Detailed findings with visualizations
    - Risk assessment matrices
    - Actionable recommendations
    - Professional styling and branding
    """
    
    def __init__(self, config: ReportConfig):
        """
        Initialize the HTML reporter.
        
        Args:
            config: Report generation configuration
        """
        self.config = config
        
        # Set up styling based on configuration
        self._setup_styling()
        
        logger.info("HTMLReporter initialized")
    
    def _setup_styling(self) -> None:
        """Set up styling configuration for reports."""
        
        # Color schemes
        self.color_schemes = {
            "default": {
                "primary": "#2E86AB",
                "secondary": "#A23B72", 
                "success": "#F18F01",
                "warning": "#C73E1D",
                "danger": "#8B0000",
                "background": "#FFFFFF",
                "text": "#333333",
                "light_gray": "#F8F9FA",
                "medium_gray": "#E9ECEF"
            },
            "colorblind": {
                "primary": "#0173B2",
                "secondary": "#029E73",
                "success": "#D55E00", 
                "warning": "#CC78BC",
                "danger": "#CA9161",
                "background": "#FFFFFF",
                "text": "#333333",
                "light_gray": "#F8F9FA",
                "medium_gray": "#E9ECEF"
            },
            "high_contrast": {
                "primary": "#000080",
                "secondary": "#800080",
                "success": "#008000",
                "warning": "#FF8C00",
                "danger": "#DC143C",
                "background": "#FFFFFF",
                "text": "#000000",
                "light_gray": "#F0F0F0",
                "medium_gray": "#D3D3D3"
            }
        }
        
        self.colors = self.color_schemes.get(self.config.color_scheme, self.color_schemes["default"])
        
        # Risk level colors
        self.risk_colors = {
            RiskLevel.LOW: "#28A745",      # Green
            RiskLevel.MEDIUM: "#FFC107",   # Yellow
            RiskLevel.HIGH: "#FD7E14",     # Orange
            RiskLevel.CRITICAL: "#DC3545"  # Red
        }
    
    def generate_report(
        self,
        audit_result: AuditResult,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Generate comprehensive HTML audit report.
        
        Args:
            audit_result: Complete audit results
            output_path: Optional output file path
            
        Returns:
            Path: Path to generated HTML report
        """
        try:
            logger.info("Generating HTML audit report")
            
            # Set up output path
            if output_path is None:
                output_dir = Path(self.config.output_directory)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = audit_result.model_info.get("type", "model")
                filename = self.config.filename_template.format(
                    timestamp=timestamp,
                    model_name=model_name
                ) + ".html"
                output_path = output_dir / filename
            else:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate report content
            report_html = self._generate_report_html(audit_result)
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_html)
            
            logger.info(f"HTML report generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"HTML report generation failed: {e}")
            raise ReportGenerationError(f"Failed to generate HTML report: {str(e)}")
    
    def _generate_report_html(self, audit_result: AuditResult) -> str:
        """Generate complete HTML report content."""
        
        # Create report sections
        sections = {
            "header": self._generate_header(audit_result),
            "executive_summary": self._generate_executive_summary(audit_result),
            "risk_overview": self._generate_risk_overview(audit_result),
            "bias_analysis": self._generate_bias_analysis(audit_result),
            "safety_analysis": self._generate_safety_analysis(audit_result),
            "recommendations": self._generate_recommendations(audit_result),
            "technical_details": self._generate_technical_details(audit_result),
            "appendix": self._generate_appendix(audit_result)
        }
        
        # Generate visualizations
        if self.config.include_visualizations:
            visualizations = self._generate_visualizations(audit_result)
            sections["visualizations"] = visualizations
        
        # Combine into complete HTML
        template = self._get_html_template()
        
        # Import RiskLevel for template access
        from sentinel.core.results import RiskLevel
        
        report_html = template.render(
            audit_result=audit_result,
            sections=sections,
            config=self.config,
            colors=self.colors,
            risk_colors=self.risk_colors,
            RiskLevel=RiskLevel,  # Add RiskLevel to template context
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        return report_html
    
    def _generate_header(self, audit_result: AuditResult) -> str:
        """Generate report header section."""
        
        header_html = f'''
        <div class="report-header">
            <div class="container">
                <div class="row">
                    <div class="col-md-8">
                        <h1 class="report-title">AI Ethics Audit Report</h1>
                        <p class="report-subtitle">Comprehensive Analysis of Model Ethics and Safety</p>
                        <div class="audit-info">
                            <span class="badge badge-info">Audit ID: {audit_result.audit_id}</span>
                            <span class="badge badge-secondary">Generated: {audit_result.timestamp.strftime("%Y-%m-%d %H:%M:%S")}</span>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="score-card">
                            <div class="score-value">{audit_result.overall_ethics_score:.2f}</div>
                            <div class="score-label">Overall Ethics Score</div>
                            <div class="score-status {'pass' if audit_result.passes_audit else 'fail'}">
                                {'PASS' if audit_result.passes_audit else 'FAIL'}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        '''
        
        return header_html
    
    def _generate_executive_summary(self, audit_result: AuditResult) -> str:
        """Generate executive summary section."""
        
        # Calculate key metrics
        risk_summary = audit_result.risk_summary
        total_violations = audit_result.total_violations
        critical_count = risk_summary.get(RiskLevel.CRITICAL.value, 0)
        high_count = risk_summary.get(RiskLevel.HIGH.value, 0)
        
        # Determine overall assessment
        if critical_count > 0:
            assessment = "CRITICAL ISSUES DETECTED"
            assessment_class = "critical"
        elif high_count > 0:
            assessment = "HIGH RISK ISSUES FOUND"
            assessment_class = "high"
        elif total_violations > 0:
            assessment = "MODERATE CONCERNS IDENTIFIED"
            assessment_class = "medium"
        else:
            assessment = "NO MAJOR ISSUES DETECTED"
            assessment_class = "low"
        
        summary_html = f'''
        <div class="executive-summary">
            <h2>Executive Summary</h2>
            
            <div class="assessment-banner {assessment_class}">
                <h3>{assessment}</h3>
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    <div class="summary-card">
                        <h4>Audit Overview</h4>
                        <ul>
                            <li><strong>Model Type:</strong> {audit_result.model_info.get('type', 'Unknown')}</li>
                            <li><strong>Dataset Size:</strong> {audit_result.dataset_info.get('shape', ['Unknown', 'Unknown'])[0]} samples</li>
                            <li><strong>Execution Time:</strong> {audit_result.execution_time:.2f} seconds</li>
                            <li><strong>Components Tested:</strong> {', '.join(audit_result.config_used.get('audit_components', []))}</li>
                        </ul>
                    </div>
                </div>
                
                <div class="col-md-6">
                    <div class="summary-card">
                        <h4>Key Findings</h4>
                        <ul>
                            <li><strong>Total Violations:</strong> {total_violations}</li>
                            <li><strong>Critical Issues:</strong> {critical_count}</li>
                            <li><strong>High Risk Issues:</strong> {high_count}</li>
                            <li><strong>Ethics Score:</strong> {audit_result.overall_ethics_score:.3f} / 1.000</li>
                        </ul>
                    </div>
                </div>
            </div>
            
            <div class="key-recommendations">
                <h4>Priority Actions Required</h4>
                {self._generate_priority_actions(audit_result)}
            </div>
        </div>
        '''
        
        return summary_html
    
    def _generate_priority_actions(self, audit_result: AuditResult) -> str:
        """Generate priority actions based on findings."""
        
        actions = []
        
        # Get critical and high-risk violations
        critical_violations = []
        high_violations = []
        
        if audit_result.bias_result:
            for violation in audit_result.bias_result.violations:
                if violation.risk_level == RiskLevel.CRITICAL:
                    critical_violations.append(violation)
                elif violation.risk_level == RiskLevel.HIGH:
                    high_violations.append(violation)
        
        if audit_result.safety_result:
            for violation in audit_result.safety_result.violations:
                if violation.risk_level == RiskLevel.CRITICAL:
                    critical_violations.append(violation)
                elif violation.risk_level == RiskLevel.HIGH:
                    high_violations.append(violation)
        
        # Generate actions based on violations
        if critical_violations:
            actions.append("🚨 IMMEDIATE: Address critical security vulnerabilities")
            actions.append("🛡️ IMMEDIATE: Implement safety constraints and monitoring")
        
        if high_violations:
            actions.append("⚠️ HIGH PRIORITY: Remediate bias and fairness issues")
            actions.append("🔍 HIGH PRIORITY: Enhance model robustness and testing")
        
        if not critical_violations and not high_violations:
            actions.append("✅ Continue regular monitoring and maintenance")
            actions.append("📊 Consider enhanced testing for edge cases")
        
        # Add general recommendations
        actions.append("📋 Review and update ethics policies")
        actions.append("🎓 Provide team training on AI ethics")
        
        actions_html = "<ul>" + "".join(f"<li>{action}</li>" for action in actions) + "</ul>"
        return actions_html
    
    def _generate_risk_overview(self, audit_result: AuditResult) -> str:
        """Generate risk overview section."""
        
        risk_summary = audit_result.risk_summary
        
        risk_html = f'''
        <div class="risk-overview">
            <h2>Risk Assessment Overview</h2>
            
            <div class="risk-matrix">
                <div class="row">
                    <div class="col-md-3">
                        <div class="risk-card critical">
                            <div class="risk-count">{risk_summary.get(RiskLevel.CRITICAL.value, 0)}</div>
                            <div class="risk-label">Critical</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="risk-card high">
                            <div class="risk-count">{risk_summary.get(RiskLevel.HIGH.value, 0)}</div>
                            <div class="risk-label">High</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="risk-card medium">
                            <div class="risk-count">{risk_summary.get(RiskLevel.MEDIUM.value, 0)}</div>
                            <div class="risk-label">Medium</div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="risk-card low">
                            <div class="risk-count">{risk_summary.get(RiskLevel.LOW.value, 0)}</div>
                            <div class="risk-label">Low</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        '''
        
        return risk_html
    
    def _generate_bias_analysis(self, audit_result: AuditResult) -> str:
        """Generate bias analysis section."""
        
        if not audit_result.bias_result:
            return '<div class="bias-analysis"><h2>Bias Analysis</h2><p>Bias analysis was not performed.</p></div>'
        
        bias_result = audit_result.bias_result
        
        bias_html = f'''
        <div class="bias-analysis">
            <h2>Bias and Fairness Analysis</h2>
            
            <div class="bias-summary">
                <div class="row">
                    <div class="col-md-4">
                        <div class="metric-card">
                            <div class="metric-value">{bias_result.overall_bias_score:.3f}</div>
                            <div class="metric-label">Bias Score</div>
                            <div class="metric-description">0.0 = No Bias, 1.0 = Maximum Bias</div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="metric-card">
                            <div class="metric-value">{'YES' if bias_result.has_significant_bias else 'NO'}</div>
                            <div class="metric-label">Significant Bias</div>
                            <div class="metric-description">Statistical significance detected</div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="metric-card">
                            <div class="metric-value">{len(bias_result.protected_attributes)}</div>
                            <div class="metric-label">Attributes Tested</div>
                            <div class="metric-description">{', '.join(bias_result.protected_attributes)}</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="bias-violations">
                <h3>Detected Bias Violations</h3>
                {self._generate_violations_table(bias_result.violations)}
            </div>
            
            <div class="fairness-metrics">
                <h3>Fairness Metrics Summary</h3>
                {self._generate_fairness_metrics_table(bias_result.fairness_metrics)}
            </div>
        </div>
        '''
        
        return bias_html
    
    def _generate_safety_analysis(self, audit_result: AuditResult) -> str:
        """Generate safety analysis section."""
        
        if not audit_result.safety_result:
            return '<div class="safety-analysis"><h2>Safety Analysis</h2><p>Safety analysis was not performed.</p></div>'
        
        safety_result = audit_result.safety_result
        
        safety_html = f'''
        <div class="safety-analysis">
            <h2>Safety and Security Analysis</h2>
            
            <div class="safety-summary">
                <div class="row">
                    <div class="col-md-4">
                        <div class="metric-card">
                            <div class="metric-value">{safety_result.overall_safety_score:.3f}</div>
                            <div class="metric-label">Safety Score</div>
                            <div class="metric-description">0.0 = Unsafe, 1.0 = Completely Safe</div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="metric-card">
                            <div class="metric-value">{'SAFE' if safety_result.is_safe else 'UNSAFE'}</div>
                            <div class="metric-label">Safety Status</div>
                            <div class="metric-description">Overall safety assessment</div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="metric-card">
                            <div class="metric-value">{safety_result.vulnerabilities_found}</div>
                            <div class="metric-label">Vulnerabilities</div>
                            <div class="metric-description">Total security issues found</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="attack-vectors">
                <h3>Attack Vectors Tested</h3>
                <div class="attack-vector-list">
                    {self._generate_attack_vector_list(safety_result.attack_vectors)}
                </div>
            </div>
            
            <div class="safety-violations">
                <h3>Security Vulnerabilities</h3>
                {self._generate_violations_table(safety_result.violations)}
            </div>
            
            <div class="robustness-metrics">
                <h3>Robustness Metrics</h3>
                {self._generate_robustness_metrics_table(safety_result.robustness_metrics)}
            </div>
        </div>
        '''
        
        return safety_html
    
    def _generate_violations_table(self, violations: List) -> str:
        """Generate HTML table for violations."""
        
        if not violations:
            return '<p class="no-violations">No violations detected.</p>'
        
        table_rows = []
        for violation in violations:
            risk_class = violation.risk_level.value
            confidence_pct = int(violation.confidence * 100)
            
            table_rows.append(f'''
                <tr class="violation-row {risk_class}">
                    <td><span class="risk-badge {risk_class}">{violation.risk_level.value.upper()}</span></td>
                    <td>{violation.violation_type}</td>
                    <td>{violation.description}</td>
                    <td>{confidence_pct}%</td>
                    <td>{', '.join(violation.affected_groups) if violation.affected_groups else 'N/A'}</td>
                </tr>
            ''')
        
        table_html = f'''
        <div class="table-responsive">
            <table class="table table-striped violations-table">
                <thead>
                    <tr>
                        <th>Risk Level</th>
                        <th>Type</th>
                        <th>Description</th>
                        <th>Confidence</th>
                        <th>Affected Groups</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(table_rows)}
                </tbody>
            </table>
        </div>
        '''
        
        return table_html
    
    def _generate_fairness_metrics_table(self, metrics: Dict[str, float]) -> str:
        """Generate fairness metrics table."""
        
        if not metrics:
            return '<p>No fairness metrics available.</p>'
        
        table_rows = []
        for metric_name, value in metrics.items():
            formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            
            table_rows.append(f'''
                <tr>
                    <td>{metric_name.replace('_', ' ').title()}</td>
                    <td>{formatted_value}</td>
                </tr>
            ''')
        
        table_html = f'''
        <div class="table-responsive">
            <table class="table table-striped metrics-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(table_rows)}
                </tbody>
            </table>
        </div>
        '''
        
        return table_html
    
    def _generate_robustness_metrics_table(self, metrics: Dict[str, float]) -> str:
        """Generate robustness metrics table."""
        return self._generate_fairness_metrics_table(metrics)  # Same format
    
    def _generate_attack_vector_list(self, attack_vectors: List[str]) -> str:
        """Generate attack vector list."""
        
        if not attack_vectors:
            return '<p>No attack vectors tested.</p>'
        
        vector_items = []
        for vector in attack_vectors:
            vector_items.append(f'<li class="attack-vector-item">{vector.replace("_", " ").title()}</li>')
        
        return f'<ul class="attack-vector-list">{"".join(vector_items)}</ul>'
    
    def _generate_recommendations(self, audit_result: AuditResult) -> str:
        """Generate recommendations section."""
        
        recommendations = []
        
        # Collect recommendations from violations
        if audit_result.bias_result:
            for violation in audit_result.bias_result.violations:
                recommendations.extend(violation.recommendations)
        
        if audit_result.safety_result:
            for violation in audit_result.safety_result.violations:
                recommendations.extend(violation.recommendations)
        
        # Remove duplicates while preserving order
        unique_recommendations = list(dict.fromkeys(recommendations))
        
        if not unique_recommendations:
            unique_recommendations = [
                "Continue regular monitoring and auditing",
                "Maintain current ethical AI practices",
                "Stay updated with latest ethics guidelines"
            ]
        
        rec_items = []
        for i, rec in enumerate(unique_recommendations[:10], 1):  # Limit to top 10
            rec_items.append(f'<li class="recommendation-item"><strong>{i}.</strong> {rec}</li>')
        
        rec_html = f'''
        <div class="recommendations">
            <h2>Recommendations</h2>
            <div class="recommendations-content">
                <p>Based on the audit findings, we recommend the following actions to improve your AI system's ethics and safety:</p>
                <ul class="recommendations-list">
                    {''.join(rec_items)}
                </ul>
            </div>
        </div>
        '''
        
        return rec_html
    
    def _generate_technical_details(self, audit_result: AuditResult) -> str:
        """Generate technical details section."""
        
        details_html = f'''
        <div class="technical-details">
            <h2>Technical Details</h2>
            
            <div class="accordion" id="technicalAccordion">
                <div class="card">
                    <div class="card-header" id="modelInfoHeader">
                        <h5 class="mb-0">
                            <button class="btn btn-link" type="button" data-toggle="collapse" data-target="#modelInfo">
                                Model Information
                            </button>
                        </h5>
                    </div>
                    <div id="modelInfo" class="collapse" data-parent="#technicalAccordion">
                        <div class="card-body">
                            <pre>{json.dumps(audit_result.model_info, indent=2, default=str)}</pre>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header" id="datasetInfoHeader">
                        <h5 class="mb-0">
                            <button class="btn btn-link" type="button" data-toggle="collapse" data-target="#datasetInfo">
                                Dataset Information
                            </button>
                        </h5>
                    </div>
                    <div id="datasetInfo" class="collapse" data-parent="#technicalAccordion">
                        <div class="card-body">
                            <pre>{json.dumps(audit_result.dataset_info, indent=2, default=str)}</pre>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <div class="card-header" id="configHeader">
                        <h5 class="mb-0">
                            <button class="btn btn-link" type="button" data-toggle="collapse" data-target="#configInfo">
                                Configuration Used
                            </button>
                        </h5>
                    </div>
                    <div id="configInfo" class="collapse" data-parent="#technicalAccordion">
                        <div class="card-body">
                            <pre>{json.dumps(audit_result.config_used, indent=2, default=str)}</pre>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        '''
        
        return details_html
    
    def _generate_appendix(self, audit_result: AuditResult) -> str:
        """Generate appendix section."""
        
        appendix_html = f'''
        <div class="appendix">
            <h2>Appendix</h2>
            
            <div class="audit-metadata">
                <h3>Audit Metadata</h3>
                <ul>
                    <li><strong>Audit ID:</strong> {audit_result.audit_id}</li>
                    <li><strong>Timestamp:</strong> {audit_result.timestamp.isoformat()}</li>
                    <li><strong>Execution Time:</strong> {audit_result.execution_time:.2f} seconds</li>
                    <li><strong>Framework Version:</strong> 1.0.0</li>
                </ul>
            </div>
            
            <div class="warnings">
                <h3>Warnings and Notes</h3>
                {self._format_warnings(audit_result.warnings)}
            </div>
            
            <div class="methodology">
                <h3>Methodology</h3>
                <p>This audit was conducted using the AI Ethics Auditor Sentinel framework, 
                which implements industry-standard fairness metrics and security assessments.</p>
                
                <h4>Bias Detection Methods:</h4>
                <ul>
                    <li>Demographic Parity Analysis</li>
                    <li>Equalized Odds Testing</li>
                    <li>Calibration Assessment</li>
                    <li>Statistical Significance Testing</li>
                </ul>
                
                <h4>Safety Assessment Methods:</h4>
                <ul>
                    <li>Adversarial Example Generation</li>
                    <li>Robustness Testing</li>
                    <li>Privacy Vulnerability Assessment</li>
                    <li>Input Manipulation Testing</li>
                </ul>
            </div>
        </div>
        '''
        
        return appendix_html
    
    def _format_warnings(self, warnings: List[str]) -> str:
        """Format warnings list."""
        
        if not warnings:
            return '<p class="no-warnings">No warnings generated during audit.</p>'
        
        warning_items = []
        for warning in warnings:
            warning_items.append(f'<li class="warning-item">⚠️ {warning}</li>')
        
        return f'<ul class="warnings-list">{"".join(warning_items)}</ul>'
    
    def _generate_visualizations(self, audit_result: AuditResult) -> str:
        """Generate visualization section with charts."""
        
        visualizations_html = '''
        <div class="visualizations">
            <h2>Visualizations</h2>
        '''
        
        # Risk distribution chart
        risk_chart = self._create_risk_distribution_chart(audit_result)
        if risk_chart:
            visualizations_html += f'''
            <div class="chart-container">
                <h3>Risk Distribution</h3>
                {risk_chart}
            </div>
            '''
        
        # Bias metrics chart
        if audit_result.bias_result:
            bias_chart = self._create_bias_metrics_chart(audit_result.bias_result)
            if bias_chart:
                visualizations_html += f'''
                <div class="chart-container">
                    <h3>Bias Metrics</h3>
                    {bias_chart}
                </div>
                '''
        
        # Safety metrics chart
        if audit_result.safety_result:
            safety_chart = self._create_safety_metrics_chart(audit_result.safety_result)
            if safety_chart:
                visualizations_html += f'''
                <div class="chart-container">
                    <h3>Safety Metrics</h3>
                    {safety_chart}
                </div>
                '''
        
        visualizations_html += '</div>'
        
        return visualizations_html
    
    def _create_risk_distribution_chart(self, audit_result: AuditResult) -> Optional[str]:
        """Create risk distribution pie chart."""
        
        try:
            risk_summary = audit_result.risk_summary
            
            # Filter out zero values
            non_zero_risks = {k: v for k, v in risk_summary.items() if v > 0}
            
            if not non_zero_risks:
                return None
            
            # Create plotly pie chart
            fig = go.Figure(data=[go.Pie(
                labels=list(non_zero_risks.keys()),
                values=list(non_zero_risks.values()),
                hole=0.3,
                marker_colors=[self.risk_colors.get(RiskLevel(k), '#999999') for k in non_zero_risks.keys()]
            )])
            
            fig.update_layout(
                title="Risk Level Distribution",
                showlegend=True,
                height=400
            )
            
            return fig.to_html(include_plotlyjs='inline', div_id="risk-chart")
        
        except Exception as e:
            logger.warning(f"Failed to create risk distribution chart: {e}")
            return None
    
    def _create_bias_metrics_chart(self, bias_result) -> Optional[str]:
        """Create bias metrics bar chart."""
        
        try:
            metrics = bias_result.fairness_metrics
            
            if not metrics:
                return None
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=list(metrics.keys()),
                    y=list(metrics.values()),
                    marker_color=self.colors["primary"]
                )
            ])
            
            fig.update_layout(
                title="Fairness Metrics",
                xaxis_title="Metric",
                yaxis_title="Value",
                height=400
            )
            
            return fig.to_html(include_plotlyjs='inline', div_id="bias-chart")
        
        except Exception as e:
            logger.warning(f"Failed to create bias metrics chart: {e}")
            return None
    
    def _create_safety_metrics_chart(self, safety_result) -> Optional[str]:
        """Create safety metrics radar chart."""
        
        try:
            metrics = safety_result.robustness_metrics
            
            if not metrics:
                return None
            
            # Create radar chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=list(metrics.values()),
                theta=list(metrics.keys()),
                fill='toself',
                name='Safety Metrics',
                line_color=self.colors["secondary"]
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                title="Safety and Robustness Metrics",
                height=400
            )
            
            return fig.to_html(include_plotlyjs='inline', div_id="safety-chart")
        
        except Exception as e:
            logger.warning(f"Failed to create safety metrics chart: {e}")
            return None
    
    def _get_html_template(self) -> Template:
        """Get the main HTML template."""
        
        template_str = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Ethics Audit Report</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <style>
        :root {
            --primary-color: {{ colors.primary }};
            --secondary-color: {{ colors.secondary }};
            --success-color: {{ colors.success }};
            --warning-color: {{ colors.warning }};
            --danger-color: {{ colors.danger }};
            --text-color: {{ colors.text }};
            --bg-color: {{ colors.background }};
            --light-gray: {{ colors.light_gray }};
            --medium-gray: {{ colors.medium_gray }};
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: var(--text-color);
            background-color: var(--bg-color);
            line-height: 1.6;
        }
        
        .report-header {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 2rem 0;
            margin-bottom: 2rem;
        }
        
        .report-title {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        
        .report-subtitle {
            font-size: 1.2rem;
            opacity: 0.9;
            margin-bottom: 1rem;
        }
        
        .score-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
            backdrop-filter: blur(10px);
        }
        
        .score-value {
            font-size: 3rem;
            font-weight: bold;
            line-height: 1;
        }
        
        .score-label {
            font-size: 1rem;
            opacity: 0.8;
            margin: 0.5rem 0;
        }
        
        .score-status {
            font-size: 1.2rem;
            font-weight: bold;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            margin-top: 1rem;
        }
        
        .score-status.pass {
            background-color: var(--success-color);
        }
        
        .score-status.fail {
            background-color: var(--danger-color);
        }
        
        .assessment-banner {
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            margin: 1rem 0 2rem 0;
            font-weight: bold;
        }
        
        .assessment-banner.critical {
            background-color: var(--danger-color);
            color: white;
        }
        
        .assessment-banner.high {
            background-color: var(--warning-color);
            color: white;
        }
        
        .assessment-banner.medium {
            background-color: var(--success-color);
            color: white;
        }
        
        .assessment-banner.low {
            background-color: var(--primary-color);
            color: white;
        }
        
        .summary-card, .metric-card {
            background: var(--light-gray);
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .metric-card {
            text-align: center;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: var(--primary-color);
        }
        
        .metric-label {
            font-size: 1.1rem;
            font-weight: 600;
            margin: 0.5rem 0;
        }
        
        .metric-description {
            font-size: 0.9rem;
            color: #666;
        }
        
        .risk-matrix {
            margin: 2rem 0;
        }
        
        .risk-card {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid;
        }
        
        .risk-card.critical {
            border-left-color: {{ risk_colors[RiskLevel.CRITICAL] }};
        }
        
        .risk-card.high {
            border-left-color: {{ risk_colors[RiskLevel.HIGH] }};
        }
        
        .risk-card.medium {
            border-left-color: {{ risk_colors[RiskLevel.MEDIUM] }};
        }
        
        .risk-card.low {
            border-left-color: {{ risk_colors[RiskLevel.LOW] }};
        }
        
        .risk-count {
            font-size: 2rem;
            font-weight: bold;
        }
        
        .risk-label {
            font-size: 1rem;
            text-transform: uppercase;
            font-weight: 600;
            margin-top: 0.5rem;
        }
        
        .violations-table {
            margin-top: 1rem;
        }
        
        .risk-badge {
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: bold;
            color: white;
            text-transform: uppercase;
        }
        
        .risk-badge.critical {
            background-color: {{ risk_colors[RiskLevel.CRITICAL] }};
        }
        
        .risk-badge.high {
            background-color: {{ risk_colors[RiskLevel.HIGH] }};
        }
        
        .risk-badge.medium {
            background-color: {{ risk_colors[RiskLevel.MEDIUM] }};
        }
        
        .risk-badge.low {
            background-color: {{ risk_colors[RiskLevel.LOW] }};
        }
        
        .chart-container {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 2rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .no-violations {
            background: var(--light-gray);
            padding: 1rem;
            border-radius: 5px;
            text-align: center;
            color: var(--success-color);
            font-weight: 600;
        }
        
        .recommendations-list {
            list-style: none;
            padding: 0;
        }
        
        .recommendation-item {
            background: var(--light-gray);
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 5px;
            border-left: 4px solid var(--primary-color);
        }
        
        .attack-vector-list {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }
        
        .attack-vector-item {
            background: var(--primary-color);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.9rem;
            list-style: none;
        }
        
        .technical-details pre {
            background: var(--light-gray);
            padding: 1rem;
            border-radius: 5px;
            font-size: 0.9rem;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .warnings-list {
            list-style: none;
            padding: 0;
        }
        
        .warning-item {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 0.75rem;
            margin: 0.5rem 0;
            border-radius: 5px;
        }
        
        .no-warnings {
            background: var(--light-gray);
            padding: 1rem;
            border-radius: 5px;
            text-align: center;
            color: var(--success-color);
        }
        
        section {
            margin-bottom: 3rem;
        }
        
        h2 {
            color: var(--primary-color);
            border-bottom: 2px solid var(--primary-color);
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
        }
        
        .container {
            max-width: 1200px;
        }
        
        @media print {
            .chart-container {
                break-inside: avoid;
            }
        }
    </style>
</head>
<body>
    <div class="container-fluid">
        {{ sections.header | safe }}
        
        <div class="container">
            <section>
                {{ sections.executive_summary | safe }}
            </section>
            
            <section>
                {{ sections.risk_overview | safe }}
            </section>
            
            <section>
                {{ sections.bias_analysis | safe }}
            </section>
            
            <section>
                {{ sections.safety_analysis | safe }}
            </section>
            
            {% if config.include_visualizations and sections.visualizations %}
            <section>
                {{ sections.visualizations | safe }}
            </section>
            {% endif %}
            
            <section>
                {{ sections.recommendations | safe }}
            </section>
            
            {% if config.detailed_explanations %}
            <section>
                {{ sections.technical_details | safe }}
            </section>
            
            <section>
                {{ sections.appendix | safe }}
            </section>
            {% endif %}
        </div>
    </div>
    
    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    
    <script>
        // Add any interactive functionality here
        document.addEventListener('DOMContentLoaded', function() {
            // Initialize tooltips
            var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
            var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
                return new bootstrap.Tooltip(tooltipTriggerEl);
            });
        });
    </script>
</body>
</html>
        '''
        
        return Template(template_str)