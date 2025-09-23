"""
ML Models Module
Machine Learning components for risk prediction and analysis
"""

__version__ = "1.0.0"
__author__ = "UAE Telecom RMS Team"

from .feature_engineering import FeatureEngineer
from .risk_predictor import RiskPredictor

__all__ = [
    'FeatureEngineer',
    'RiskPredictor'
]