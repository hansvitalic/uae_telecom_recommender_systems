"""
UAE Telecom Recommender Systems

A comprehensive sector-aware recommender system for project risk management
in UAE telecom infrastructure.

This package provides:
- Sector-specific risk assessment models
- Project recommendation algorithms
- Risk mitigation strategy recommendations
- Performance monitoring and analytics
"""

__version__ = "1.0.0"
__author__ = "Bet_Hans"

from .core.recommender import TelecomRecommender
from .core.risk_assessor import RiskAssessor
from .models.project import Project, ProjectRisk
from .models.sector import TelecomSector

__all__ = [
    "TelecomRecommender",
    "RiskAssessor", 
    "Project",
    "ProjectRisk",
    "TelecomSector",
]