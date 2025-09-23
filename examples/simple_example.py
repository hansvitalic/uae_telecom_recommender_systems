#!/usr/bin/env python3
"""
Simple Usage Example for UAE Telecom Recommender System
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta
from uae_telecom_recommender import TelecomRecommender, RiskAssessor, Project, ProjectRisk
from uae_telecom_recommender.models.project import ProjectStatus, RiskLevel, RiskCategory
from uae_telecom_recommender.models.sector import TelecomSectorType
from uae_telecom_recommender.utils import format_currency_aed


def main():
    """Simple example demonstrating the recommender system"""
    print("UAE Telecom Recommender System - Simple Example")
    print("=" * 50)
    
    # Create a sample project
    project = Project(
        project_id="UAE_5G_DXB_001",
        name="5G Network Rollout - Dubai Marina",
        description="Deployment of 5G infrastructure in Dubai Marina area",
        sector=TelecomSectorType.MOBILE_NETWORKS,
        status=ProjectStatus.PLANNING,
        budget_aed=75_000_000,
        start_date=date.today(),
        planned_end_date=date.today() + timedelta(days=365),
        project_manager="Sara Al Mansoori",
        sponsor="Etisalat UAE",
        emirates=["Dubai"],
        complexity_score=7.5,
        strategic_importance=8.0
    )
    
    # Add a risk
    project.add_risk(ProjectRisk(
        risk_id="RISK_001",
        title="Spectrum Interference",
        description="Potential interference from existing 4G networks",
        category=RiskCategory.TECHNICAL,
        probability=0.4,
        impact=RiskLevel.HIGH,
        risk_level=RiskLevel.MEDIUM,
        mitigation_strategies=["Frequency coordination with TDRA"]
    ))
    
    # Initialize recommender
    recommender = TelecomRecommender()
    
    # Get recommendations
    recommendations = recommender.get_project_recommendations(project)
    
    # Display results
    print(f"Project: {project.name}")
    print(f"Budget: {format_currency_aed(project.budget_aed)}")
    print(f"Risk Score: {project.calculate_overall_risk_score():.2f}")
    print(f"\nTop 3 Recommendations:")
    
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"{i}. {rec.title}")
        print(f"   Priority: {rec.priority}/5")
        print(f"   {rec.description}")
        print()
    
    print("✅ Example completed successfully!")


if __name__ == "__main__":
    main()