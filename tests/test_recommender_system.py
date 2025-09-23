"""
Test suite for UAE Telecom Recommender Systems

Tests core functionality of the recommender system including risk assessment,
project recommendations, and sector-specific features.
"""

import pytest
from datetime import date, datetime
import numpy as np
from uae_telecom_recommender.models.project import (
    Project, ProjectRisk, ProjectStatus, RiskLevel, RiskCategory,
    create_sample_project
)
from uae_telecom_recommender.models.sector import (
    TelecomSectorType, get_sector_by_type, UAE_TELECOM_SECTORS
)
from uae_telecom_recommender.core.risk_assessor import RiskAssessor
from uae_telecom_recommender.core.recommender import TelecomRecommender
from uae_telecom_recommender.config.config_manager import ConfigManager
from uae_telecom_recommender.utils.helpers import (
    create_sample_dataset, calculate_portfolio_metrics
)


class TestProjectModels:
    """Test project and risk models"""
    
    def test_project_creation(self):
        """Test basic project creation"""
        project = create_sample_project()
        
        assert project.project_id == "UAE_TEL_001"
        assert project.name == "5G Network Expansion - Dubai"
        assert project.sector == TelecomSectorType.MOBILE_NETWORKS
        assert project.status == ProjectStatus.IN_PROGRESS
        assert len(project.risks) >= 2
        
    def test_risk_calculation(self):
        """Test risk score calculations"""
        risk = ProjectRisk(
            risk_id="TEST_001",
            title="Test Risk",
            description="Test description",
            category=RiskCategory.TECHNICAL,
            probability=0.5,
            impact=RiskLevel.HIGH,
            risk_level=RiskLevel.HIGH
        )
        
        # Risk score = probability * impact value
        expected_score = 0.5 * 4  # HIGH = 4
        assert risk.calculate_risk_score() == expected_score
        
    def test_project_risk_assessment(self):
        """Test project-level risk assessment"""
        project = create_sample_project()
        overall_score = project.calculate_overall_risk_score()
        
        assert 0 <= overall_score <= 5
        assert len(project.get_high_risks()) >= 0
        
    def test_project_progress_calculation(self):
        """Test project progress calculation"""
        project = create_sample_project()
        progress = project.get_progress_percentage()
        
        assert 0 <= progress <= 100
        
    def test_project_overdue_detection(self):
        """Test overdue project detection"""
        project = Project(
            project_id="TEST_002",
            name="Test Overdue Project",
            description="Test",
            sector=TelecomSectorType.MOBILE_NETWORKS,
            status=ProjectStatus.IN_PROGRESS,
            budget_aed=1_000_000,
            start_date=date(2023, 1, 1),
            planned_end_date=date(2023, 6, 1),  # Past date
            project_manager="Test Manager",
            sponsor="Test Sponsor"
        )
        
        assert project.is_overdue() == True


class TestSectorModels:
    """Test sector-specific models"""
    
    def test_sector_retrieval(self):
        """Test sector configuration retrieval"""
        mobile_sector = get_sector_by_type(TelecomSectorType.MOBILE_NETWORKS)
        
        assert mobile_sector is not None
        assert mobile_sector.name == "Mobile Networks"
        assert len(mobile_sector.characteristics.regulatory_requirements) > 0
        
    def test_all_sectors_defined(self):
        """Test that all major UAE telecom sectors are defined"""
        expected_sectors = [
            TelecomSectorType.MOBILE_NETWORKS,
            TelecomSectorType.FIXED_LINE,
            TelecomSectorType.FIBER_OPTIC,
            TelecomSectorType.IOT_M2M
        ]
        
        for sector_type in expected_sectors:
            sector = get_sector_by_type(sector_type)
            assert sector is not None
            assert sector.regulatory_body == "TDRA"


class TestRiskAssessor:
    """Test risk assessment engine"""
    
    def test_risk_assessor_initialization(self):
        """Test risk assessor initialization"""
        assessor = RiskAssessor()
        
        assert assessor.sector_risk_weights is not None
        assert len(assessor.sector_risk_weights) > 0
        
    def test_project_risk_assessment(self):
        """Test comprehensive project risk assessment"""
        assessor = RiskAssessor()
        project = create_sample_project()
        
        result = assessor.assess_project_risk(project)
        
        assert result.overall_risk_score >= 0
        assert result.risk_level in RiskLevel
        assert len(result.recommendations) > 0
        assert 0 <= result.confidence_score <= 1
        
    def test_risk_level_determination(self):
        """Test risk level determination from scores"""
        assessor = RiskAssessor()
        
        assert assessor._determine_risk_level(1.0) == RiskLevel.VERY_LOW
        assert assessor._determine_risk_level(2.0) == RiskLevel.LOW
        assert assessor._determine_risk_level(3.0) == RiskLevel.MEDIUM
        assert assessor._determine_risk_level(4.0) == RiskLevel.HIGH
        assert assessor._determine_risk_level(5.0) == RiskLevel.VERY_HIGH
        
    def test_sector_specific_recommendations(self):
        """Test sector-specific risk recommendations"""
        assessor = RiskAssessor()
        
        mobile_recs = assessor._get_sector_specific_recommendations(
            TelecomSectorType.MOBILE_NETWORKS
        )
        fiber_recs = assessor._get_sector_specific_recommendations(
            TelecomSectorType.FIBER_OPTIC
        )
        
        assert len(mobile_recs) > 0
        assert len(fiber_recs) > 0
        assert mobile_recs != fiber_recs  # Should be different


class TestTelecomRecommender:
    """Test main recommender system"""
    
    def test_recommender_initialization(self):
        """Test recommender initialization"""
        recommender = TelecomRecommender()
        
        assert recommender.risk_assessor is not None
        assert recommender.recommendation_weights is not None
        assert recommender.sector_expertise is not None
        
    def test_project_recommendations(self):
        """Test project recommendation generation"""
        recommender = TelecomRecommender()
        project = create_sample_project()
        
        recommendations = recommender.get_project_recommendations(project)
        
        assert len(recommendations) > 0
        assert all(rec.project_id == project.project_id for rec in recommendations)
        assert all(1 <= rec.priority <= 5 for rec in recommendations)
        assert all(0 <= rec.success_probability <= 1 for rec in recommendations)
        
    def test_portfolio_analysis(self):
        """Test portfolio-level analysis"""
        recommender = TelecomRecommender()
        projects = create_sample_dataset(10)
        
        insights = recommender.analyze_portfolio(projects)
        
        assert insights.total_projects == 10
        assert insights.total_budget_aed > 0
        assert len(insights.key_insights) > 0
        assert len(insights.strategic_recommendations) > 0
        
    def test_similar_projects_finding(self):
        """Test finding similar projects"""
        recommender = TelecomRecommender()
        projects = create_sample_dataset(5)
        target_project = projects[0]
        other_projects = projects[1:]
        
        similar_projects = recommender.find_similar_projects(target_project, other_projects)
        
        assert len(similar_projects) <= len(other_projects)
        assert all(0 <= similarity <= 1 for _, similarity in similar_projects)


class TestConfigManager:
    """Test configuration management"""
    
    def test_config_manager_initialization(self):
        """Test config manager initialization"""
        config_manager = ConfigManager()
        
        assert config_manager.config is not None
        assert config_manager.sector_configs is not None
        
    def test_default_config_values(self):
        """Test default configuration values"""
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        assert config.default_regulatory_body == "TDRA"
        assert config.default_currency == "AED"
        assert len(config.major_emirates) == 7
        
    def test_sector_config_retrieval(self):
        """Test sector configuration retrieval"""
        config_manager = ConfigManager()
        
        mobile_config = config_manager.get_sector_config(TelecomSectorType.MOBILE_NETWORKS)
        
        assert mobile_config is not None
        assert mobile_config.sector_type == TelecomSectorType.MOBILE_NETWORKS
        assert len(mobile_config.risk_weights) > 0
        
    def test_uae_market_parameters(self):
        """Test UAE market parameters"""
        config_manager = ConfigManager()
        market_params = config_manager.get_uae_market_parameters()
        
        assert "Etisalat" in market_params["major_operators"]
        assert "du" in market_params["major_operators"]
        assert market_params["regulatory_body"] == "TDRA"
        assert market_params["currency"] == "AED"
        
    def test_config_validation(self):
        """Test configuration validation"""
        config_manager = ConfigManager()
        validation_results = config_manager.validate_config()
        
        # All validations should pass for default config
        assert all(validation_results.values())


class TestUtilities:
    """Test utility functions"""
    
    def test_sample_dataset_creation(self):
        """Test sample dataset creation"""
        projects = create_sample_dataset(5)
        
        assert len(projects) == 5
        assert all(isinstance(p, Project) for p in projects)
        assert all(len(p.risks) > 0 for p in projects)
        
    def test_portfolio_metrics_calculation(self):
        """Test portfolio metrics calculation"""
        projects = create_sample_dataset(10)
        metrics = calculate_portfolio_metrics(projects)
        
        assert metrics["overview"]["total_projects"] == 10
        assert metrics["overview"]["total_budget_aed"] > 0
        assert "risk_metrics" in metrics
        assert "progress_metrics" in metrics
        assert "distribution" in metrics
        
    def test_empty_portfolio_handling(self):
        """Test handling of empty project portfolio"""
        metrics = calculate_portfolio_metrics([])
        
        assert metrics == {}


class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        # Create sample data
        projects = create_sample_dataset(3)
        target_project = projects[0]
        
        # Initialize system components
        config_manager = ConfigManager()
        recommender = TelecomRecommender()
        
        # Get project recommendations
        recommendations = recommender.get_project_recommendations(target_project)
        
        # Analyze portfolio
        portfolio_insights = recommender.analyze_portfolio(projects)
        
        # Verify results
        assert len(recommendations) > 0
        assert portfolio_insights.total_projects == 3
        assert len(portfolio_insights.key_insights) > 0
        
    def test_uae_specific_features(self):
        """Test UAE-specific features"""
        project = Project(
            project_id="UAE_TEST_001",
            name="Dubai Smart City Project",
            description="Test UAE project",
            sector=TelecomSectorType.MOBILE_NETWORKS,
            status=ProjectStatus.PLANNING,
            budget_aed=100_000_000,
            start_date=date.today(),
            planned_end_date=date(2024, 12, 31),
            project_manager="Ahmed Al Rashid",
            sponsor="Dubai Municipality",
            emirates=["Dubai"],
            regulatory_contact="TDRA"
        )
        
        recommender = TelecomRecommender()
        recommendations = recommender.get_project_recommendations(project)
        
        # Should include UAE-specific recommendations
        uae_recommendations = [
            rec for rec in recommendations 
            if "TDRA" in rec.description or "Dubai" in rec.description
        ]
        
        assert len(uae_recommendations) > 0


if __name__ == "__main__":
    # Run basic tests
    print("Running UAE Telecom Recommender System Tests...")
    
    # Test project creation
    project = create_sample_project()
    print(f"✓ Created sample project: {project.name}")
    
    # Test risk assessment
    assessor = RiskAssessor()
    risk_result = assessor.assess_project_risk(project)
    print(f"✓ Risk assessment completed. Overall score: {risk_result.overall_risk_score:.2f}")
    
    # Test recommendations
    recommender = TelecomRecommender()
    recommendations = recommender.get_project_recommendations(project)
    print(f"✓ Generated {len(recommendations)} recommendations")
    
    # Test portfolio analysis
    projects = create_sample_dataset(5)
    portfolio_insights = recommender.analyze_portfolio(projects)
    print(f"✓ Portfolio analysis completed for {portfolio_insights.total_projects} projects")
    
    print("\nAll basic tests passed! ✅")
    print("\nRun 'pytest tests/' for comprehensive test suite.")