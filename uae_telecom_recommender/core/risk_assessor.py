"""
Risk Assessment Engine

Provides comprehensive risk assessment capabilities for UAE telecom projects.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from ..models.project import Project, ProjectRisk, RiskCategory, RiskLevel
from ..models.sector import TelecomSector, TelecomSectorType, get_sector_by_type


@dataclass
class RiskAssessmentResult:
    """Result of risk assessment analysis"""
    
    overall_risk_score: float
    risk_level: RiskLevel
    top_risks: List[ProjectRisk]
    risk_distribution: Dict[RiskCategory, int]
    recommendations: List[str]
    confidence_score: float


class RiskAssessor:
    """
    Advanced risk assessment engine for UAE telecom projects.
    
    Provides sector-specific risk analysis, predictive risk modeling,
    and mitigation strategy recommendations.
    """
    
    def __init__(self):
        self.sector_risk_weights = self._initialize_sector_weights()
        self.historical_risk_data = {}
        
    def _initialize_sector_weights(self) -> Dict[TelecomSectorType, Dict[RiskCategory, float]]:
        """Initialize sector-specific risk category weights"""
        return {
            TelecomSectorType.MOBILE_NETWORKS: {
                RiskCategory.TECHNICAL: 0.25,
                RiskCategory.REGULATORY: 0.20,
                RiskCategory.FINANCIAL: 0.15,
                RiskCategory.OPERATIONAL: 0.15,
                RiskCategory.SECURITY: 0.15,
                RiskCategory.MARKET: 0.10,
                RiskCategory.ENVIRONMENTAL: 0.05,
                RiskCategory.STAKEHOLDER: 0.05
            },
            TelecomSectorType.FIBER_OPTIC: {
                RiskCategory.OPERATIONAL: 0.30,
                RiskCategory.REGULATORY: 0.25,
                RiskCategory.TECHNICAL: 0.20,
                RiskCategory.FINANCIAL: 0.15,
                RiskCategory.ENVIRONMENTAL: 0.10,
                RiskCategory.SECURITY: 0.05,
                RiskCategory.MARKET: 0.05,
                RiskCategory.STAKEHOLDER: 0.05
            },
            TelecomSectorType.IOT_M2M: {
                RiskCategory.SECURITY: 0.30,
                RiskCategory.TECHNICAL: 0.25,
                RiskCategory.MARKET: 0.20,
                RiskCategory.REGULATORY: 0.15,
                RiskCategory.OPERATIONAL: 0.10,
                RiskCategory.FINANCIAL: 0.05,
                RiskCategory.ENVIRONMENTAL: 0.05,
                RiskCategory.STAKEHOLDER: 0.05
            }
        }
    
    def assess_project_risk(self, project: Project) -> RiskAssessmentResult:
        """
        Perform comprehensive risk assessment for a project.
        
        Args:
            project: Project to assess
            
        Returns:
            RiskAssessmentResult with detailed analysis
        """
        # Calculate weighted risk score
        overall_score = self._calculate_weighted_risk_score(project)
        
        # Determine risk level
        risk_level = self._determine_risk_level(overall_score)
        
        # Get top risks
        top_risks = self._get_top_risks(project, limit=5)
        
        # Calculate risk distribution
        risk_distribution = self._calculate_risk_distribution(project)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(project, overall_score, top_risks)
        
        # Calculate confidence score
        confidence = self._calculate_confidence_score(project)
        
        return RiskAssessmentResult(
            overall_risk_score=overall_score,
            risk_level=risk_level,
            top_risks=top_risks,
            risk_distribution=risk_distribution,
            recommendations=recommendations,
            confidence_score=confidence
        )
    
    def _calculate_weighted_risk_score(self, project: Project) -> float:
        """Calculate weighted risk score based on sector-specific weights"""
        if not project.risks:
            return 0.0
        
        sector_weights = self.sector_risk_weights.get(
            project.sector, 
            {category: 1.0/len(RiskCategory) for category in RiskCategory}
        )
        
        weighted_scores = []
        for risk in project.risks:
            weight = sector_weights.get(risk.category, 1.0)
            risk_score = risk.calculate_risk_score()
            weighted_scores.append(risk_score * weight)
        
        # Include project complexity and strategic importance
        base_score = np.mean(weighted_scores) if weighted_scores else 0.0
        complexity_factor = 1 + (project.complexity_score - 5) * 0.1
        importance_factor = 1 + (project.strategic_importance - 5) * 0.05
        
        return min(5.0, base_score * complexity_factor * importance_factor)
    
    def _determine_risk_level(self, score: float) -> RiskLevel:
        """Determine risk level from numerical score"""
        if score <= 1.5:
            return RiskLevel.VERY_LOW
        elif score <= 2.5:
            return RiskLevel.LOW
        elif score <= 3.5:
            return RiskLevel.MEDIUM
        elif score <= 4.5:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH
    
    def _get_top_risks(self, project: Project, limit: int = 5) -> List[ProjectRisk]:
        """Get top risks sorted by risk score"""
        sorted_risks = sorted(
            project.risks,
            key=lambda r: r.calculate_risk_score(),
            reverse=True
        )
        return sorted_risks[:limit]
    
    def _calculate_risk_distribution(self, project: Project) -> Dict[RiskCategory, int]:
        """Calculate distribution of risks by category"""
        distribution = {category: 0 for category in RiskCategory}
        for risk in project.risks:
            distribution[risk.category] += 1
        return distribution
    
    def _generate_recommendations(
        self, 
        project: Project, 
        overall_score: float, 
        top_risks: List[ProjectRisk]
    ) -> List[str]:
        """Generate risk mitigation recommendations"""
        recommendations = []
        
        # General recommendations based on overall score
        if overall_score > 4.0:
            recommendations.append("Immediate risk mitigation required - consider project timeline review")
            recommendations.append("Establish daily risk monitoring and reporting")
        elif overall_score > 3.0:
            recommendations.append("Enhanced risk monitoring recommended")
            recommendations.append("Weekly risk review meetings suggested")
        
        # Sector-specific recommendations
        sector = get_sector_by_type(project.sector)
        if sector:
            recommendations.extend(self._get_sector_specific_recommendations(project.sector))
        
        # Risk-specific recommendations
        for risk in top_risks[:3]:  # Top 3 risks
            if risk.mitigation_strategies:
                recommendations.append(f"For {risk.title}: {risk.mitigation_strategies[0]}")
        
        # UAE-specific recommendations
        recommendations.extend(self._get_uae_specific_recommendations(project))
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _get_sector_specific_recommendations(self, sector_type: TelecomSectorType) -> List[str]:
        """Get recommendations specific to telecom sector"""
        sector_recommendations = {
            TelecomSectorType.MOBILE_NETWORKS: [
                "Ensure TDRA spectrum compliance documentation is complete",
                "Implement network security measures per UAE cybersecurity framework"
            ],
            TelecomSectorType.FIBER_OPTIC: [
                "Coordinate with Dubai Municipality for right-of-way permissions",
                "Plan installation around UAE weather patterns"
            ],
            TelecomSectorType.IOT_M2M: [
                "Ensure device certification with TDRA",
                "Implement UAE data localization requirements"
            ]
        }
        return sector_recommendations.get(sector_type, [])
    
    def _get_uae_specific_recommendations(self, project: Project) -> List[str]:
        """Get UAE market-specific recommendations"""
        recommendations = []
        
        # Emirates-specific considerations
        if "Dubai" in project.emirates:
            recommendations.append("Leverage Dubai Smart City initiative alignment opportunities")
        if "Abu Dhabi" in project.emirates:
            recommendations.append("Consider ADNOC and government sector requirements")
        
        # Weather and environmental considerations
        recommendations.append("Plan critical installations outside peak summer months")
        recommendations.append("Implement sand storm protection measures for equipment")
        
        return recommendations
    
    def _calculate_confidence_score(self, project: Project) -> float:
        """Calculate confidence in risk assessment"""
        factors = []
        
        # Number of identified risks
        risk_count_factor = min(1.0, len(project.risks) / 10)
        factors.append(risk_count_factor)
        
        # Project maturity (more mature = higher confidence)
        maturity_map = {"emerging": 0.6, "developing": 0.7, "proven": 0.9, "legacy": 0.8}
        maturity_factor = maturity_map.get(project.technology_maturity, 0.7)
        factors.append(maturity_factor)
        
        # Historical data availability (simulated)
        historical_factor = 0.8  # Would be based on actual historical data
        factors.append(historical_factor)
        
        return np.mean(factors)
    
    def compare_projects_risk(self, projects: List[Project]) -> Dict[str, RiskAssessmentResult]:
        """Compare risk assessments across multiple projects"""
        results = {}
        for project in projects:
            results[project.project_id] = self.assess_project_risk(project)
        return results
    
    def predict_emerging_risks(self, project: Project) -> List[str]:
        """Predict potential emerging risks based on project characteristics"""
        emerging_risks = []
        
        # Technology-based predictions
        if project.technology_maturity == "emerging":
            emerging_risks.append("Technology adoption challenges")
            emerging_risks.append("Skills gap and training requirements")
        
        # Sector-based predictions
        if project.sector == TelecomSectorType.IOT_M2M:
            emerging_risks.append("Device interoperability issues")
            emerging_risks.append("Privacy compliance challenges")
        
        # UAE market predictions
        if project.budget_aed > 100_000_000:
            emerging_risks.append("Currency fluctuation impact")
            emerging_risks.append("Large project coordination complexity")
        
        return emerging_risks