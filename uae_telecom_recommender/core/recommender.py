"""
Telecom Recommender System

Main recommender engine that provides project recommendations, risk mitigation
strategies, and decision support for UAE telecom infrastructure projects.
"""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from datetime import date, timedelta
from ..models.project import Project, ProjectStatus, ProjectRisk
from ..models.sector import TelecomSector, TelecomSectorType, get_sector_by_type, get_all_sectors
from .risk_assessor import RiskAssessor, RiskAssessmentResult


@dataclass
class ProjectRecommendation:
    """Recommendation for a specific project"""
    
    project_id: str
    recommendation_type: str  # mitigation, optimization, strategic
    title: str
    description: str
    priority: int  # 1-5, 5 being highest
    impact: str  # low, medium, high
    effort_required: str  # low, medium, high
    timeline: str  # immediate, short-term, long-term
    success_probability: float  # 0.0 to 1.0
    
    # UAE-specific considerations
    regulatory_impact: str = "none"  # none, low, medium, high
    local_stakeholders: List[str] = None
    
    def __post_init__(self):
        if self.local_stakeholders is None:
            self.local_stakeholders = []


@dataclass
class PortfolioInsight:
    """Portfolio-level insights and recommendations"""
    
    total_projects: int
    total_budget_aed: float
    high_risk_projects: int
    sector_distribution: Dict[TelecomSectorType, int]
    average_risk_score: float
    key_insights: List[str]
    strategic_recommendations: List[str]


class TelecomRecommender:
    """
    Advanced recommender system for UAE telecom project management.
    
    Provides intelligent recommendations for:
    - Risk mitigation strategies
    - Project optimization opportunities
    - Strategic decision support
    - Portfolio management
    """
    
    def __init__(self):
        self.risk_assessor = RiskAssessor()
        self.recommendation_weights = self._initialize_recommendation_weights()
        self.sector_expertise = self._load_sector_expertise()
        
    def _initialize_recommendation_weights(self) -> Dict[str, float]:
        """Initialize weights for different recommendation factors"""
        return {
            "risk_score": 0.3,
            "strategic_importance": 0.25,
            "budget_impact": 0.2,
            "complexity": 0.15,
            "urgency": 0.1
        }
    
    def _load_sector_expertise(self) -> Dict[TelecomSectorType, Dict[str, Any]]:
        """Load sector-specific expertise and best practices"""
        return {
            TelecomSectorType.MOBILE_NETWORKS: {
                "best_practices": [
                    "Implement network slicing for enterprise customers",
                    "Use AI-driven network optimization",
                    "Deploy edge computing capabilities"
                ],
                "common_pitfalls": [
                    "Inadequate spectrum planning",
                    "Insufficient security measures",
                    "Poor coverage optimization"
                ],
                "success_factors": [
                    "Strong vendor partnerships",
                    "Regulatory compliance",
                    "Customer experience focus"
                ]
            },
            TelecomSectorType.FIBER_OPTIC: {
                "best_practices": [
                    "Use micro-trenching for urban deployment",
                    "Implement fiber-to-the-home strategy",
                    "Deploy smart fiber management systems"
                ],
                "common_pitfalls": [
                    "Underestimating permitting timelines",
                    "Poor route planning",
                    "Inadequate weather protection"
                ],
                "success_factors": [
                    "Municipal partnerships",
                    "Efficient installation processes",
                    "Quality construction practices"
                ]
            },
            TelecomSectorType.IOT_M2M: {
                "best_practices": [
                    "Use low-power wide-area networks",
                    "Implement device lifecycle management",
                    "Focus on security by design"
                ],
                "common_pitfalls": [
                    "Ignoring battery life optimization",
                    "Poor device interoperability",
                    "Inadequate data privacy measures"
                ],
                "success_factors": [
                    "Platform scalability",
                    "Partner ecosystem development",
                    "Vertical market focus"
                ]
            }
        }
    
    def get_project_recommendations(self, project: Project) -> List[ProjectRecommendation]:
        """
        Generate comprehensive recommendations for a specific project.
        
        Args:
            project: Project to analyze
            
        Returns:
            List of prioritized recommendations
        """
        recommendations = []
        
        # Get risk assessment
        risk_assessment = self.risk_assessor.assess_project_risk(project)
        
        # Generate risk-based recommendations
        recommendations.extend(self._generate_risk_recommendations(project, risk_assessment))
        
        # Generate optimization recommendations
        recommendations.extend(self._generate_optimization_recommendations(project))
        
        # Generate strategic recommendations
        recommendations.extend(self._generate_strategic_recommendations(project))
        
        # Generate UAE-specific recommendations
        recommendations.extend(self._generate_uae_recommendations(project))
        
        # Sort by priority and return top recommendations
        recommendations.sort(key=lambda x: x.priority, reverse=True)
        return recommendations[:15]  # Top 15 recommendations
    
    def _generate_risk_recommendations(
        self, 
        project: Project, 
        risk_assessment: RiskAssessmentResult
    ) -> List[ProjectRecommendation]:
        """Generate recommendations based on risk assessment"""
        recommendations = []
        
        for risk in risk_assessment.top_risks[:3]:  # Top 3 risks
            if risk.mitigation_strategies:
                priority = 5 if risk.risk_level.value >= 4 else 3
                
                recommendation = ProjectRecommendation(
                    project_id=project.project_id,
                    recommendation_type="mitigation",
                    title=f"Mitigate {risk.title}",
                    description=risk.mitigation_strategies[0],
                    priority=priority,
                    impact="high" if risk.impact.value >= 4 else "medium",
                    effort_required=self._estimate_effort(risk),
                    timeline="immediate" if priority == 5 else "short-term",
                    success_probability=0.8,
                    regulatory_impact=self._assess_regulatory_impact(risk)
                )
                recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_optimization_recommendations(self, project: Project) -> List[ProjectRecommendation]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Budget optimization
        if project.budget_aed > 50_000_000:
            recommendations.append(ProjectRecommendation(
                project_id=project.project_id,
                recommendation_type="optimization",
                title="Implement Value Engineering",
                description="Conduct value engineering analysis to optimize costs while maintaining quality",
                priority=3,
                impact="high",
                effort_required="medium",
                timeline="short-term",
                success_probability=0.7
            ))
        
        # Timeline optimization
        if project.is_overdue():
            recommendations.append(ProjectRecommendation(
                project_id=project.project_id,
                recommendation_type="optimization",
                title="Accelerate Project Timeline",
                description="Implement fast-track delivery methods and parallel processing",
                priority=4,
                impact="high",
                effort_required="high",
                timeline="immediate",
                success_probability=0.6
            ))
        
        # Technology optimization
        sector_expertise = self.sector_expertise.get(project.sector, {})
        best_practices = sector_expertise.get("best_practices", [])
        
        for practice in best_practices[:2]:  # Top 2 best practices
            recommendations.append(ProjectRecommendation(
                project_id=project.project_id,
                recommendation_type="optimization",
                title=f"Implement Best Practice: {practice}",
                description=f"Apply industry best practice: {practice}",
                priority=2,
                impact="medium",
                effort_required="medium",
                timeline="long-term",
                success_probability=0.8
            ))
        
        return recommendations
    
    def _generate_strategic_recommendations(self, project: Project) -> List[ProjectRecommendation]:
        """Generate strategic recommendations"""
        recommendations = []
        
        # Strategic importance considerations
        if project.strategic_importance >= 8.0:
            recommendations.append(ProjectRecommendation(
                project_id=project.project_id,
                recommendation_type="strategic",
                title="Establish Executive Steering Committee",
                description="Form high-level steering committee for strategic oversight",
                priority=4,
                impact="high",
                effort_required="low",
                timeline="immediate",
                success_probability=0.9,
                regulatory_impact="medium",
                local_stakeholders=["TDRA", "Dubai Smart City Office"]
            ))
        
        # Innovation opportunities
        if project.technology_maturity == "emerging":
            recommendations.append(ProjectRecommendation(
                project_id=project.project_id,
                recommendation_type="strategic",
                title="Establish Innovation Partnership",
                description="Partner with UAE universities and research centers for R&D",
                priority=2,
                impact="medium",
                effort_required="medium",
                timeline="long-term",
                success_probability=0.7,
                local_stakeholders=["UAE University", "Khalifa University"]
            ))
        
        return recommendations
    
    def _generate_uae_recommendations(self, project: Project) -> List[ProjectRecommendation]:
        """Generate UAE market-specific recommendations"""
        recommendations = []
        
        # Regulatory compliance
        recommendations.append(ProjectRecommendation(
            project_id=project.project_id,
            recommendation_type="mitigation",
            title="Enhance TDRA Engagement",
            description="Establish regular communication with TDRA for smooth approvals",
            priority=3,
            impact="medium",
            effort_required="low",
            timeline="immediate",
            success_probability=0.9,
            regulatory_impact="high",
            local_stakeholders=["TDRA"]
        ))
        
        # Local market considerations
        if "Dubai" in project.emirates:
            recommendations.append(ProjectRecommendation(
                project_id=project.project_id,
                recommendation_type="strategic",
                title="Align with Dubai 2071 Vision",
                description="Ensure project alignment with Dubai's long-term smart city goals",
                priority=2,
                impact="medium",
                effort_required="low",
                timeline="short-term",
                success_probability=0.8,
                local_stakeholders=["Dubai Smart City Office"]
            ))
        
        # Weather considerations
        recommendations.append(ProjectRecommendation(
            project_id=project.project_id,
            recommendation_type="mitigation",
            title="Implement Weather Protection",
            description="Install sand storm and extreme heat protection for equipment",
            priority=2,
            impact="medium",
            effort_required="medium",
            timeline="short-term",
            success_probability=0.9
        ))
        
        return recommendations
    
    def analyze_portfolio(self, projects: List[Project]) -> PortfolioInsight:
        """Analyze entire project portfolio and provide insights"""
        if not projects:
            return PortfolioInsight(0, 0.0, 0, {}, 0.0, [], [])
        
        total_projects = len(projects)
        total_budget = sum(p.budget_aed for p in projects)
        
        # Risk analysis
        risk_scores = []
        high_risk_count = 0
        
        for project in projects:
            assessment = self.risk_assessor.assess_project_risk(project)
            risk_scores.append(assessment.overall_risk_score)
            if assessment.risk_level.value >= 4:  # High or very high
                high_risk_count += 1
        
        average_risk = np.mean(risk_scores) if risk_scores else 0.0
        
        # Sector distribution
        sector_dist = {}
        for project in projects:
            sector_dist[project.sector] = sector_dist.get(project.sector, 0) + 1
        
        # Generate insights
        insights = self._generate_portfolio_insights(projects, average_risk, high_risk_count)
        strategic_recs = self._generate_portfolio_strategic_recommendations(projects, sector_dist)
        
        return PortfolioInsight(
            total_projects=total_projects,
            total_budget_aed=total_budget,
            high_risk_projects=high_risk_count,
            sector_distribution=sector_dist,
            average_risk_score=average_risk,
            key_insights=insights,
            strategic_recommendations=strategic_recs
        )
    
    def _generate_portfolio_insights(
        self, 
        projects: List[Project], 
        avg_risk: float, 
        high_risk_count: int
    ) -> List[str]:
        """Generate portfolio-level insights"""
        insights = []
        
        # Risk insights
        risk_percentage = (high_risk_count / len(projects)) * 100
        if risk_percentage > 30:
            insights.append(f"{risk_percentage:.1f}% of projects are high risk - immediate attention required")
        elif avg_risk > 3.5:
            insights.append("Portfolio has elevated risk levels - consider risk mitigation focus")
        
        # Budget insights
        total_budget = sum(p.budget_aed for p in projects)
        if total_budget > 1_000_000_000:  # 1 billion AED
            insights.append("Large portfolio investment - consider phased delivery approach")
        
        # Timeline insights
        overdue_projects = [p for p in projects if p.is_overdue()]
        if len(overdue_projects) > len(projects) * 0.2:  # More than 20% overdue
            insights.append("High percentage of overdue projects - review delivery capabilities")
        
        return insights
    
    def _generate_portfolio_strategic_recommendations(
        self, 
        projects: List[Project], 
        sector_dist: Dict[TelecomSectorType, int]
    ) -> List[str]:
        """Generate strategic recommendations for the portfolio"""
        recommendations = []
        
        # Sector balance recommendations
        total_projects = len(projects)
        if sector_dist.get(TelecomSectorType.MOBILE_NETWORKS, 0) / total_projects > 0.6:
            recommendations.append("Consider diversifying into fiber optic and IoT sectors")
        
        # Innovation recommendations
        emerging_tech_count = sum(1 for p in projects if p.technology_maturity == "emerging")
        if emerging_tech_count / total_projects < 0.2:
            recommendations.append("Increase investment in emerging technologies for future competitiveness")
        
        # UAE market recommendations
        recommendations.append("Establish centralized TDRA relationship management")
        recommendations.append("Create shared weather protection standards across projects")
        
        return recommendations
    
    def _estimate_effort(self, risk: ProjectRisk) -> str:
        """Estimate effort required to mitigate risk"""
        if risk.risk_level.value >= 4:
            return "high"
        elif risk.risk_level.value >= 3:
            return "medium"
        else:
            return "low"
    
    def _assess_regulatory_impact(self, risk: ProjectRisk) -> str:
        """Assess regulatory impact of risk"""
        if risk.regulatory_implications:
            return "high"
        elif "regulatory" in risk.title.lower():
            return "medium"
        else:
            return "low"
    
    def find_similar_projects(self, target_project: Project, projects: List[Project]) -> List[Tuple[Project, float]]:
        """Find projects similar to target project with similarity scores"""
        similarities = []
        
        for project in projects:
            if project.project_id == target_project.project_id:
                continue
                
            similarity = self._calculate_project_similarity(target_project, project)
            similarities.append((project, similarity))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:5]  # Top 5 similar projects
    
    def _calculate_project_similarity(self, project1: Project, project2: Project) -> float:
        """Calculate similarity score between two projects"""
        score = 0.0
        
        # Sector similarity
        if project1.sector == project2.sector:
            score += 0.3
        
        # Budget similarity
        budget_ratio = min(project1.budget_aed, project2.budget_aed) / max(project1.budget_aed, project2.budget_aed)
        score += 0.2 * budget_ratio
        
        # Complexity similarity
        complexity_diff = abs(project1.complexity_score - project2.complexity_score)
        complexity_similarity = max(0, 1 - complexity_diff / 10)
        score += 0.2 * complexity_similarity
        
        # Geographic similarity
        common_emirates = set(project1.emirates) & set(project2.emirates)
        if common_emirates:
            score += 0.3 * (len(common_emirates) / max(len(project1.emirates), len(project2.emirates)))
        
        return score