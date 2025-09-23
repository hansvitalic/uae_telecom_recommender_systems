"""
Machine Learning Algorithms for UAE Telecom Recommender Systems

Implements various recommendation algorithms including collaborative filtering,
content-based filtering, and hybrid approaches.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from dataclasses import dataclass
from ..models.project import Project, ProjectRisk, RiskLevel
from ..models.sector import TelecomSectorType


@dataclass
class RecommendationScore:
    """Represents a recommendation with its confidence score"""
    item_id: str
    score: float
    reason: str
    confidence: float


class CollaborativeFilteringRecommender:
    """
    Collaborative filtering recommender for project recommendations
    based on similar projects and outcomes.
    """
    
    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.svd_model = TruncatedSVD(n_components=n_components)
        self.project_features = None
        self.project_outcomes = None
        self.is_fitted = False
    
    def fit(self, projects: List[Project], outcomes: Dict[str, float]):
        """
        Fit the collaborative filtering model.
        
        Args:
            projects: List of historical projects
            outcomes: Project outcomes (success scores 0-1)
        """
        # Create feature matrix
        feature_matrix = self._create_feature_matrix(projects)
        
        # Fit SVD model
        self.project_features = self.svd_model.fit_transform(feature_matrix)
        self.project_outcomes = [outcomes.get(p.project_id, 0.5) for p in projects]
        self.projects = projects
        self.is_fitted = True
    
    def recommend_similar_projects(
        self, 
        target_project: Project, 
        n_recommendations: int = 5
    ) -> List[RecommendationScore]:
        """Recommend projects similar to target project"""
        if not self.is_fitted:
            return []
        
        # Transform target project
        target_features = self._project_to_features(target_project)
        target_transformed = self.svd_model.transform(target_features.reshape(1, -1))
        
        # Calculate similarities
        similarities = cosine_similarity(target_transformed, self.project_features)[0]
        
        # Get top recommendations
        top_indices = np.argsort(similarities)[-n_recommendations-1:-1][::-1]
        
        recommendations = []
        for idx in top_indices:
            project = self.projects[idx]
            score = similarities[idx]
            outcome = self.project_outcomes[idx]
            
            recommendations.append(RecommendationScore(
                item_id=project.project_id,
                score=score * outcome,  # Weight by historical outcome
                reason=f"Similar to successful projects in {project.sector.value}",
                confidence=score * 0.8  # Collaborative filtering confidence
            ))
        
        return recommendations
    
    def _create_feature_matrix(self, projects: List[Project]) -> np.ndarray:
        """Create feature matrix from projects"""
        features = []
        for project in projects:
            feature_vector = self._project_to_features(project)
            features.append(feature_vector)
        return np.array(features)
    
    def _project_to_features(self, project: Project) -> np.ndarray:
        """Convert project to feature vector"""
        features = []
        
        # Sector features (one-hot encoding)
        sector_features = [0] * len(TelecomSectorType)
        sector_features[list(TelecomSectorType).index(project.sector)] = 1
        features.extend(sector_features)
        
        # Numerical features (normalized)
        features.extend([
            project.budget_aed / 1_000_000_000,  # Normalize to billions
            project.complexity_score / 10,
            project.strategic_importance / 10,
            len(project.risks) / 20,  # Normalize to max 20 risks
            project.calculate_overall_risk_score() / 5
        ])
        
        # Emirates features (binary for major emirates)
        major_emirates = ["Dubai", "Abu Dhabi", "Sharjah", "Ajman"]
        for emirate in major_emirates:
            features.append(1 if emirate in project.emirates else 0)
        
        return np.array(features)


class ContentBasedRecommender:
    """
    Content-based recommender using project attributes and sector expertise.
    """
    
    def __init__(self):
        self.sector_profiles = self._initialize_sector_profiles()
        self.risk_patterns = {}
    
    def _initialize_sector_profiles(self) -> Dict[TelecomSectorType, Dict[str, float]]:
        """Initialize sector preference profiles"""
        return {
            TelecomSectorType.MOBILE_NETWORKS: {
                "high_budget_preference": 0.8,
                "complexity_tolerance": 0.9,
                "risk_tolerance": 0.6,
                "innovation_preference": 0.7,
                "regulatory_sensitivity": 0.8
            },
            TelecomSectorType.FIBER_OPTIC: {
                "high_budget_preference": 0.9,
                "complexity_tolerance": 0.7,
                "risk_tolerance": 0.5,
                "innovation_preference": 0.5,
                "regulatory_sensitivity": 0.9
            },
            TelecomSectorType.IOT_M2M: {
                "high_budget_preference": 0.4,
                "complexity_tolerance": 0.8,
                "risk_tolerance": 0.7,
                "innovation_preference": 0.9,
                "regulatory_sensitivity": 0.6
            }
        }
    
    def recommend_mitigation_strategies(
        self, 
        project: Project, 
        risk: ProjectRisk
    ) -> List[RecommendationScore]:
        """Recommend mitigation strategies based on content similarity"""
        strategies = []
        
        # Get sector profile
        sector_profile = self.sector_profiles.get(project.sector, {})
        
        # Base strategies for each risk category
        base_strategies = self._get_base_strategies(risk.category)
        
        for strategy_id, strategy_info in base_strategies.items():
            # Calculate content-based score
            score = self._calculate_strategy_score(
                project, risk, strategy_info, sector_profile
            )
            
            strategies.append(RecommendationScore(
                item_id=strategy_id,
                score=score,
                reason=f"Effective for {risk.category.value} risks in {project.sector.value}",
                confidence=0.7  # Content-based confidence
            ))
        
        # Sort by score and return top strategies
        strategies.sort(key=lambda x: x.score, reverse=True)
        return strategies[:5]
    
    def _get_base_strategies(self, risk_category) -> Dict[str, Dict[str, Any]]:
        """Get base mitigation strategies for risk category"""
        strategies = {
            "technical": {
                "TECH_001": {
                    "name": "Technology Assessment",
                    "effectiveness": 0.8,
                    "complexity": 0.6,
                    "cost": 0.4
                },
                "TECH_002": {
                    "name": "Vendor Diversification", 
                    "effectiveness": 0.7,
                    "complexity": 0.5,
                    "cost": 0.6
                }
            },
            "regulatory": {
                "REG_001": {
                    "name": "Early TDRA Engagement",
                    "effectiveness": 0.9,
                    "complexity": 0.3,
                    "cost": 0.2
                },
                "REG_002": {
                    "name": "Compliance Framework",
                    "effectiveness": 0.8,
                    "complexity": 0.7,
                    "cost": 0.5
                }
            },
            "operational": {
                "OPS_001": {
                    "name": "Process Optimization",
                    "effectiveness": 0.7,
                    "complexity": 0.6,
                    "cost": 0.4
                },
                "OPS_002": {
                    "name": "Resource Augmentation",
                    "effectiveness": 0.8,
                    "complexity": 0.4,
                    "cost": 0.8
                }
            }
        }
        
        return strategies.get(risk_category.value, {})
    
    def _calculate_strategy_score(
        self, 
        project: Project, 
        risk: ProjectRisk, 
        strategy_info: Dict[str, Any],
        sector_profile: Dict[str, float]
    ) -> float:
        """Calculate content-based strategy score"""
        base_score = strategy_info.get("effectiveness", 0.5)
        
        # Adjust based on sector preferences
        complexity_factor = 1 - (strategy_info.get("complexity", 0.5) * 
                                 (1 - sector_profile.get("complexity_tolerance", 0.5)))
        
        cost_factor = 1 - (strategy_info.get("cost", 0.5) * 
                          (1 - sector_profile.get("high_budget_preference", 0.5)))
        
        # Risk level adjustment
        risk_urgency = risk.risk_level.value / 5.0
        urgency_bonus = risk_urgency * 0.2
        
        final_score = base_score * complexity_factor * cost_factor + urgency_bonus
        return min(1.0, final_score)


class HybridRecommender:
    """
    Hybrid recommender combining collaborative filtering and content-based approaches.
    """
    
    def __init__(self, cf_weight: float = 0.6, cb_weight: float = 0.4):
        self.cf_recommender = CollaborativeFilteringRecommender()
        self.cb_recommender = ContentBasedRecommender()
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.clustering_model = KMeans(n_clusters=5, random_state=42)
        self.project_clusters = None
    
    def fit(self, projects: List[Project], outcomes: Dict[str, float]):
        """Fit the hybrid model"""
        # Fit collaborative filtering
        self.cf_recommender.fit(projects, outcomes)
        
        # Fit clustering for project segmentation
        feature_matrix = self.cf_recommender._create_feature_matrix(projects)
        self.project_clusters = self.clustering_model.fit_predict(feature_matrix)
        self.projects = projects
    
    def get_hybrid_recommendations(
        self, 
        target_project: Project, 
        n_recommendations: int = 10
    ) -> List[RecommendationScore]:
        """Get hybrid recommendations combining multiple approaches"""
        recommendations = {}
        
        # Get collaborative filtering recommendations
        cf_recs = self.cf_recommender.recommend_similar_projects(
            target_project, n_recommendations * 2
        )
        
        for rec in cf_recs:
            rec_id = rec.item_id
            if rec_id not in recommendations:
                recommendations[rec_id] = rec
                recommendations[rec_id].score *= self.cf_weight
                recommendations[rec_id].confidence *= self.cf_weight
            else:
                recommendations[rec_id].score += rec.score * self.cf_weight
                recommendations[rec_id].confidence += rec.confidence * self.cf_weight
        
        # Add cluster-based recommendations
        cluster_recs = self._get_cluster_recommendations(target_project, n_recommendations)
        
        for rec in cluster_recs:
            rec_id = rec.item_id
            if rec_id not in recommendations:
                recommendations[rec_id] = rec
                recommendations[rec_id].score *= self.cb_weight
                recommendations[rec_id].confidence *= self.cb_weight
            else:
                recommendations[rec_id].score += rec.score * self.cb_weight
                recommendations[rec_id].confidence += rec.confidence * self.cb_weight
        
        # Sort and return top recommendations
        final_recs = list(recommendations.values())
        final_recs.sort(key=lambda x: x.score, reverse=True)
        return final_recs[:n_recommendations]
    
    def _get_cluster_recommendations(
        self, 
        target_project: Project, 
        n_recommendations: int
    ) -> List[RecommendationScore]:
        """Get recommendations based on project clustering"""
        if self.project_clusters is None:
            return []
        
        # Find target project's cluster
        target_features = self.cf_recommender._project_to_features(target_project)
        target_cluster = self.clustering_model.predict(target_features.reshape(1, -1))[0]
        
        # Find projects in same cluster
        cluster_projects = [
            self.projects[i] for i, cluster in enumerate(self.project_clusters)
            if cluster == target_cluster
        ]
        
        recommendations = []
        for project in cluster_projects[:n_recommendations]:
            recommendations.append(RecommendationScore(
                item_id=project.project_id,
                score=0.7,  # Base cluster similarity score
                reason=f"Similar project characteristics in cluster {target_cluster}",
                confidence=0.6
            ))
        
        return recommendations


class RiskPredictionAlgorithm:
    """
    Advanced risk prediction using machine learning techniques.
    """
    
    def __init__(self):
        self.risk_patterns = {}
        self.sector_risk_models = {}
    
    def analyze_risk_patterns(self, projects: List[Project]) -> Dict[str, Any]:
        """Analyze risk patterns across projects"""
        patterns = {
            "sector_risk_distribution": {},
            "budget_risk_correlation": {},
            "complexity_risk_correlation": {},
            "timeline_risk_patterns": {}
        }
        
        for project in projects:
            sector = project.sector
            if sector not in patterns["sector_risk_distribution"]:
                patterns["sector_risk_distribution"][sector] = []
            
            risk_score = project.calculate_overall_risk_score()
            patterns["sector_risk_distribution"][sector].append(risk_score)
            
            # Budget correlation
            budget_tier = self._get_budget_tier(project.budget_aed)
            if budget_tier not in patterns["budget_risk_correlation"]:
                patterns["budget_risk_correlation"][budget_tier] = []
            patterns["budget_risk_correlation"][budget_tier].append(risk_score)
        
        return patterns
    
    def predict_project_risks(self, project: Project) -> List[str]:
        """Predict potential risks for a new project"""
        predicted_risks = []
        
        # Sector-based risk prediction
        sector_risks = self._get_sector_typical_risks(project.sector)
        predicted_risks.extend(sector_risks)
        
        # Budget-based risk prediction
        if project.budget_aed > 100_000_000:
            predicted_risks.extend([
                "Budget overrun risk",
                "Stakeholder management complexity",
                "Procurement delays"
            ])
        
        # Complexity-based risk prediction
        if project.complexity_score > 7:
            predicted_risks.extend([
                "Technical integration challenges",
                "Resource shortage risk",
                "Timeline extension risk"
            ])
        
        return list(set(predicted_risks))  # Remove duplicates
    
    def _get_budget_tier(self, budget: float) -> str:
        """Categorize project by budget tier"""
        if budget < 10_000_000:
            return "small"
        elif budget < 50_000_000:
            return "medium"
        elif budget < 200_000_000:
            return "large"
        else:
            return "mega"
    
    def _get_sector_typical_risks(self, sector: TelecomSectorType) -> List[str]:
        """Get typical risks for a sector"""
        sector_risks = {
            TelecomSectorType.MOBILE_NETWORKS: [
                "Spectrum interference",
                "Network security vulnerabilities",
                "Technology obsolescence"
            ],
            TelecomSectorType.FIBER_OPTIC: [
                "Right-of-way delays",
                "Construction weather delays",
                "Equipment supply chain issues"
            ],
            TelecomSectorType.IOT_M2M: [
                "Device interoperability",
                "Security vulnerabilities",
                "Platform scalability challenges"
            ]
        }
        
        return sector_risks.get(sector, [])