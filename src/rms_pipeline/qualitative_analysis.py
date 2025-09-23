"""
Qualitative Analysis Module
Performs qualitative risk analysis including likelihood, impact, and priority assessment
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import config

class QualitativeAnalyzer:
    """
    Handles qualitative risk analysis and assessment
    """
    
    def __init__(self, risks_df: pd.DataFrame = None):
        """Initialize with risk data"""
        self.risks_df = risks_df
        if self.risks_df is None:
            self.risks_df = pd.read_csv(config.RISK_REGISTER_CSV)
    
    def assess_probability_impact_matrix(self) -> Dict[str, any]:
        """Create probability-impact matrix analysis"""
        # Create matrix categories
        matrix_data = []
        
        for _, risk in self.risks_df.iterrows():
            prob = risk["Probability Rating"]
            impact = risk["Impact Rating"]
            
            # Determine risk level based on probability and impact
            if prob == "High" and impact == "High":
                risk_level = "Critical"
            elif (prob == "High" and impact == "Medium") or (prob == "Medium" and impact == "High"):
                risk_level = "High"
            elif prob == "Medium" and impact == "Medium":
                risk_level = "Medium"
            else:
                risk_level = "Low"
                
            matrix_data.append({
                "Risk ID": risk["Risk ID"],
                "Risk Title": risk["Risk Title"],
                "Probability": prob,
                "Impact": impact,
                "Risk Level": risk_level,
                "Risk Score": risk["Risk Score"],
                "Category": risk["Risk Category"]
            })
        
        matrix_df = pd.DataFrame(matrix_data)
        
        # Summary statistics
        summary = {
            "risk_distribution": matrix_df["Risk Level"].value_counts().to_dict(),
            "avg_risk_score_by_level": matrix_df.groupby("Risk Level")["Risk Score"].mean().to_dict(),
            "matrix_data": matrix_df
        }
        
        return summary
    
    def analyze_qualitative_factors(self) -> Dict[str, any]:
        """Analyze qualitative risk factors and patterns"""
        analysis = {
            "stakeholder_impact": {},
            "project_phase_risks": {},
            "departmental_risks": {},
            "risk_interconnections": {}
        }
        
        # Stakeholder impact analysis
        stakeholder_groups = self.risks_df["Stakeholder Group"].value_counts()
        analysis["stakeholder_impact"] = {
            "most_affected_stakeholders": stakeholder_groups.head().to_dict(),
            "total_stakeholder_groups": len(stakeholder_groups)
        }
        
        # Project phase risk distribution
        phase_risks = self.risks_df.groupby("Project Phase").agg({
            "Risk Score": ["count", "mean", "max"],
            "Probability Rating": lambda x: (x == "High").sum(),
            "Impact Rating": lambda x: (x == "High").sum()
        }).round(2)
        
        analysis["project_phase_risks"] = phase_risks.to_dict()
        
        # Departmental risk analysis
        dept_analysis = self.risks_df.groupby("Department / Unit").agg({
            "Risk ID": "count",
            "Risk Score": "mean"
        }).round(2)
        
        analysis["departmental_risks"] = dept_analysis.to_dict()
        
        # Risk interconnections (based on similar categories/phases)
        interconnections = []
        for category in self.risks_df["Risk Category"].unique():
            category_risks = self.risks_df[self.risks_df["Risk Category"] == category]
            if len(category_risks) > 1:
                interconnections.append({
                    "category": category,
                    "risk_count": len(category_risks),
                    "avg_score": category_risks["Risk Score"].mean(),
                    "high_priority_count": len(category_risks[
                        (category_risks["Probability Rating"] == "High") | 
                        (category_risks["Impact Rating"] == "High")
                    ])
                })
        
        analysis["risk_interconnections"] = interconnections
        
        return analysis
    
    def prioritize_risks(self) -> List[Dict]:
        """Prioritize risks based on qualitative assessment"""
        # Create enhanced risk scores
        priority_risks = []
        
        for _, risk in self.risks_df.iterrows():
            # Base score from existing data
            base_score = risk["Risk Score"] if pd.notna(risk["Risk Score"]) else 0
            
            # Enhancement factors
            category_factor = 1.2 if risk["Risk Category"] in config.HIGH_PRIORITY_CATEGORIES else 1.0
            sector_factor = 1.1 if risk["Sector"] in config.CRITICAL_SECTORS else 1.0
            status_factor = 1.3 if risk["Status"] == "Open" else 0.8
            
            # Calculate enhanced priority score
            enhanced_score = base_score * category_factor * sector_factor * status_factor
            
            priority_risks.append({
                "Risk ID": risk["Risk ID"],
                "Risk Title": risk["Risk Title"],
                "Base Score": base_score,
                "Enhanced Score": round(enhanced_score, 2),
                "Priority Level": self._determine_priority_level(enhanced_score),
                "Category": risk["Risk Category"],
                "Status": risk["Status"],
                "Mitigation Strategy": risk["Mitigation Strategy"]
            })
        
        # Sort by enhanced score
        priority_risks.sort(key=lambda x: x["Enhanced Score"], reverse=True)
        
        return priority_risks
    
    def _determine_priority_level(self, score: float) -> str:
        """Determine priority level based on enhanced score"""
        if score >= 15:
            return "Critical"
        elif score >= 10:
            return "High"
        elif score >= 5:
            return "Medium"
        else:
            return "Low"
    
    def generate_risk_heat_map_data(self) -> Dict[str, any]:
        """Generate data for risk heat map visualization"""
        heat_map_data = {
            "probability_impact_grid": {},
            "category_phase_grid": {},
            "risk_counts": {}
        }
        
        # Probability-Impact heat map
        prob_impact_counts = self.risks_df.groupby(["Probability Rating", "Impact Rating"]).size().reset_index(name="Count")
        heat_map_data["probability_impact_grid"] = prob_impact_counts.to_dict("records")
        
        # Category-Phase heat map
        category_phase_counts = self.risks_df.groupby(["Risk Category", "Project Phase"]).size().reset_index(name="Count")
        heat_map_data["category_phase_grid"] = category_phase_counts.to_dict("records")
        
        # Overall risk counts
        heat_map_data["risk_counts"] = {
            "total_risks": len(self.risks_df),
            "high_prob_risks": len(self.risks_df[self.risks_df["Probability Rating"] == "High"]),
            "high_impact_risks": len(self.risks_df[self.risks_df["Impact Rating"] == "High"]),
            "critical_risks": len(self.risks_df[
                (self.risks_df["Probability Rating"] == "High") & 
                (self.risks_df["Impact Rating"] == "High")
            ])
        }
        
        return heat_map_data
    
    def generate_qualitative_report(self) -> Dict[str, any]:
        """Generate comprehensive qualitative analysis report"""
        prob_impact_matrix = self.assess_probability_impact_matrix()
        qualitative_factors = self.analyze_qualitative_factors()
        prioritized_risks = self.prioritize_risks()
        heat_map_data = self.generate_risk_heat_map_data()
        
        report = {
            "analysis_summary": {
                "total_risks_analyzed": len(self.risks_df),
                "critical_risks": len([r for r in prioritized_risks if r["Priority Level"] == "Critical"]),
                "high_priority_risks": len([r for r in prioritized_risks if r["Priority Level"] == "High"]),
                "completion_status": "Qualitative Analysis Complete"
            },
            "probability_impact_matrix": prob_impact_matrix,
            "qualitative_factors": qualitative_factors,
            "prioritized_risks": prioritized_risks[:20],  # Top 20 risks
            "heat_map_data": heat_map_data,
            "recommendations": self._generate_qualitative_recommendations(prioritized_risks, qualitative_factors)
        }
        
        return report
    
    def _generate_qualitative_recommendations(self, prioritized_risks: List, factors: Dict) -> List[str]:
        """Generate recommendations based on qualitative analysis"""
        recommendations = []
        
        # Priority-based recommendations
        critical_count = len([r for r in prioritized_risks if r["Priority Level"] == "Critical"])
        if critical_count > 0:
            recommendations.append(f"Immediate action required for {critical_count} critical risks")
        
        # Category-based recommendations
        if "Construction" in [r["Category"] for r in prioritized_risks[:10]]:
            recommendations.append("Focus on construction risk mitigation strategies")
            
        # Phase-based recommendations
        phase_data = factors.get("project_phase_risks", {})
        if phase_data:
            recommendations.append("Review risk distribution across project phases for optimization")
        
        return recommendations

def main():
    """Main execution function for qualitative analysis"""
    analyzer = QualitativeAnalyzer()
    
    print("📊 Starting Qualitative Risk Analysis...")
    print("=" * 50)
    
    # Generate qualitative analysis report
    report = analyzer.generate_qualitative_report()
    
    print(f"✓ Analyzed {report['analysis_summary']['total_risks_analyzed']} risks")
    print(f"✓ Identified {report['analysis_summary']['critical_risks']} critical risks")
    print(f"✓ Identified {report['analysis_summary']['high_priority_risks']} high priority risks")
    
    return report

if __name__ == "__main__":
    main()