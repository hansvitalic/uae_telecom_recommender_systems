"""
Documentation Module
Generates comprehensive documentation and reports for the RMS pipeline
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import json
import config

class RMSDocumentationGenerator:
    """
    Handles comprehensive documentation generation for RMS pipeline
    """
    
    def __init__(self, risks_df: pd.DataFrame = None, all_analysis_results: Dict = None):
        """Initialize with risk data and analysis results"""
        self.risks_df = risks_df
        if self.risks_df is None:
            self.risks_df = pd.read_csv(config.RISK_REGISTER_CSV)
        
        self.analysis_results = all_analysis_results or {}
        self.generation_timestamp = datetime.now()
    
    def generate_executive_summary(self) -> Dict[str, any]:
        """Generate executive summary of risk management status"""
        summary = {
            "overview": {
                "report_date": self.generation_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "total_risks": len(self.risks_df),
                "project_scope": "UAE Telecom Network Infrastructure",
                "analysis_completeness": self._assess_analysis_completeness()
            },
            "key_findings": self._extract_key_findings(),
            "critical_risks": self._identify_critical_risks_summary(),
            "risk_distribution": self._summarize_risk_distribution(),
            "strategic_recommendations": self._generate_strategic_recommendations(),
            "next_steps": self._define_next_steps()
        }
        
        return summary
    
    def _assess_analysis_completeness(self) -> Dict[str, str]:
        """Assess completeness of RMS analysis"""
        completeness = {}
        
        rms_steps = [
            "Risk Identification",
            "Qualitative Analysis", 
            "Quantitative Analysis",
            "Response Planning",
            "Monitoring",
            "Documentation"
        ]
        
        for step in rms_steps:
            # Check if analysis results contain this step
            if step.lower().replace(" ", "_") in self.analysis_results:
                completeness[step] = "Complete"
            else:
                completeness[step] = "In Progress"
        
        # Override documentation as in progress since we're generating it
        completeness["Documentation"] = "In Progress"
        
        return completeness
    
    def _extract_key_findings(self) -> List[str]:
        """Extract key findings from all analyses"""
        findings = []
        
        # Risk identification findings
        findings.append(f"Identified {len(self.risks_df)} risks across {len(self.risks_df['Risk Category'].unique())} categories")
        
        # Critical risk finding
        critical_risks = self.risks_df[
            (self.risks_df["Probability Rating"] == "High") & 
            (self.risks_df["Impact Rating"] == "High")
        ]
        findings.append(f"{len(critical_risks)} critical risks require immediate attention")
        
        # Category concentration finding
        top_category = self.risks_df["Risk Category"].value_counts().index[0]
        top_category_count = self.risks_df["Risk Category"].value_counts().iloc[0]
        findings.append(f"Construction category represents {top_category_count} risks, indicating high concentration")
        
        # Status finding
        open_risks = len(self.risks_df[self.risks_df["Status"] == "Open"])
        findings.append(f"{open_risks} risks remain open and require ongoing management")
        
        return findings
    
    def _identify_critical_risks_summary(self) -> Dict[str, any]:
        """Summarize critical risks for executive attention"""
        critical_risks = self.risks_df[
            (self.risks_df["Probability Rating"] == "High") & 
            (self.risks_df["Impact Rating"] == "High")
        ]
        
        if len(critical_risks) == 0:
            return {"count": 0, "message": "No critical risks identified"}
        
        summary = {
            "count": len(critical_risks),
            "categories": critical_risks["Risk Category"].value_counts().to_dict(),
            "phases": critical_risks["Project Phase"].value_counts().to_dict(),
            "top_critical_risks": critical_risks[["Risk ID", "Risk Title", "Risk Category"]].head().to_dict("records"),
            "mitigation_status": {
                "with_mitigation": len(critical_risks[critical_risks["Mitigation Strategy"].notna()]),
                "without_mitigation": len(critical_risks[critical_risks["Mitigation Strategy"].isna()])
            }
        }
        
        return summary
    
    def _summarize_risk_distribution(self) -> Dict[str, any]:
        """Summarize risk distribution across key dimensions"""
        distribution = {
            "by_category": self.risks_df["Risk Category"].value_counts().to_dict(),
            "by_phase": self.risks_df["Project Phase"].value_counts().to_dict(),
            "by_probability": self.risks_df["Probability Rating"].value_counts().to_dict(),
            "by_impact": self.risks_df["Impact Rating"].value_counts().to_dict(),
            "by_status": self.risks_df["Status"].value_counts().to_dict(),
            "by_rms_step": self.risks_df["Primary RMS Step"].value_counts().to_dict()
        }
        
        return distribution
    
    def _generate_strategic_recommendations(self) -> List[str]:
        """Generate strategic recommendations for leadership"""
        recommendations = []
        
        # Critical risk recommendation
        critical_count = len(self.risks_df[
            (self.risks_df["Probability Rating"] == "High") & 
            (self.risks_df["Impact Rating"] == "High")
        ])
        
        if critical_count > 0:
            recommendations.append(f"Prioritize immediate action on {critical_count} critical risks to prevent project delays")
        
        # Construction risk recommendation
        construction_risks = len(self.risks_df[self.risks_df["Risk Category"] == "Construction"])
        if construction_risks > 50:  # Threshold for high concentration
            recommendations.append("Establish dedicated construction risk management team due to high risk concentration")
        
        # Mitigation coverage recommendation
        mitigation_coverage = len(self.risks_df[self.risks_df["Mitigation Strategy"].notna()]) / len(self.risks_df)
        if mitigation_coverage < 0.8:
            recommendations.append("Improve mitigation strategy coverage - currently below 80% threshold")
        
        # Resource allocation recommendation
        open_risks = len(self.risks_df[self.risks_df["Status"] == "Open"])
        if open_risks > len(self.risks_df) * 0.7:
            recommendations.append("Allocate additional resources to accelerate risk closure activities")
        
        return recommendations
    
    def _define_next_steps(self) -> List[str]:
        """Define immediate next steps"""
        next_steps = [
            "Execute immediate response strategies for critical risks",
            "Implement continuous monitoring framework",
            "Schedule weekly risk review meetings",
            "Establish automated risk reporting system",
            "Conduct quarterly comprehensive risk assessment"
        ]
        
        return next_steps
    
    def generate_detailed_risk_register(self) -> Dict[str, any]:
        """Generate detailed risk register documentation"""
        register = {
            "register_metadata": {
                "generation_date": self.generation_timestamp.strftime("%Y-%m-%d"),
                "total_risks": len(self.risks_df),
                "data_source": "network_infrastructure_risk_register.csv",
                "last_updated": "Current"
            },
            "risk_categories": self._document_risk_categories(),
            "risk_analysis": self._document_risk_analysis(),
            "mitigation_strategies": self._document_mitigation_strategies(),
            "risk_matrix": self._create_risk_matrix_documentation()
        }
        
        return register
    
    def _document_risk_categories(self) -> Dict[str, any]:
        """Document risk categories and their characteristics"""
        categories = {}
        
        for category in self.risks_df["Risk Category"].unique():
            category_data = self.risks_df[self.risks_df["Risk Category"] == category]
            
            categories[category] = {
                "total_risks": len(category_data),
                "subcategories": category_data["Sub-Category"].value_counts().to_dict(),
                "avg_risk_score": round(category_data["Risk Score"].mean(), 2),
                "probability_distribution": category_data["Probability Rating"].value_counts().to_dict(),
                "impact_distribution": category_data["Impact Rating"].value_counts().to_dict(),
                "phase_distribution": category_data["Project Phase"].value_counts().to_dict(),
                "sample_risks": category_data[["Risk ID", "Risk Title"]].head(3).to_dict("records")
            }
        
        return categories
    
    def _document_risk_analysis(self) -> Dict[str, any]:
        """Document comprehensive risk analysis"""
        analysis = {
            "quantitative_metrics": {
                "risk_score_statistics": {
                    "mean": round(self.risks_df["Risk Score"].mean(), 2),
                    "median": round(self.risks_df["Risk Score"].median(), 2),
                    "std": round(self.risks_df["Risk Score"].std(), 2),
                    "min": self.risks_df["Risk Score"].min(),
                    "max": self.risks_df["Risk Score"].max()
                },
                "risk_distribution": self.risks_df["Risk Score"].value_counts().sort_index().to_dict()
            },
            "qualitative_assessment": {
                "high_probability_risks": len(self.risks_df[self.risks_df["Probability Rating"] == "High"]),
                "high_impact_risks": len(self.risks_df[self.risks_df["Impact Rating"] == "High"]),
                "critical_risks": len(self.risks_df[
                    (self.risks_df["Probability Rating"] == "High") & 
                    (self.risks_df["Impact Rating"] == "High")
                ])
            },
            "trend_analysis": self._analyze_trends_for_documentation()
        }
        
        return analysis
    
    def _analyze_trends_for_documentation(self) -> Dict[str, any]:
        """Analyze trends for documentation purposes"""
        trends = {
            "identification_trends": {},
            "category_trends": {},
            "phase_trends": {}
        }
        
        # RMS step distribution trends
        for step in config.RMS_STEPS:
            step_risks = self.risks_df[self.risks_df["Primary RMS Step"].str.contains(step, na=False)]
            trends["identification_trends"][step] = len(step_risks)
        
        # Category trends
        trends["category_trends"] = self.risks_df["Risk Category"].value_counts().to_dict()
        
        # Phase trends
        trends["phase_trends"] = self.risks_df["Project Phase"].value_counts().to_dict()
        
        return trends
    
    def _document_mitigation_strategies(self) -> Dict[str, any]:
        """Document mitigation strategies and their effectiveness"""
        mitigation_doc = {
            "strategy_coverage": {
                "total_risks": len(self.risks_df),
                "with_mitigation": len(self.risks_df[self.risks_df["Mitigation Strategy"].notna()]),
                "with_contingency": len(self.risks_df[self.risks_df["Contingency Plan"].notna()]),
                "coverage_percentage": round(
                    len(self.risks_df[self.risks_df["Mitigation Strategy"].notna()]) / len(self.risks_df) * 100, 1
                )
            },
            "strategy_types": self._categorize_mitigation_strategies(),
            "effectiveness_indicators": self._assess_mitigation_effectiveness()
        }
        
        return mitigation_doc
    
    def _categorize_mitigation_strategies(self) -> Dict[str, List]:
        """Categorize mitigation strategies"""
        strategy_categories = {
            "Preventive": [],
            "Detective": [],
            "Corrective": [],
            "Contingency": []
        }
        
        # Simple keyword-based categorization
        for _, risk in self.risks_df.iterrows():
            strategy = str(risk.get("Mitigation Strategy", "")).lower()
            risk_id = risk["Risk ID"]
            
            if any(word in strategy for word in ["prevent", "avoid", "eliminate", "reduce"]):
                strategy_categories["Preventive"].append(risk_id)
            elif any(word in strategy for word in ["monitor", "track", "inspect", "audit"]):
                strategy_categories["Detective"].append(risk_id)
            elif any(word in strategy for word in ["correct", "fix", "repair", "resolve"]):
                strategy_categories["Corrective"].append(risk_id)
            elif strategy:  # Has strategy but doesn't fit other categories
                strategy_categories["Contingency"].append(risk_id)
        
        return strategy_categories
    
    def _assess_mitigation_effectiveness(self) -> Dict[str, any]:
        """Assess effectiveness of mitigation strategies"""
        with_mitigation = self.risks_df[self.risks_df["Mitigation Strategy"].notna()]
        without_mitigation = self.risks_df[self.risks_df["Mitigation Strategy"].isna()]
        
        effectiveness = {
            "avg_score_with_mitigation": round(with_mitigation["Risk Score"].mean(), 2) if len(with_mitigation) > 0 else 0,
            "avg_score_without_mitigation": round(without_mitigation["Risk Score"].mean(), 2) if len(without_mitigation) > 0 else 0,
            "mitigation_impact": "Positive" if len(with_mitigation) > 0 and len(without_mitigation) > 0 and 
                                 with_mitigation["Risk Score"].mean() < without_mitigation["Risk Score"].mean() else "Neutral"
        }
        
        return effectiveness
    
    def _create_risk_matrix_documentation(self) -> Dict[str, any]:
        """Create risk matrix documentation"""
        matrix = {
            "probability_impact_matrix": {},
            "risk_levels": {"Critical": [], "High": [], "Medium": [], "Low": []},
            "matrix_analysis": {}
        }
        
        # Create probability-impact combinations
        for prob in ["Low", "Medium", "High"]:
            for impact in ["Low", "Medium", "High"]:
                risks_in_cell = self.risks_df[
                    (self.risks_df["Probability Rating"] == prob) & 
                    (self.risks_df["Impact Rating"] == impact)
                ]
                
                matrix["probability_impact_matrix"][f"{prob}-{impact}"] = {
                    "count": len(risks_in_cell),
                    "risk_ids": risks_in_cell["Risk ID"].tolist()
                }
        
        # Categorize risks by level
        for _, risk in self.risks_df.iterrows():
            prob = risk["Probability Rating"]
            impact = risk["Impact Rating"]
            
            if prob == "High" and impact == "High":
                matrix["risk_levels"]["Critical"].append(risk["Risk ID"])
            elif (prob == "High" and impact == "Medium") or (prob == "Medium" and impact == "High"):
                matrix["risk_levels"]["High"].append(risk["Risk ID"])
            elif prob == "Medium" and impact == "Medium":
                matrix["risk_levels"]["Medium"].append(risk["Risk ID"])
            else:
                matrix["risk_levels"]["Low"].append(risk["Risk ID"])
        
        # Matrix analysis
        matrix["matrix_analysis"] = {
            "concentration": self._analyze_matrix_concentration(matrix["probability_impact_matrix"]),
            "risk_level_distribution": {level: len(risks) for level, risks in matrix["risk_levels"].items()}
        }
        
        return matrix
    
    def _analyze_matrix_concentration(self, matrix_data: Dict) -> Dict[str, any]:
        """Analyze concentration of risks in matrix"""
        concentrations = {}
        total_risks = len(self.risks_df)
        
        for cell, data in matrix_data.items():
            percentage = (data["count"] / total_risks) * 100 if total_risks > 0 else 0
            concentrations[cell] = {
                "count": data["count"],
                "percentage": round(percentage, 1)
            }
        
        # Find highest concentration
        max_cell = max(concentrations.keys(), key=lambda x: concentrations[x]["count"])
        
        return {
            "highest_concentration": max_cell,
            "concentration_details": concentrations,
            "distribution_balance": "Balanced" if concentrations[max_cell]["percentage"] < 30 else "Concentrated"
        }
    
    def generate_lessons_learned(self) -> Dict[str, any]:
        """Generate lessons learned documentation"""
        lessons = {
            "data_quality_lessons": self._extract_data_quality_lessons(),
            "risk_management_lessons": self._extract_risk_management_lessons(),
            "process_improvement_lessons": self._extract_process_lessons(),
            "recommendations_for_future": self._generate_future_recommendations()
        }
        
        return lessons
    
    def _extract_data_quality_lessons(self) -> List[str]:
        """Extract lessons learned about data quality"""
        lessons = []
        
        # Check data completeness
        missing_data_fields = []
        for col in ["Risk Title", "Risk Category", "Probability Rating", "Impact Rating"]:
            missing_count = self.risks_df[col].isnull().sum()
            if missing_count > 0:
                missing_data_fields.append(f"{col} ({missing_count} missing)")
        
        if missing_data_fields:
            lessons.append(f"Improve data collection for: {', '.join(missing_data_fields)}")
        
        # Check data consistency
        inconsistent_scores = 0
        for _, risk in self.risks_df.iterrows():
            if pd.notna(risk["Risk Score"]) and pd.notna(risk["Probability Rating"]) and pd.notna(risk["Impact Rating"]):
                prob_weight = config.PROBABILITY_WEIGHTS.get(risk["Probability Rating"], 0)
                impact_weight = config.IMPACT_WEIGHTS.get(risk["Impact Rating"], 0)
                expected_score = prob_weight * impact_weight
                if abs(risk["Risk Score"] - expected_score) > 2:
                    inconsistent_scores += 1
        
        if inconsistent_scores > 0:
            lessons.append(f"Standardize risk scoring methodology - {inconsistent_scores} inconsistent scores found")
        
        return lessons
    
    def _extract_risk_management_lessons(self) -> List[str]:
        """Extract lessons learned about risk management"""
        lessons = []
        
        # Mitigation coverage
        mitigation_coverage = len(self.risks_df[self.risks_df["Mitigation Strategy"].notna()]) / len(self.risks_df)
        if mitigation_coverage < 0.8:
            lessons.append("Ensure all risks have documented mitigation strategies")
        
        # Risk concentration
        top_category = self.risks_df["Risk Category"].value_counts().index[0]
        top_category_pct = (self.risks_df["Risk Category"].value_counts().iloc[0] / len(self.risks_df)) * 100
        if top_category_pct > 40:
            lessons.append(f"Address risk concentration in {top_category} category ({top_category_pct:.1f}% of total risks)")
        
        # Critical risk management
        critical_risks = len(self.risks_df[
            (self.risks_df["Probability Rating"] == "High") & 
            (self.risks_df["Impact Rating"] == "High")
        ])
        if critical_risks > 0:
            lessons.append("Establish immediate response protocols for critical risks")
        
        return lessons
    
    def _extract_process_lessons(self) -> List[str]:
        """Extract lessons learned about RMS process"""
        lessons = [
            "Implement automated risk monitoring systems to reduce manual effort",
            "Establish regular risk review cycles for all project phases",
            "Integrate risk management with project management tools",
            "Develop standardized risk assessment templates",
            "Create risk management training programs for project teams"
        ]
        
        return lessons
    
    def _generate_future_recommendations(self) -> List[str]:
        """Generate recommendations for future projects"""
        recommendations = [
            "Implement predictive analytics for early risk detection",
            "Establish risk management centers of excellence",
            "Develop industry-specific risk libraries and templates",
            "Create automated risk reporting and dashboard systems",
            "Implement integrated risk and project management platforms",
            "Establish risk management KPIs and performance metrics",
            "Create cross-project risk knowledge sharing mechanisms"
        ]
        
        return recommendations
    
    def export_comprehensive_documentation(self) -> str:
        """Export comprehensive documentation to file"""
        # Generate all documentation sections
        executive_summary = self.generate_executive_summary()
        detailed_register = self.generate_detailed_risk_register()
        lessons_learned = self.generate_lessons_learned()
        
        # Compile comprehensive documentation
        comprehensive_doc = {
            "document_metadata": {
                "title": "UAE Telecom Network Infrastructure Risk Management System Documentation",
                "generation_date": self.generation_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "version": "1.0",
                "scope": "Comprehensive RMS Pipeline Documentation"
            },
            "executive_summary": executive_summary,
            "detailed_risk_register": detailed_register,
            "lessons_learned": lessons_learned,
            "appendices": {
                "risk_data_summary": self._create_data_summary(),
                "methodology": self._document_methodology(),
                "glossary": self._create_glossary()
            }
        }
        
        # Export to JSON file
        output_file = config.OUTPUT_DIR / "comprehensive_rms_documentation.json"
        with open(output_file, 'w') as f:
            json.dump(comprehensive_doc, f, indent=2, default=str)
        
        print(f"✓ Comprehensive documentation exported to {output_file}")
        
        return str(output_file)
    
    def _create_data_summary(self) -> Dict[str, any]:
        """Create data summary for appendix"""
        return {
            "data_source": "network_infrastructure_risk_register.csv",
            "total_records": len(self.risks_df),
            "data_fields": list(self.risks_df.columns),
            "data_types": self.risks_df.dtypes.astype(str).to_dict(),
            "completeness_by_field": {
                col: round((self.risks_df[col].notna().sum() / len(self.risks_df)) * 100, 1)
                for col in self.risks_df.columns
            }
        }
    
    def _document_methodology(self) -> Dict[str, str]:
        """Document RMS methodology"""
        return {
            "Risk Identification": "Comprehensive review of project scope and stakeholder input",
            "Qualitative Analysis": "Probability-impact matrix assessment with expert judgment",
            "Quantitative Analysis": "Statistical analysis and Monte Carlo simulation",
            "Response Planning": "Strategy optimization based on risk characteristics",
            "Monitoring": "Continuous tracking with automated alerts and reporting",
            "Documentation": "Comprehensive documentation generation with lessons learned"
        }
    
    def _create_glossary(self) -> Dict[str, str]:
        """Create risk management glossary"""
        return {
            "Risk": "An uncertain event that could have positive or negative impact on project objectives",
            "Risk Register": "A repository of identified risks and their characteristics",
            "Probability": "The likelihood that a risk event will occur",
            "Impact": "The effect on project objectives if the risk occurs",
            "Risk Score": "Quantitative measure calculated from probability and impact",
            "Mitigation": "Actions taken to reduce risk probability or impact",
            "Contingency": "Predetermined responses if risk events occur",
            "Risk Owner": "Individual responsible for managing a specific risk"
        }
    
    def generate_documentation_report(self) -> Dict[str, any]:
        """Generate documentation completion report"""
        documentation_file = self.export_comprehensive_documentation()
        
        report = {
            "documentation_summary": {
                "total_risks_documented": len(self.risks_df),
                "documentation_file": documentation_file,
                "sections_completed": 6,  # Executive, Register, Lessons, etc.
                "completion_status": "Documentation Complete"
            },
            "deliverables": {
                "executive_summary": "Generated",
                "detailed_risk_register": "Generated",
                "lessons_learned": "Generated",
                "comprehensive_documentation": documentation_file
            },
            "recommendations": [
                "Review documentation with stakeholders",
                "Implement lessons learned in future projects",
                "Update documentation as risks evolve",
                "Use documentation for training and knowledge transfer"
            ]
        }
        
        return report

def main():
    """Main execution function for documentation generation"""
    doc_generator = RMSDocumentationGenerator()
    
    print("📋 Starting RMS Documentation Generation...")
    print("=" * 50)
    
    # Generate documentation report
    report = doc_generator.generate_documentation_report()
    
    print(f"✓ Documented {report['documentation_summary']['total_risks_documented']} risks")
    print(f"✓ Generated comprehensive documentation")
    print(f"✓ Documentation file: {report['documentation_summary']['documentation_file']}")
    
    return report

if __name__ == "__main__":
    main()