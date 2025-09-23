"""
Risk Identification Module
Processes the risk register and identifies risks based on various criteria
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import config

class RiskIdentifier:
    """
    Handles risk identification and categorization from the risk register
    """
    
    def __init__(self, data_path: str = None):
        """Initialize the risk identifier with data path"""
        self.data_path = data_path or config.RISK_REGISTER_CSV
        self.risks_df = None
        
    def load_risk_data(self) -> pd.DataFrame:
        """Load and validate risk register data"""
        try:
            self.risks_df = pd.read_csv(self.data_path)
            print(f"✓ Loaded {len(self.risks_df)} risks from register")
            return self.risks_df
        except Exception as e:
            raise Exception(f"Error loading risk data: {e}")
    
    def validate_data_quality(self) -> Dict[str, any]:
        """Validate data quality and completeness"""
        if self.risks_df is None:
            self.load_risk_data()
            
        validation_results = {
            "total_risks": len(self.risks_df),
            "missing_data": {},
            "data_quality_score": 0,
            "completeness": {}
        }
        
        # Check for missing critical fields
        critical_fields = ["Risk ID", "Risk Title", "Risk Category", "Probability Rating", "Impact Rating"]
        for field in critical_fields:
            missing_count = self.risks_df[field].isnull().sum()
            validation_results["missing_data"][field] = missing_count
            validation_results["completeness"][field] = (len(self.risks_df) - missing_count) / len(self.risks_df)
        
        # Calculate overall data quality score
        avg_completeness = np.mean(list(validation_results["completeness"].values()))
        validation_results["data_quality_score"] = avg_completeness
        
        return validation_results
    
    def identify_risk_patterns(self) -> Dict[str, any]:
        """Identify patterns and trends in risk data"""
        if self.risks_df is None:
            self.load_risk_data()
            
        patterns = {
            "risk_by_category": self.risks_df["Risk Category"].value_counts().to_dict(),
            "risk_by_phase": self.risks_df["Project Phase"].value_counts().to_dict(),
            "risk_by_sector": self.risks_df["Sector"].value_counts().to_dict(),
            "high_priority_risks": [],
            "emerging_patterns": {}
        }
        
        # Identify high-priority risks
        high_risk_mask = (
            (self.risks_df["Probability Rating"] == "High") & 
            (self.risks_df["Impact Rating"] == "High")
        )
        patterns["high_priority_risks"] = self.risks_df[high_risk_mask]["Risk ID"].tolist()
        
        # Identify emerging patterns
        patterns["emerging_patterns"] = {
            "construction_risks": len(self.risks_df[self.risks_df["Risk Category"] == "Construction"]),
            "requirements_risks": len(self.risks_df[self.risks_df["Risk Category"] == "Requirements"]),
            "open_risks": len(self.risks_df[self.risks_df["Status"] == "Open"])
        }
        
        return patterns
    
    def categorize_risks_by_rms_step(self) -> Dict[str, List[str]]:
        """Categorize risks by their primary RMS step"""
        if self.risks_df is None:
            self.load_risk_data()
            
        rms_categorization = {}
        for step in config.RMS_STEPS:
            # Find risks primarily in this step
            step_risks = self.risks_df[
                self.risks_df["Primary RMS Step"].str.contains(step, na=False)
            ]["Risk ID"].tolist()
            rms_categorization[step] = step_risks
            
        return rms_categorization
    
    def generate_identification_report(self) -> Dict[str, any]:
        """Generate comprehensive risk identification report"""
        validation = self.validate_data_quality()
        patterns = self.identify_risk_patterns()
        rms_categorization = self.categorize_risks_by_rms_step()
        
        report = {
            "identification_summary": {
                "total_risks_identified": validation["total_risks"],
                "data_quality_score": round(validation["data_quality_score"], 3),
                "critical_risks_count": len(patterns["high_priority_risks"]),
                "completion_status": "Identification Complete"
            },
            "data_validation": validation,
            "risk_patterns": patterns,
            "rms_step_distribution": rms_categorization,
            "recommendations": self._generate_identification_recommendations(patterns, validation)
        }
        
        return report
    
    def _generate_identification_recommendations(self, patterns: Dict, validation: Dict) -> List[str]:
        """Generate recommendations based on identification analysis"""
        recommendations = []
        
        # Data quality recommendations
        if validation["data_quality_score"] < 0.9:
            recommendations.append("Improve data quality by filling missing fields")
            
        # Risk pattern recommendations
        if patterns["emerging_patterns"]["open_risks"] > patterns["emerging_patterns"]["construction_risks"] * 0.8:
            recommendations.append("Focus on closing open risks, particularly in construction category")
            
        if len(patterns["high_priority_risks"]) > 10:
            recommendations.append("Prioritize mitigation for high-impact, high-probability risks")
            
        return recommendations

def main():
    """Main execution function for risk identification"""
    identifier = RiskIdentifier()
    
    print("🔍 Starting Risk Identification Process...")
    print("=" * 50)
    
    # Load and validate data
    identifier.load_risk_data()
    
    # Generate identification report
    report = identifier.generate_identification_report()
    
    print(f"✓ Identified {report['identification_summary']['total_risks_identified']} total risks")
    print(f"✓ Data quality score: {report['identification_summary']['data_quality_score']}")
    print(f"✓ Critical risks: {report['identification_summary']['critical_risks_count']}")
    
    return report

if __name__ == "__main__":
    main()