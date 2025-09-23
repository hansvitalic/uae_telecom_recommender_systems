"""
Quantitative Analysis Module
Performs quantitative risk analysis including statistical modeling and financial impact assessment
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import config

class QuantitativeAnalyzer:
    """
    Handles quantitative risk analysis and statistical modeling
    """
    
    def __init__(self, risks_df: pd.DataFrame = None):
        """Initialize with risk data"""
        self.risks_df = risks_df
        if self.risks_df is None:
            self.risks_df = pd.read_csv(config.RISK_REGISTER_CSV)
        
        # Convert ratings to numerical values for calculations
        self.prob_numeric = self.risks_df["Probability Rating"].map(config.PROBABILITY_WEIGHTS)
        self.impact_numeric = self.risks_df["Impact Rating"].map(config.IMPACT_WEIGHTS)
    
    def calculate_risk_exposure(self) -> Dict[str, any]:
        """Calculate risk exposure and expected monetary value"""
        # Estimate cost impacts (using dummy values for demonstration)
        # In real implementation, these would come from historical data or expert estimates
        base_cost_estimates = {
            "Low": 10000,      # $10K
            "Medium": 50000,   # $50K  
            "High": 200000     # $200K
        }
        
        risk_exposure_data = []
        
        for _, risk in self.risks_df.iterrows():
            prob_rating = risk["Probability Rating"]
            impact_rating = risk["Impact Rating"]
            
            # Calculate probability as percentage
            prob_pct = config.PROBABILITY_WEIGHTS.get(prob_rating, 0) * 20  # 20%, 60%, 100%
            
            # Estimate cost impact
            base_cost = base_cost_estimates.get(impact_rating, 0)
            
            # Calculate expected monetary value (EMV)
            emv = (prob_pct / 100) * base_cost
            
            risk_exposure_data.append({
                "Risk ID": risk["Risk ID"],
                "Risk Title": risk["Risk Title"],
                "Probability %": prob_pct,
                "Impact Cost": base_cost,
                "Expected Monetary Value": round(emv, 2),
                "Risk Category": risk["Risk Category"],
                "Project Phase": risk["Project Phase"]
            })
        
        exposure_df = pd.DataFrame(risk_exposure_data)
        
        # Summary statistics
        summary = {
            "total_risk_exposure": exposure_df["Expected Monetary Value"].sum(),
            "avg_risk_exposure": exposure_df["Expected Monetary Value"].mean(),
            "max_risk_exposure": exposure_df["Expected Monetary Value"].max(),
            "min_risk_exposure": exposure_df["Expected Monetary Value"].min(),
            "exposure_by_category": exposure_df.groupby("Risk Category")["Expected Monetary Value"].sum().to_dict(),
            "exposure_by_phase": exposure_df.groupby("Project Phase")["Expected Monetary Value"].sum().to_dict(),
            "exposure_data": exposure_df
        }
        
        return summary
    
    def perform_statistical_analysis(self) -> Dict[str, any]:
        """Perform statistical analysis on risk data"""
        # Risk score statistics
        risk_scores = self.risks_df["Risk Score"].dropna()
        
        # Correlation analysis
        numerical_cols = ["Risk Score"]
        prob_impact_df = pd.DataFrame({
            "Risk Score": risk_scores,
            "Probability Numeric": self.prob_numeric[risk_scores.index],
            "Impact Numeric": self.impact_numeric[risk_scores.index]
        })
        
        correlation_matrix = prob_impact_df.corr()
        
        # Distribution analysis
        score_distribution = {
            "mean": risk_scores.mean(),
            "median": risk_scores.median(),
            "std": risk_scores.std(),
            "min": risk_scores.min(),
            "max": risk_scores.max(),
            "q25": risk_scores.quantile(0.25),
            "q75": risk_scores.quantile(0.75)
        }
        
        # Risk concentration analysis
        category_stats = self.risks_df.groupby("Risk Category")["Risk Score"].agg([
            "count", "mean", "std", "min", "max"
        ]).round(2)
        
        phase_stats = self.risks_df.groupby("Project Phase")["Risk Score"].agg([
            "count", "mean", "std", "min", "max"
        ]).round(2)
        
        analysis = {
            "correlation_matrix": correlation_matrix.to_dict(),
            "risk_score_distribution": score_distribution,
            "category_statistics": category_stats.to_dict(),
            "phase_statistics": phase_stats.to_dict(),
            "data_quality_metrics": self._calculate_data_quality_metrics()
        }
        
        return analysis
    
    def _calculate_data_quality_metrics(self) -> Dict[str, float]:
        """Calculate data quality metrics for quantitative analysis"""
        total_records = len(self.risks_df)
        
        metrics = {
            "completeness_risk_score": (self.risks_df["Risk Score"].notna().sum() / total_records),
            "completeness_probability": (self.risks_df["Probability Rating"].notna().sum() / total_records),
            "completeness_impact": (self.risks_df["Impact Rating"].notna().sum() / total_records),
            "consistency_score": self._check_score_consistency(),
            "validity_ratings": self._check_rating_validity()
        }
        
        return {k: round(v, 3) for k, v in metrics.items()}
    
    def _check_score_consistency(self) -> float:
        """Check consistency between risk scores and probability/impact ratings"""
        consistent_records = 0
        total_valid_records = 0
        
        for _, risk in self.risks_df.iterrows():
            if pd.notna(risk["Risk Score"]) and pd.notna(risk["Probability Rating"]) and pd.notna(risk["Impact Rating"]):
                total_valid_records += 1
                
                # Calculate expected score
                prob_weight = config.PROBABILITY_WEIGHTS.get(risk["Probability Rating"], 0)
                impact_weight = config.IMPACT_WEIGHTS.get(risk["Impact Rating"], 0)
                expected_score = prob_weight * impact_weight
                
                # Check if actual score is close to expected (within tolerance)
                if abs(risk["Risk Score"] - expected_score) <= 2:  # tolerance of 2
                    consistent_records += 1
        
        return consistent_records / total_valid_records if total_valid_records > 0 else 0
    
    def _check_rating_validity(self) -> float:
        """Check validity of probability and impact ratings"""
        valid_prob_values = set(config.PROBABILITY_WEIGHTS.keys())
        valid_impact_values = set(config.IMPACT_WEIGHTS.keys())
        
        prob_valid = self.risks_df["Probability Rating"].isin(valid_prob_values).sum()
        impact_valid = self.risks_df["Impact Rating"].isin(valid_impact_values).sum()
        
        total_ratings = len(self.risks_df) * 2  # probability + impact
        valid_ratings = prob_valid + impact_valid
        
        return valid_ratings / total_ratings
    
    def monte_carlo_simulation(self, iterations: int = 1000) -> Dict[str, any]:
        """Perform Monte Carlo simulation for risk scenarios"""
        # Simplified Monte Carlo for total project risk
        simulation_results = []
        
        for _ in range(iterations):
            total_impact = 0
            
            for _, risk in self.risks_df.iterrows():
                # Random probability realization
                prob_rating = risk["Probability Rating"]
                if prob_rating == "High":
                    prob = np.random.uniform(0.7, 0.9)
                elif prob_rating == "Medium":  
                    prob = np.random.uniform(0.3, 0.7)
                else:  # Low
                    prob = np.random.uniform(0.1, 0.3)
                
                # Random impact realization
                impact_rating = risk["Impact Rating"]
                if impact_rating == "High":
                    impact = np.random.uniform(150000, 250000)
                elif impact_rating == "Medium":
                    impact = np.random.uniform(30000, 70000)
                else:  # Low
                    impact = np.random.uniform(5000, 15000)
                
                # Determine if risk occurs
                if np.random.random() < prob:
                    total_impact += impact
            
            simulation_results.append(total_impact)
        
        # Calculate statistics
        simulation_array = np.array(simulation_results)
        
        monte_carlo_summary = {
            "iterations": iterations,
            "mean_total_impact": simulation_array.mean(),
            "std_total_impact": simulation_array.std(),
            "min_impact": simulation_array.min(),
            "max_impact": simulation_array.max(),
            "percentile_50": np.percentile(simulation_array, 50),
            "percentile_80": np.percentile(simulation_array, 80),
            "percentile_95": np.percentile(simulation_array, 95),
            "value_at_risk_95": np.percentile(simulation_array, 95),
            "simulation_data": simulation_results[:100]  # Sample for visualization
        }
        
        return monte_carlo_summary
    
    def sensitivity_analysis(self) -> Dict[str, any]:
        """Perform sensitivity analysis on key risk parameters"""
        base_exposure = self.calculate_risk_exposure()
        sensitivity_results = {}
        
        # Test sensitivity to probability changes
        prob_scenarios = ["Low", "Medium", "High"]
        for scenario in prob_scenarios:
            # Temporarily modify all probabilities
            modified_df = self.risks_df.copy()
            modified_df["Probability Rating"] = scenario
            
            temp_analyzer = QuantitativeAnalyzer(modified_df)
            scenario_exposure = temp_analyzer.calculate_risk_exposure()
            
            sensitivity_results[f"all_prob_{scenario.lower()}"] = {
                "total_exposure": scenario_exposure["total_risk_exposure"],
                "change_from_base": scenario_exposure["total_risk_exposure"] - base_exposure["total_risk_exposure"]
            }
        
        # Test sensitivity to impact changes
        impact_scenarios = ["Low", "Medium", "High"]  
        for scenario in impact_scenarios:
            modified_df = self.risks_df.copy()
            modified_df["Impact Rating"] = scenario
            
            temp_analyzer = QuantitativeAnalyzer(modified_df)
            scenario_exposure = temp_analyzer.calculate_risk_exposure()
            
            sensitivity_results[f"all_impact_{scenario.lower()}"] = {
                "total_exposure": scenario_exposure["total_risk_exposure"],
                "change_from_base": scenario_exposure["total_risk_exposure"] - base_exposure["total_risk_exposure"]
            }
        
        return {
            "base_exposure": base_exposure["total_risk_exposure"],
            "sensitivity_scenarios": sensitivity_results,
            "most_sensitive_parameter": self._identify_most_sensitive_parameter(sensitivity_results)
        }
    
    def _identify_most_sensitive_parameter(self, sensitivity_results: Dict) -> str:
        """Identify the parameter with highest sensitivity"""
        max_change = 0
        most_sensitive = ""
        
        for scenario, results in sensitivity_results.items():
            change = abs(results["change_from_base"])
            if change > max_change:
                max_change = change
                most_sensitive = scenario
        
        return most_sensitive
    
    def generate_quantitative_report(self) -> Dict[str, any]:
        """Generate comprehensive quantitative analysis report"""
        risk_exposure = self.calculate_risk_exposure()
        statistical_analysis = self.perform_statistical_analysis()
        monte_carlo_results = self.monte_carlo_simulation()
        sensitivity_results = self.sensitivity_analysis()
        
        report = {
            "analysis_summary": {
                "total_risks_analyzed": len(self.risks_df),
                "total_risk_exposure": round(risk_exposure["total_risk_exposure"], 2),
                "avg_risk_exposure": round(risk_exposure["avg_risk_exposure"], 2),
                "value_at_risk_95": round(monte_carlo_results["value_at_risk_95"], 2),
                "completion_status": "Quantitative Analysis Complete"
            },
            "risk_exposure": risk_exposure,
            "statistical_analysis": statistical_analysis,
            "monte_carlo_simulation": monte_carlo_results,
            "sensitivity_analysis": sensitivity_results,
            "recommendations": self._generate_quantitative_recommendations(risk_exposure, monte_carlo_results, sensitivity_results)
        }
        
        return report
    
    def _generate_quantitative_recommendations(self, exposure: Dict, monte_carlo: Dict, sensitivity: Dict) -> List[str]:
        """Generate recommendations based on quantitative analysis"""
        recommendations = []
        
        # Exposure-based recommendations
        if exposure["total_risk_exposure"] > 1000000:  # $1M
            recommendations.append("Consider risk transfer mechanisms for high exposure projects")
        
        # Monte Carlo-based recommendations
        var_95 = monte_carlo["value_at_risk_95"]
        mean_impact = monte_carlo["mean_total_impact"]
        if var_95 > mean_impact * 2:
            recommendations.append("High variability detected - establish larger contingency reserves")
        
        # Sensitivity-based recommendations
        most_sensitive = sensitivity["most_sensitive_parameter"]
        if "prob" in most_sensitive:
            recommendations.append("Focus on improving probability estimation accuracy")
        elif "impact" in most_sensitive:
            recommendations.append("Focus on better impact assessment and cost estimation")
        
        return recommendations

def main():
    """Main execution function for quantitative analysis"""
    analyzer = QuantitativeAnalyzer()
    
    print("📈 Starting Quantitative Risk Analysis...")
    print("=" * 50)
    
    # Generate quantitative analysis report
    report = analyzer.generate_quantitative_report()
    
    print(f"✓ Analyzed {report['analysis_summary']['total_risks_analyzed']} risks")
    print(f"✓ Total risk exposure: ${report['analysis_summary']['total_risk_exposure']:,.2f}")
    print(f"✓ Value at Risk (95%): ${report['analysis_summary']['value_at_risk_95']:,.2f}")
    
    return report

if __name__ == "__main__":
    main()