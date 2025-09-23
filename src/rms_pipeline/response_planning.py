"""
Response Planning Module
Develops and optimizes risk response strategies and mitigation plans
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import config

class ResponsePlanner:
    """
    Handles risk response planning and strategy development
    """
    
    def __init__(self, risks_df: pd.DataFrame = None, analysis_results: Dict = None):
        """Initialize with risk data and analysis results"""
        self.risks_df = risks_df
        if self.risks_df is None:
            self.risks_df = pd.read_csv(config.RISK_REGISTER_CSV)
        
        self.analysis_results = analysis_results or {}
    
    def develop_response_strategies(self) -> Dict[str, any]:
        """Develop comprehensive response strategies for each risk"""
        response_strategies = []
        
        # Define response strategy templates
        strategy_templates = {
            "Avoid": "Eliminate the risk by changing project approach or scope",
            "Transfer": "Transfer risk to third party through insurance or contracts", 
            "Mitigate": "Reduce probability or impact through preventive measures",
            "Accept": "Acknowledge risk and prepare contingency response"
        }
        
        for _, risk in self.risks_df.iterrows():
            risk_score = risk.get("Risk Score", 0)
            prob_rating = risk.get("Probability Rating", "Low")
            impact_rating = risk.get("Impact Rating", "Low")
            existing_strategy = risk.get("Mitigation Strategy", "")
            
            # Determine optimal response strategy
            recommended_strategy = self._determine_optimal_strategy(risk_score, prob_rating, impact_rating)
            
            # Enhance existing mitigation strategy
            enhanced_strategy = self._enhance_mitigation_strategy(existing_strategy, recommended_strategy)
            
            # Develop contingency plan
            contingency_plan = self._develop_contingency_plan(risk, recommended_strategy)
            
            response_strategies.append({
                "Risk ID": risk["Risk ID"],
                "Risk Title": risk["Risk Title"],
                "Current Strategy": existing_strategy,
                "Recommended Strategy Type": recommended_strategy,
                "Enhanced Strategy": enhanced_strategy,
                "Contingency Plan": contingency_plan,
                "Priority Level": self._calculate_response_priority(risk_score, prob_rating, impact_rating),
                "Resource Requirements": self._estimate_resource_requirements(recommended_strategy, risk_score),
                "Timeline": self._estimate_response_timeline(recommended_strategy, risk["Project Phase"]),
                "Success Metrics": self._define_success_metrics(recommended_strategy)
            })
        
        return {
            "response_strategies": response_strategies,
            "strategy_summary": self._summarize_strategies(response_strategies),
            "resource_allocation": self._plan_resource_allocation(response_strategies)
        }
    
    def _determine_optimal_strategy(self, risk_score: float, probability: str, impact: str) -> str:
        """Determine optimal response strategy based on risk characteristics"""
        # High impact risks - consider avoid or transfer
        if impact == "High":
            if probability == "High":
                return "Avoid"  # High-high risks should be avoided if possible
            elif probability == "Medium":
                return "Transfer"  # High impact, medium probability - transfer
            else:
                return "Mitigate"  # High impact, low probability - mitigate
        
        # Medium impact risks - typically mitigate
        elif impact == "Medium":
            if probability == "High":
                return "Mitigate"  # Medium impact, high probability - mitigate
            else:
                return "Accept"   # Medium impact, low/medium probability - accept with monitoring
        
        # Low impact risks - typically accept
        else:
            return "Accept"
    
    def _enhance_mitigation_strategy(self, existing_strategy: str, recommended_type: str) -> str:
        """Enhance existing mitigation strategy based on analysis"""
        if pd.isna(existing_strategy) or existing_strategy == "":
            base_strategy = f"Implement {recommended_type.lower()} strategy"
        else:
            base_strategy = existing_strategy
        
        # Add specific enhancements based on strategy type
        enhancements = {
            "Avoid": "Consider alternative approaches, change project scope, or eliminate risk source",
            "Transfer": "Explore insurance options, contractual risk transfer, or outsourcing",
            "Mitigate": "Implement preventive controls, monitoring systems, and early warning indicators",
            "Accept": "Establish monitoring protocols and prepare rapid response procedures"
        }
        
        enhancement = enhancements.get(recommended_type, "")
        
        if enhancement and enhancement not in base_strategy:
            return f"{base_strategy}. Enhanced approach: {enhancement}"
        
        return base_strategy
    
    def _develop_contingency_plan(self, risk: pd.Series, strategy_type: str) -> str:
        """Develop specific contingency plan for the risk"""
        existing_contingency = risk.get("Contingency Plan", "")
        
        # Template contingency plans based on risk category and strategy
        contingency_templates = {
            "Construction": {
                "Avoid": "Have alternative construction methods ready",
                "Transfer": "Activate contractor insurance and warranty claims",
                "Mitigate": "Deploy backup equipment and additional safety measures",
                "Accept": "Implement emergency response and recovery procedures"
            },
            "Requirements": {
                "Avoid": "Prepare alternative requirement specifications",
                "Transfer": "Engage legal team for contractual resolution",
                "Mitigate": "Initiate stakeholder engagement and negotiation",
                "Accept": "Document requirement changes and impact assessment"
            },
            "Technology": {
                "Avoid": "Switch to proven alternative technology",
                "Transfer": "Activate vendor support and technical warranties",
                "Mitigate": "Deploy technical workarounds and patches",
                "Accept": "Implement system monitoring and performance optimization"
            }
        }
        
        category = risk.get("Risk Category", "General")
        template = contingency_templates.get(category, {}).get(strategy_type, "Implement standard contingency response")
        
        if existing_contingency and template not in existing_contingency:
            return f"{existing_contingency}. Additional contingency: {template}"
        elif not existing_contingency:
            return template
        
        return existing_contingency
    
    def _calculate_response_priority(self, risk_score: float, probability: str, impact: str) -> str:
        """Calculate response priority level"""
        if probability == "High" and impact == "High":
            return "Immediate"
        elif risk_score >= 9:
            return "High"
        elif risk_score >= 6:
            return "Medium"
        else:
            return "Low"
    
    def _estimate_resource_requirements(self, strategy_type: str, risk_score: float) -> Dict[str, str]:
        """Estimate resource requirements for strategy implementation"""
        base_resources = {
            "Avoid": {"budget": "High", "time": "Medium", "personnel": "Medium"},
            "Transfer": {"budget": "Medium", "time": "Low", "personnel": "Low"},
            "Mitigate": {"budget": "Medium", "time": "Medium", "personnel": "Medium"},
            "Accept": {"budget": "Low", "time": "Low", "personnel": "Low"}
        }
        
        resources = base_resources.get(strategy_type, {"budget": "Medium", "time": "Medium", "personnel": "Medium"})
        
        # Adjust based on risk score
        if risk_score >= 15:
            resources = {k: "High" if v != "High" else "Very High" for k, v in resources.items()}
        elif risk_score >= 9:
            resources = {k: "High" if v == "Medium" else v for k, v in resources.items()}
        
        return resources
    
    def _estimate_response_timeline(self, strategy_type: str, project_phase: str) -> str:
        """Estimate timeline for strategy implementation"""
        base_timelines = {
            "Avoid": "2-4 weeks",
            "Transfer": "1-2 weeks", 
            "Mitigate": "1-3 weeks",
            "Accept": "Immediate"
        }
        
        timeline = base_timelines.get(strategy_type, "1-2 weeks")
        
        # Adjust based on project phase
        if project_phase in ["Requirements", "Planning"]:
            return timeline  # Standard timeline
        elif project_phase in ["Implementation", "Executing"]:
            return "Immediate - 1 week"  # Faster response needed
        else:
            return timeline
    
    def _define_success_metrics(self, strategy_type: str) -> List[str]:
        """Define success metrics for strategy implementation"""
        metrics = {
            "Avoid": [
                "Risk completely eliminated from register",
                "No occurrence of risk event",
                "Alternative approach successfully implemented"
            ],
            "Transfer": [
                "Risk transfer agreement executed",
                "Third party accepts risk ownership", 
                "Insurance/contract coverage confirmed"
            ],
            "Mitigate": [
                "Risk probability reduced by 50%",
                "Risk impact reduced by 50%",
                "Early warning system operational"
            ],
            "Accept": [
                "Monitoring system established",
                "Contingency plan activated if needed",
                "Risk remains within acceptable tolerance"
            ]
        }
        
        return metrics.get(strategy_type, ["Strategy implementation completed"])
    
    def _summarize_strategies(self, strategies: List[Dict]) -> Dict[str, any]:
        """Summarize response strategies across all risks"""
        strategy_counts = {}
        priority_counts = {}
        
        for strategy in strategies:
            # Count strategy types
            strategy_type = strategy["Recommended Strategy Type"]
            strategy_counts[strategy_type] = strategy_counts.get(strategy_type, 0) + 1
            
            # Count priorities
            priority = strategy["Priority Level"]
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        return {
            "total_strategies": len(strategies),
            "strategy_type_distribution": strategy_counts,
            "priority_distribution": priority_counts,
            "immediate_action_required": priority_counts.get("Immediate", 0),
            "high_priority_strategies": priority_counts.get("High", 0)
        }
    
    def _plan_resource_allocation(self, strategies: List[Dict]) -> Dict[str, any]:
        """Plan resource allocation across all response strategies"""
        resource_plan = {
            "total_budget_requirement": "Medium",
            "total_time_requirement": "Medium", 
            "total_personnel_requirement": "Medium",
            "immediate_resources_needed": 0,
            "high_priority_resources": 0
        }
        
        immediate_count = 0
        high_priority_count = 0
        
        for strategy in strategies:
            if strategy["Priority Level"] == "Immediate":
                immediate_count += 1
            elif strategy["Priority Level"] == "High":
                high_priority_count += 1
        
        resource_plan["immediate_resources_needed"] = immediate_count
        resource_plan["high_priority_resources"] = high_priority_count
        
        # Adjust overall requirements based on high-priority strategies
        if immediate_count > 5:
            resource_plan["total_budget_requirement"] = "Very High"
            resource_plan["total_time_requirement"] = "High"
            resource_plan["total_personnel_requirement"] = "High"
        elif immediate_count > 2 or high_priority_count > 10:
            resource_plan["total_budget_requirement"] = "High"
            resource_plan["total_time_requirement"] = "Medium"
            resource_plan["total_personnel_requirement"] = "Medium"
        
        return resource_plan
    
    def optimize_response_portfolio(self, budget_constraint: float = None) -> Dict[str, any]:
        """Optimize the portfolio of response strategies given constraints"""
        strategies = self.develop_response_strategies()["response_strategies"]
        
        # Simple optimization based on priority and resource efficiency
        optimized_strategies = []
        
        # Sort by priority and resource efficiency
        priority_order = {"Immediate": 4, "High": 3, "Medium": 2, "Low": 1}
        
        for strategy in strategies:
            priority_score = priority_order.get(strategy["Priority Level"], 1)
            
            # Simple resource efficiency score (inverse of resource requirements)
            budget_req = strategy["Resource Requirements"]["budget"]
            efficiency_score = {"Low": 3, "Medium": 2, "High": 1, "Very High": 0.5}.get(budget_req, 2)
            
            combined_score = priority_score * efficiency_score
            
            optimized_strategies.append({
                **strategy,
                "optimization_score": combined_score,
                "recommended_for_implementation": combined_score >= 4
            })
        
        # Sort by optimization score
        optimized_strategies.sort(key=lambda x: x["optimization_score"], reverse=True)
        
        return {
            "optimized_strategies": optimized_strategies,
            "recommended_immediate": [s for s in optimized_strategies if s["recommended_for_implementation"]],
            "optimization_summary": {
                "total_strategies": len(optimized_strategies),
                "recommended_count": len([s for s in optimized_strategies if s["recommended_for_implementation"]]),
                "optimization_criteria": "Priority and resource efficiency"
            }
        }
    
    def generate_response_report(self) -> Dict[str, any]:
        """Generate comprehensive response planning report"""
        response_strategies = self.develop_response_strategies()
        optimized_portfolio = self.optimize_response_portfolio()
        
        report = {
            "planning_summary": {
                "total_risks_planned": len(response_strategies["response_strategies"]),
                "immediate_responses_needed": response_strategies["strategy_summary"]["immediate_action_required"],
                "high_priority_responses": response_strategies["strategy_summary"]["high_priority_strategies"],
                "completion_status": "Response Planning Complete"
            },
            "response_strategies": response_strategies,
            "optimized_portfolio": optimized_portfolio,
            "recommendations": self._generate_response_recommendations(response_strategies, optimized_portfolio)
        }
        
        return report
    
    def _generate_response_recommendations(self, strategies: Dict, portfolio: Dict) -> List[str]:
        """Generate recommendations for response planning"""
        recommendations = []
        
        immediate_count = strategies["strategy_summary"]["immediate_action_required"]
        if immediate_count > 0:
            recommendations.append(f"Execute {immediate_count} immediate response strategies within 1 week")
        
        avoid_count = strategies["strategy_summary"]["strategy_type_distribution"].get("Avoid", 0)
        if avoid_count > 3:
            recommendations.append("Consider project scope changes to avoid high-risk activities")
        
        recommended_count = portfolio["optimization_summary"]["recommended_count"]
        total_count = portfolio["optimization_summary"]["total_strategies"]
        if recommended_count < total_count * 0.7:
            recommendations.append("Review resource allocation to implement more response strategies")
        
        return recommendations

def main():
    """Main execution function for response planning"""
    planner = ResponsePlanner()
    
    print("🎯 Starting Risk Response Planning...")
    print("=" * 50)
    
    # Generate response planning report
    report = planner.generate_response_report()
    
    print(f"✓ Planned responses for {report['planning_summary']['total_risks_planned']} risks")
    print(f"✓ Immediate responses needed: {report['planning_summary']['immediate_responses_needed']}")
    print(f"✓ High priority responses: {report['planning_summary']['high_priority_responses']}")
    
    return report

if __name__ == "__main__":
    main()