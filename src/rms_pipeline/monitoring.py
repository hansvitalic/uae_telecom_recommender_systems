"""
Risk Monitoring Module
Implements continuous risk monitoring and tracking systems
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import config

class RiskMonitor:
    """
    Handles continuous risk monitoring and tracking
    """
    
    def __init__(self, risks_df: pd.DataFrame = None):
        """Initialize with risk data"""
        self.risks_df = risks_df
        if self.risks_df is None:
            self.risks_df = pd.read_csv(config.RISK_REGISTER_CSV)
            
        # Convert date columns
        self._process_dates()
    
    def _process_dates(self):
        """Process and standardize date columns"""
        # Convert date columns to datetime (using dummy dates for simulation)
        date_columns = ["Date Identified", "Review Date"]
        
        for col in date_columns:
            if col in self.risks_df.columns:
                # Convert week notation to actual dates (for demonstration)
                self.risks_df[f"{col}_processed"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(
                    self.risks_df[col].str.extract(r"Week (\d+)")[0].astype(float, errors='ignore') * 7, 
                    unit='days',
                    errors='coerce'
                )
    
    def establish_monitoring_framework(self) -> Dict[str, any]:
        """Establish comprehensive risk monitoring framework"""
        monitoring_framework = {
            "monitoring_protocols": self._define_monitoring_protocols(),
            "kpi_dashboard": self._create_kpi_dashboard(),
            "alert_thresholds": self._set_alert_thresholds(),
            "reporting_schedule": self._define_reporting_schedule()
        }
        
        return monitoring_framework
    
    def _define_monitoring_protocols(self) -> Dict[str, any]:
        """Define monitoring protocols for different risk types"""
        protocols = {}
        
        # Group risks by monitoring requirements
        for category in self.risks_df["Risk Category"].unique():
            category_risks = self.risks_df[self.risks_df["Risk Category"] == category]
            
            # Determine monitoring frequency based on risk characteristics
            high_risk_count = len(category_risks[
                (category_risks["Probability Rating"] == "High") | 
                (category_risks["Impact Rating"] == "High")
            ])
            
            if high_risk_count > len(category_risks) * 0.5:
                frequency = "Daily"
            elif high_risk_count > len(category_risks) * 0.3:
                frequency = "Weekly"
            else:
                frequency = "Bi-weekly"
            
            protocols[category] = {
                "monitoring_frequency": frequency,
                "total_risks": len(category_risks),
                "high_priority_risks": high_risk_count,
                "monitoring_methods": self._get_monitoring_methods(category),
                "responsible_parties": self._identify_responsible_parties(category_risks)
            }
        
        return protocols
    
    def _get_monitoring_methods(self, category: str) -> List[str]:
        """Get appropriate monitoring methods for risk category"""
        method_mapping = {
            "Construction": ["Site inspection", "Progress tracking", "Safety audits", "Quality checks"],
            "Requirements": ["Stakeholder meetings", "Document reviews", "Approval tracking"],
            "Technology": ["System monitoring", "Performance metrics", "Technical audits"],
            "Environmental": ["Environmental surveys", "Compliance checks", "Impact assessments"],
            "Financial": ["Budget tracking", "Cost monitoring", "Financial audits"]
        }
        
        return method_mapping.get(category, ["Regular review", "Status updates", "Performance monitoring"])
    
    def _identify_responsible_parties(self, category_risks: pd.DataFrame) -> Dict[str, int]:
        """Identify responsible parties for monitoring"""
        parties = category_risks["Risk Owner"].value_counts().to_dict()
        return parties
    
    def _create_kpi_dashboard(self) -> Dict[str, any]:
        """Create key performance indicators for risk monitoring"""
        kpis = {
            "overall_metrics": {
                "total_active_risks": len(self.risks_df[self.risks_df["Status"] == "Open"]),
                "critical_risks": len(self.risks_df[
                    (self.risks_df["Probability Rating"] == "High") & 
                    (self.risks_df["Impact Rating"] == "High")
                ]),
                "risk_closure_rate": self._calculate_risk_closure_rate(),
                "average_risk_score": self.risks_df["Risk Score"].mean()
            },
            "category_metrics": self._calculate_category_metrics(),
            "phase_metrics": self._calculate_phase_metrics(),
            "trend_indicators": self._calculate_trend_indicators()
        }
        
        return kpis
    
    def _calculate_risk_closure_rate(self) -> float:
        """Calculate the rate of risk closure"""
        total_risks = len(self.risks_df)
        closed_risks = len(self.risks_df[self.risks_df["Status"] != "Open"])
        return closed_risks / total_risks if total_risks > 0 else 0
    
    def _calculate_category_metrics(self) -> Dict[str, Dict]:
        """Calculate metrics by risk category"""
        metrics = {}
        
        for category in self.risks_df["Risk Category"].unique():
            category_data = self.risks_df[self.risks_df["Risk Category"] == category]
            
            metrics[category] = {
                "total_risks": len(category_data),
                "open_risks": len(category_data[category_data["Status"] == "Open"]),
                "avg_risk_score": category_data["Risk Score"].mean(),
                "high_priority_count": len(category_data[
                    (category_data["Probability Rating"] == "High") | 
                    (category_data["Impact Rating"] == "High")
                ])
            }
        
        return metrics
    
    def _calculate_phase_metrics(self) -> Dict[str, Dict]:
        """Calculate metrics by project phase"""
        metrics = {}
        
        for phase in self.risks_df["Project Phase"].unique():
            phase_data = self.risks_df[self.risks_df["Project Phase"] == phase]
            
            metrics[phase] = {
                "total_risks": len(phase_data),
                "open_risks": len(phase_data[phase_data["Status"] == "Open"]),
                "avg_risk_score": phase_data["Risk Score"].mean(),
                "risk_density": len(phase_data) / len(self.risks_df)
            }
        
        return metrics
    
    def _calculate_trend_indicators(self) -> Dict[str, any]:
        """Calculate trend indicators for risk evolution"""
        # Simulate trend data (in real implementation, this would use historical data)
        trends = {
            "risk_emergence_rate": "Stable",  # Based on new risks identified
            "risk_escalation_rate": "Decreasing",  # Based on risk score increases
            "mitigation_effectiveness": "Good",  # Based on risk score reductions
            "risk_materialization_rate": "Low"  # Based on risks that occurred
        }
        
        # Add some quantitative indicators
        open_risks_pct = len(self.risks_df[self.risks_df["Status"] == "Open"]) / len(self.risks_df)
        
        if open_risks_pct > 0.8:
            trends["overall_trend"] = "Deteriorating"
        elif open_risks_pct > 0.6:
            trends["overall_trend"] = "Stable"
        else:
            trends["overall_trend"] = "Improving"
        
        trends["open_risks_percentage"] = round(open_risks_pct * 100, 1)
        
        return trends
    
    def _set_alert_thresholds(self) -> Dict[str, any]:
        """Set alert thresholds for automated monitoring"""
        thresholds = {
            "critical_risk_threshold": {
                "description": "Risks with both high probability and high impact",
                "threshold": 0,  # Any critical risk triggers alert
                "current_count": len(self.risks_df[
                    (self.risks_df["Probability Rating"] == "High") & 
                    (self.risks_df["Impact Rating"] == "High")
                ]),
                "alert_level": "Immediate"
            },
            "risk_score_threshold": {
                "description": "Risks with score above threshold",
                "threshold": 12,
                "current_count": len(self.risks_df[self.risks_df["Risk Score"] > 12]),
                "alert_level": "High"
            },
            "open_risks_threshold": {
                "description": "Percentage of open risks",
                "threshold": 80,  # 80% open risks
                "current_percentage": (len(self.risks_df[self.risks_df["Status"] == "Open"]) / len(self.risks_df)) * 100,
                "alert_level": "Medium"
            },
            "overdue_reviews_threshold": {
                "description": "Risks with overdue reviews",
                "threshold": 5,  # 5 overdue reviews
                "current_count": self._count_overdue_reviews(),
                "alert_level": "Medium"
            }
        }
        
        # Evaluate current alert status
        for threshold_name, threshold_data in thresholds.items():
            if "current_count" in threshold_data:
                thresholds[threshold_name]["alert_triggered"] = threshold_data["current_count"] > threshold_data["threshold"]
            elif "current_percentage" in threshold_data:
                thresholds[threshold_name]["alert_triggered"] = threshold_data["current_percentage"] > threshold_data["threshold"]
        
        return thresholds
    
    def _count_overdue_reviews(self) -> int:
        """Count risks with overdue reviews"""
        # Simplified logic - in real implementation would use actual dates
        current_week = 10  # Assume current week
        overdue_count = 0
        
        for _, risk in self.risks_df.iterrows():
            review_date = risk.get("Review Date", "")
            if "Week" in str(review_date):
                try:
                    review_week = int(str(review_date).split("Week ")[1])
                    if review_week < current_week:
                        overdue_count += 1
                except:
                    continue
        
        return overdue_count
    
    def _define_reporting_schedule(self) -> Dict[str, any]:
        """Define automated reporting schedule"""
        schedule = {
            "daily_reports": {
                "recipients": ["Project Manager", "Risk Owner"],
                "content": ["Critical risk status", "New risks identified", "Alert notifications"],
                "format": "Email summary"
            },
            "weekly_reports": {
                "recipients": ["Senior Management", "Project Team"],
                "content": ["Risk dashboard", "Trend analysis", "Mitigation progress"],
                "format": "Dashboard + PDF report"
            },
            "monthly_reports": {
                "recipients": ["Executive Team", "Stakeholders"],
                "content": ["Comprehensive risk review", "Strategic recommendations", "Performance metrics"],
                "format": "Executive summary + presentation"
            },
            "ad_hoc_reports": {
                "triggers": ["Critical risk identification", "Major risk escalation", "Incident occurrence"],
                "recipients": ["All stakeholders"],
                "content": ["Immediate notification", "Impact assessment", "Response actions"],
                "format": "Urgent alert + detailed report"
            }
        }
        
        return schedule
    
    def track_risk_evolution(self) -> Dict[str, any]:
        """Track how risks evolve over time"""
        evolution_data = {
            "risk_lifecycle_analysis": self._analyze_risk_lifecycle(),
            "score_evolution": self._track_score_evolution(),
            "status_transitions": self._analyze_status_transitions(),
            "mitigation_effectiveness": self._evaluate_mitigation_effectiveness()
        }
        
        return evolution_data
    
    def _analyze_risk_lifecycle(self) -> Dict[str, any]:
        """Analyze risk lifecycle patterns"""
        lifecycle_analysis = {
            "identification_patterns": {},
            "resolution_patterns": {},
            "escalation_patterns": {}
        }
        
        # Group by RMS step to understand lifecycle
        for step in config.RMS_STEPS:
            step_risks = self.risks_df[self.risks_df["Primary RMS Step"].str.contains(step, na=False)]
            
            lifecycle_analysis["identification_patterns"][step] = {
                "risk_count": len(step_risks),
                "avg_score": step_risks["Risk Score"].mean(),
                "predominant_category": step_risks["Risk Category"].mode().iloc[0] if len(step_risks) > 0 else "N/A"
            }
        
        return lifecycle_analysis
    
    def _track_score_evolution(self) -> Dict[str, any]:
        """Track risk score evolution (simulated)"""
        # In real implementation, this would track historical score changes
        score_evolution = {
            "score_distribution": self.risks_df["Risk Score"].value_counts().sort_index().to_dict(),
            "score_statistics": {
                "mean": self.risks_df["Risk Score"].mean(),
                "median": self.risks_df["Risk Score"].median(),
                "std": self.risks_df["Risk Score"].std(),
                "max": self.risks_df["Risk Score"].max(),
                "min": self.risks_df["Risk Score"].min()
            },
            "high_score_risks": len(self.risks_df[self.risks_df["Risk Score"] >= 15]),
            "score_trend": "Stable"  # Would be calculated from historical data
        }
        
        return score_evolution
    
    def _analyze_status_transitions(self) -> Dict[str, any]:
        """Analyze risk status transitions"""
        status_analysis = {
            "current_status_distribution": self.risks_df["Status"].value_counts().to_dict(),
            "status_by_category": {},
            "status_by_phase": {}
        }
        
        # Status distribution by category
        for category in self.risks_df["Risk Category"].unique():
            category_data = self.risks_df[self.risks_df["Risk Category"] == category]
            status_analysis["status_by_category"][category] = category_data["Status"].value_counts().to_dict()
        
        # Status distribution by phase
        for phase in self.risks_df["Project Phase"].unique():
            phase_data = self.risks_df[self.risks_df["Project Phase"] == phase]
            status_analysis["status_by_phase"][phase] = phase_data["Status"].value_counts().to_dict()
        
        return status_analysis
    
    def _evaluate_mitigation_effectiveness(self) -> Dict[str, any]:
        """Evaluate effectiveness of mitigation strategies"""
        effectiveness = {
            "strategies_with_mitigation": len(self.risks_df[self.risks_df["Mitigation Strategy"].notna()]),
            "strategies_with_contingency": len(self.risks_df[self.risks_df["Contingency Plan"].notna()]),
            "mitigation_coverage": len(self.risks_df[self.risks_df["Mitigation Strategy"].notna()]) / len(self.risks_df),
            "avg_score_with_mitigation": self.risks_df[self.risks_df["Mitigation Strategy"].notna()]["Risk Score"].mean(),
            "avg_score_without_mitigation": self.risks_df[self.risks_df["Mitigation Strategy"].isna()]["Risk Score"].mean()
        }
        
        # Calculate effectiveness indicator
        if effectiveness["avg_score_with_mitigation"] < effectiveness["avg_score_without_mitigation"]:
            effectiveness["mitigation_impact"] = "Positive"
        else:
            effectiveness["mitigation_impact"] = "Needs Review"
        
        return effectiveness
    
    def generate_monitoring_report(self) -> Dict[str, any]:
        """Generate comprehensive monitoring report"""
        monitoring_framework = self.establish_monitoring_framework()
        risk_evolution = self.track_risk_evolution()
        
        # Check for active alerts
        active_alerts = []
        for threshold_name, threshold_data in monitoring_framework["alert_thresholds"].items():
            if threshold_data.get("alert_triggered", False):
                active_alerts.append({
                    "alert_type": threshold_name,
                    "alert_level": threshold_data["alert_level"],
                    "description": threshold_data["description"]
                })
        
        report = {
            "monitoring_summary": {
                "total_risks_monitored": len(self.risks_df),
                "active_risks": len(self.risks_df[self.risks_df["Status"] == "Open"]),
                "active_alerts": len(active_alerts),
                "monitoring_effectiveness": "Good",  # Based on comprehensive framework
                "completion_status": "Risk Monitoring Active"
            },
            "monitoring_framework": monitoring_framework,
            "risk_evolution": risk_evolution,
            "active_alerts": active_alerts,
            "recommendations": self._generate_monitoring_recommendations(monitoring_framework, active_alerts)
        }
        
        return report
    
    def _generate_monitoring_recommendations(self, framework: Dict, alerts: List) -> List[str]:
        """Generate recommendations for monitoring improvement"""
        recommendations = []
        
        # Alert-based recommendations
        if len(alerts) > 0:
            high_alerts = [a for a in alerts if a["alert_level"] in ["Critical", "High", "Immediate"]]
            if high_alerts:
                recommendations.append(f"Address {len(high_alerts)} high-priority alerts immediately")
        
        # Framework-based recommendations
        protocols = framework["monitoring_protocols"]
        daily_monitoring_categories = [cat for cat, data in protocols.items() if data["monitoring_frequency"] == "Daily"]
        
        if len(daily_monitoring_categories) > 3:
            recommendations.append("Consider automated monitoring tools to reduce manual effort")
        
        # KPI-based recommendations
        kpis = framework["kpi_dashboard"]
        if kpis["overall_metrics"]["risk_closure_rate"] < 0.3:
            recommendations.append("Accelerate risk closure activities and mitigation implementation")
        
        return recommendations

def main():
    """Main execution function for risk monitoring"""
    monitor = RiskMonitor()
    
    print("👁️ Starting Risk Monitoring Setup...")
    print("=" * 50)
    
    # Generate monitoring report
    report = monitor.generate_monitoring_report()
    
    print(f"✓ Monitoring {report['monitoring_summary']['total_risks_monitored']} risks")
    print(f"✓ Active risks: {report['monitoring_summary']['active_risks']}")
    print(f"✓ Active alerts: {report['monitoring_summary']['active_alerts']}")
    
    return report

if __name__ == "__main__":
    main()