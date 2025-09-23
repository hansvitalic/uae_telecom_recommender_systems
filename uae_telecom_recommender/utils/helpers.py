"""
Utility Functions for UAE Telecom Recommender Systems

Common utilities, helpers, and data processing functions.
"""

import json
import csv
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
from ..models.project import Project, ProjectRisk, ProjectStatus, RiskLevel, RiskCategory
from ..models.sector import TelecomSectorType


class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder for datetime objects"""
    
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)


def calculate_project_duration(start_date: date, end_date: Optional[date] = None) -> int:
    """Calculate project duration in days"""
    end = end_date or date.today()
    return (end - start_date).days


def format_currency_aed(amount: float) -> str:
    """Format amount in AED currency"""
    if amount >= 1_000_000_000:
        return f"AED {amount/1_000_000_000:.1f}B"
    elif amount >= 1_000_000:
        return f"AED {amount/1_000_000:.1f}M"
    elif amount >= 1_000:
        return f"AED {amount/1_000:.1f}K"
    else:
        return f"AED {amount:.0f}"


def calculate_risk_score_normalized(risks: List[ProjectRisk]) -> float:
    """Calculate normalized risk score (0-1 scale)"""
    if not risks:
        return 0.0
    
    total_score = sum(risk.calculate_risk_score() for risk in risks)
    max_possible_score = len(risks) * 5.0  # Max risk level is 5
    return min(1.0, total_score / max_possible_score)


def get_project_health_status(project: Project) -> str:
    """Get overall project health status"""
    risk_score = project.calculate_overall_risk_score()
    progress = project.get_progress_percentage()
    is_overdue = project.is_overdue()
    
    if is_overdue and risk_score > 3.5:
        return "critical"
    elif risk_score > 4.0 or (is_overdue and progress < 50):
        return "poor"
    elif risk_score > 3.0 or progress < 25:
        return "fair"
    elif risk_score < 2.0 and progress > 75:
        return "excellent"
    else:
        return "good"


def filter_projects_by_criteria(
    projects: List[Project],
    sector: Optional[TelecomSectorType] = None,
    status: Optional[ProjectStatus] = None,
    min_budget: Optional[float] = None,
    max_budget: Optional[float] = None,
    emirates: Optional[List[str]] = None,
    risk_level: Optional[str] = None
) -> List[Project]:
    """Filter projects based on multiple criteria"""
    
    filtered = projects
    
    if sector:
        filtered = [p for p in filtered if p.sector == sector]
    
    if status:
        filtered = [p for p in filtered if p.status == status]
    
    if min_budget is not None:
        filtered = [p for p in filtered if p.budget_aed >= min_budget]
    
    if max_budget is not None:
        filtered = [p for p in filtered if p.budget_aed <= max_budget]
    
    if emirates:
        filtered = [p for p in filtered if any(e in p.emirates for e in emirates)]
    
    if risk_level:
        threshold_map = {
            "low": 2.0,
            "medium": 3.0,
            "high": 4.0,
            "very_high": 5.0
        }
        threshold = threshold_map.get(risk_level, 0)
        filtered = [p for p in filtered if p.calculate_overall_risk_score() >= threshold]
    
    return filtered


def generate_project_summary(project: Project) -> Dict[str, Any]:
    """Generate comprehensive project summary"""
    return {
        "basic_info": {
            "project_id": project.project_id,
            "name": project.name,
            "sector": project.sector.value,
            "status": project.status.value,
            "budget_formatted": format_currency_aed(project.budget_aed),
            "duration_days": calculate_project_duration(project.start_date, project.planned_end_date)
        },
        "progress": {
            "percentage": project.get_progress_percentage(),
            "is_overdue": project.is_overdue(),
            "health_status": get_project_health_status(project)
        },
        "risk_analysis": {
            "total_risks": len(project.risks),
            "high_risks": len(project.get_high_risks()),
            "overall_score": project.calculate_overall_risk_score(),
            "risk_distribution": _get_risk_category_distribution(project.risks)
        },
        "location": {
            "emirates": project.emirates,
            "coverage_area": project.coverage_area,
            "target_population": project.target_population
        },
        "metadata": {
            "complexity_score": project.complexity_score,
            "strategic_importance": project.strategic_importance,
            "technology_maturity": project.technology_maturity,
            "created_date": project.created_date.isoformat(),
            "last_updated": project.last_updated.isoformat()
        }
    }


def _get_risk_category_distribution(risks: List[ProjectRisk]) -> Dict[str, int]:
    """Get distribution of risks by category"""
    distribution = {}
    for risk in risks:
        category = risk.category.value
        distribution[category] = distribution.get(category, 0) + 1
    return distribution


def export_projects_to_csv(projects: List[Project], filepath: str) -> None:
    """Export projects to CSV file"""
    fieldnames = [
        'project_id', 'name', 'sector', 'status', 'budget_aed', 'start_date',
        'planned_end_date', 'emirates', 'complexity_score', 'strategic_importance',
        'total_risks', 'overall_risk_score', 'progress_percentage', 'health_status'
    ]
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for project in projects:
            writer.writerow({
                'project_id': project.project_id,
                'name': project.name,
                'sector': project.sector.value,
                'status': project.status.value,
                'budget_aed': project.budget_aed,
                'start_date': project.start_date.isoformat(),
                'planned_end_date': project.planned_end_date.isoformat(),
                'emirates': ','.join(project.emirates),
                'complexity_score': project.complexity_score,
                'strategic_importance': project.strategic_importance,
                'total_risks': len(project.risks),
                'overall_risk_score': project.calculate_overall_risk_score(),
                'progress_percentage': project.get_progress_percentage(),
                'health_status': get_project_health_status(project)
            })


def import_projects_from_csv(filepath: str) -> List[Dict[str, Any]]:
    """Import project data from CSV file"""
    projects_data = []
    
    with open(filepath, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Convert data types
            row['budget_aed'] = float(row['budget_aed'])
            row['start_date'] = datetime.fromisoformat(row['start_date']).date()
            row['planned_end_date'] = datetime.fromisoformat(row['planned_end_date']).date()
            row['emirates'] = row['emirates'].split(',') if row['emirates'] else []
            row['complexity_score'] = float(row['complexity_score'])
            row['strategic_importance'] = float(row['strategic_importance'])
            
            projects_data.append(row)
    
    return projects_data


def calculate_portfolio_metrics(projects: List[Project]) -> Dict[str, Any]:
    """Calculate comprehensive portfolio metrics"""
    if not projects:
        return {}
    
    total_budget = sum(p.budget_aed for p in projects)
    risk_scores = [p.calculate_overall_risk_score() for p in projects]
    progress_scores = [p.get_progress_percentage() for p in projects]
    
    # Sector distribution
    sector_counts = {}
    for project in projects:
        sector = project.sector.value
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    
    # Status distribution
    status_counts = {}
    for project in projects:
        status = project.status.value
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Emirates coverage
    all_emirates = set()
    for project in projects:
        all_emirates.update(project.emirates)
    
    return {
        "overview": {
            "total_projects": len(projects),
            "total_budget_aed": total_budget,
            "average_budget_aed": total_budget / len(projects),
            "total_budget_formatted": format_currency_aed(total_budget)
        },
        "risk_metrics": {
            "average_risk_score": np.mean(risk_scores),
            "max_risk_score": np.max(risk_scores),
            "high_risk_projects": sum(1 for score in risk_scores if score >= 4.0),
            "risk_distribution": {
                "low": sum(1 for score in risk_scores if score < 2.0),
                "medium": sum(1 for score in risk_scores if 2.0 <= score < 4.0),
                "high": sum(1 for score in risk_scores if score >= 4.0)
            }
        },
        "progress_metrics": {
            "average_progress": np.mean(progress_scores),
            "completed_projects": sum(1 for p in projects if p.status == ProjectStatus.COMPLETED),
            "overdue_projects": sum(1 for p in projects if p.is_overdue()),
            "on_track_projects": sum(1 for p in projects 
                                   if not p.is_overdue() and p.status == ProjectStatus.IN_PROGRESS)
        },
        "distribution": {
            "by_sector": sector_counts,
            "by_status": status_counts,
            "emirates_coverage": list(all_emirates),
            "geographic_spread": len(all_emirates)
        },
        "health_summary": {
            status: sum(1 for p in projects if get_project_health_status(p) == status)
            for status in ["excellent", "good", "fair", "poor", "critical"]
        }
    }


def generate_risk_heatmap_data(projects: List[Project]) -> Dict[str, Any]:
    """Generate data for risk heatmap visualization"""
    heatmap_data = {
        "sectors": [],
        "risk_categories": [category.value for category in RiskCategory],
        "matrix": []
    }
    
    # Group projects by sector
    sector_projects = {}
    for project in projects:
        sector = project.sector
        if sector not in sector_projects:
            sector_projects[sector] = []
            heatmap_data["sectors"].append(sector.value)
        sector_projects[sector].append(project)
    
    # Calculate risk intensity for each sector-category combination
    for sector in sector_projects:
        row = []
        for category in RiskCategory:
            # Calculate average risk score for this sector-category combination
            relevant_risks = []
            for project in sector_projects[sector]:
                category_risks = project.get_risks_by_category(category)
                relevant_risks.extend([risk.calculate_risk_score() for risk in category_risks])
            
            avg_score = np.mean(relevant_risks) if relevant_risks else 0.0
            row.append(avg_score)
        heatmap_data["matrix"].append(row)
    
    return heatmap_data


def validate_project_data(project_dict: Dict[str, Any]) -> List[str]:
    """Validate project data and return list of validation errors"""
    errors = []
    
    # Required fields
    required_fields = ['project_id', 'name', 'sector', 'budget_aed', 'start_date', 'planned_end_date']
    for field in required_fields:
        if field not in project_dict or not project_dict[field]:
            errors.append(f"Missing required field: {field}")
    
    # Data type validations
    if 'budget_aed' in project_dict:
        try:
            budget = float(project_dict['budget_aed'])
            if budget <= 0:
                errors.append("Budget must be positive")
        except (ValueError, TypeError):
            errors.append("Budget must be a valid number")
    
    # Date validations
    if 'start_date' in project_dict and 'planned_end_date' in project_dict:
        try:
            start = datetime.fromisoformat(str(project_dict['start_date'])).date()
            end = datetime.fromisoformat(str(project_dict['planned_end_date'])).date()
            if start >= end:
                errors.append("Start date must be before planned end date")
        except ValueError:
            errors.append("Invalid date format")
    
    # Sector validation
    if 'sector' in project_dict:
        valid_sectors = [s.value for s in TelecomSectorType]
        if project_dict['sector'] not in valid_sectors:
            errors.append(f"Invalid sector. Must be one of: {valid_sectors}")
    
    return errors


def create_sample_dataset(num_projects: int = 20) -> List[Project]:
    """Create sample dataset for testing and demonstration"""
    sample_projects = []
    
    sectors = list(TelecomSectorType)
    emirates_list = ["Dubai", "Abu Dhabi", "Sharjah", "Ajman"]
    
    for i in range(num_projects):
        sector = np.random.choice(sectors)
        emirates = np.random.choice(emirates_list, size=np.random.randint(1, 3), replace=False).tolist()
        
        project = Project(
            project_id=f"UAE_TEL_{i+1:03d}",
            name=f"Project {i+1} - {sector.value.title()}",
            description=f"Sample {sector.value} project for demonstration",
            sector=sector,
            status=np.random.choice(list(ProjectStatus)),
            budget_aed=np.random.uniform(5_000_000, 200_000_000),
            start_date=date.today() - timedelta(days=np.random.randint(30, 365)),
            planned_end_date=date.today() + timedelta(days=np.random.randint(30, 730)),
            project_manager=f"Manager {i+1}",
            sponsor=f"Sponsor {i+1}",
            emirates=emirates,
            complexity_score=np.random.uniform(1, 10),
            strategic_importance=np.random.uniform(1, 10),
            technology_maturity=np.random.choice(["emerging", "developing", "proven", "legacy"])
        )
        
        # Add random risks
        num_risks = np.random.randint(1, 8)
        for j in range(num_risks):
            risk = ProjectRisk(
                risk_id=f"RISK_{i+1}_{j+1}",
                title=f"Risk {j+1} for Project {i+1}",
                description=f"Sample risk description",
                category=np.random.choice(list(RiskCategory)),
                probability=np.random.uniform(0.1, 0.9),
                impact=np.random.choice(list(RiskLevel)),
                risk_level=np.random.choice(list(RiskLevel))
            )
            project.add_risk(risk)
        
        sample_projects.append(project)
    
    return sample_projects