"""
Configuration settings for the UAE Telecom RMS Pipeline
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR
OUTPUT_DIR = BASE_DIR / "output"

# Input files
RISK_REGISTER_CSV = DATA_DIR / "network_infrastructure_risk_register.csv"

# Output files
RMS_OUTPUT_FILE = OUTPUT_DIR / "rms_processed_risks.csv"
ML_DATASET_FILE = OUTPUT_DIR / "ml_ready_dataset.csv"
ANALYSIS_REPORT_FILE = OUTPUT_DIR / "rms_analysis_report.html"

# RMS Pipeline Configuration
RMS_STEPS = [
    "Risk Identification",
    "Analysis – Qualitative", 
    "Analysis – Quantitative",
    "Response Planning",
    "Monitoring",
    "Controlling",
    "Documentation"
]

# Risk scoring parameters
PROBABILITY_WEIGHTS = {"Low": 1, "Medium": 3, "High": 5}
IMPACT_WEIGHTS = {"Low": 1, "Medium": 3, "High": 5}

# ML Model Configuration
ML_TARGET_VARIABLES = ["Risk Score", "Probability Rating", "Impact Rating"]
ML_FEATURE_COLUMNS = [
    "Risk Category", "Sub-Category", "Sector", "Project Phase", 
    "Process Group", "Status", "Primary RMS Step"
]

# Risk categories and priorities
HIGH_PRIORITY_CATEGORIES = ["Construction", "Requirements", "Technology"]
CRITICAL_SECTORS = ["Network Infrastructure", "Fiber Infrastructure"]

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)