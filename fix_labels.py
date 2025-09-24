#!/usr/bin/env python3
"""
Fix RMS_Step_Label missing values based on Primary_RMS_Step
"""

import pandas as pd

def fix_labels():
    # Load the CSV file
    file_path = '/home/runner/work/uae_telecom_recommender_systems/uae_telecom_recommender_systems/network_infrastructure_risk_register.csv'
    df = pd.read_csv(file_path)
    
    print(f"Before fixing labels - Missing RMS_Step_Label: {df['RMS_Step_Label'].isna().sum()}")
    
    # Define the mapping for labels
    rms_step_label_mapping = {
        'Risk Identification': 'Risk Identification',
        'Identification': 'Risk Identification', 
        'Analysis – Qualitative': 'Qualitative Analysis',
        'Analysis – Quantitative': 'Quantitative Analysis',
        'Response Planning': 'Response Planning',
        'Monitoring': 'Risk Monitoring',
        'Controlling': 'Risk Controlling',
        'Documentation': 'Risk Documentation'
    }
    
    # Fix missing RMS_Step_Label values
    for primary_step, label in rms_step_label_mapping.items():
        mask = (df['Primary_RMS_Step'] == primary_step) & (df['RMS_Step_Label'].isna())
        count = mask.sum()
        if count > 0:
            df.loc[mask, 'RMS_Step_Label'] = label
            print(f"Fixed {count} records for {primary_step} -> {label}")
    
    print(f"After fixing labels - Missing RMS_Step_Label: {df['RMS_Step_Label'].isna().sum()}")
    
    # Save the updated file
    df.to_csv(file_path, index=False)
    print(f"Updated file saved: {file_path}")
    
    return df

if __name__ == "__main__":
    fix_labels()