#!/usr/bin/env python3
"""
Fix missing data in network_infrastructure_risk_register.csv

This script fills missing values in RMS (Risk Management System) step columns 
based on the Primary_RMS_Step value and logical relationships between different
risk management process steps.
"""

import pandas as pd
import numpy as np
import os

def fix_missing_data():
    """
    Fix missing data in the risk register CSV file.
    
    Logic:
    - Each Primary_RMS_Step should have its corresponding column marked as active (not empty/?)
    - Other RMS step columns should be marked as empty (np.nan) if they don't apply to that step
    - The sequence follows: Identification -> Analysis (Qualitative/Quantitative) -> Response Planning -> Monitoring -> Controlling -> Documentation
    """
    
    # Load the CSV file
    file_path = '/home/runner/work/uae_telecom_recommender_systems/uae_telecom_recommender_systems/network_infrastructure_risk_register.csv'
    df = pd.read_csv(file_path)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Total missing values ('?'): {(df == '?').sum().sum()}")
    
    # Create a copy for modifications
    df_fixed = df.copy()
    
    # Define RMS step columns (excluding RMS_Step_Label which appears to be different)
    rms_step_columns = [
        'RMS_Step_Identification',
        'RMS_Step_Analysis_Qualitative', 
        'RMS_Step_Analysis_Quantitative',
        'RMS_Step_Response_Planning',
        'RMS_Step_Monitoring',
        'RMS_Step_Controlling',
        'RMS_Step_Documentation'
    ]
    
    # Define the mapping logic
    # For each Primary_RMS_Step, define which columns should be active (marked) vs inactive (empty)
    rms_step_mapping = {
        'Risk Identification': {
            'active': ['RMS_Step_Identification'],
            'completed': ['RMS_Step_Analysis_Qualitative', 'RMS_Step_Analysis_Quantitative', 'RMS_Step_Response_Planning', 'RMS_Step_Controlling', 'RMS_Step_Documentation'],
            'inactive': ['RMS_Step_Monitoring']
        },
        'Identification': {  # Alternative naming
            'active': ['RMS_Step_Identification'],
            'completed': ['RMS_Step_Analysis_Qualitative'],
            'inactive': ['RMS_Step_Analysis_Quantitative', 'RMS_Step_Response_Planning', 'RMS_Step_Monitoring', 'RMS_Step_Controlling', 'RMS_Step_Documentation']
        },
        'Analysis – Qualitative': {
            'active': ['RMS_Step_Analysis_Qualitative'],
            'completed': ['RMS_Step_Analysis_Quantitative', 'RMS_Step_Response_Planning', 'RMS_Step_Controlling', 'RMS_Step_Documentation'],
            'inactive': ['RMS_Step_Identification', 'RMS_Step_Monitoring']
        },
        'Analysis – Quantitative': {
            'active': ['RMS_Step_Analysis_Quantitative'],
            'completed': ['RMS_Step_Response_Planning', 'RMS_Step_Controlling', 'RMS_Step_Documentation'],
            'inactive': ['RMS_Step_Identification', 'RMS_Step_Analysis_Qualitative', 'RMS_Step_Monitoring']
        },
        'Response Planning': {
            'active': ['RMS_Step_Response_Planning'],
            'completed': ['RMS_Step_Controlling', 'RMS_Step_Documentation'],
            'inactive': ['RMS_Step_Identification', 'RMS_Step_Analysis_Qualitative', 'RMS_Step_Analysis_Quantitative', 'RMS_Step_Monitoring']
        },
        'Monitoring': {
            'active': ['RMS_Step_Monitoring'],
            'completed': ['RMS_Step_Identification', 'RMS_Step_Analysis_Qualitative', 'RMS_Step_Documentation'],
            'inactive': ['RMS_Step_Analysis_Quantitative', 'RMS_Step_Response_Planning', 'RMS_Step_Controlling']
        },
        'Controlling': {
            'active': ['RMS_Step_Controlling'],
            'completed': ['RMS_Step_Identification', 'RMS_Step_Analysis_Qualitative', 'RMS_Step_Documentation'],
            'inactive': ['RMS_Step_Analysis_Quantitative', 'RMS_Step_Response_Planning', 'RMS_Step_Monitoring']
        },
        'Documentation': {
            'active': ['RMS_Step_Documentation'],
            'completed': ['RMS_Step_Identification', 'RMS_Step_Analysis_Qualitative', 'RMS_Step_Analysis_Quantitative', 'RMS_Step_Response_Planning', 'RMS_Step_Controlling'],
            'inactive': ['RMS_Step_Monitoring']
        }
    }
    
    # Apply the fixing logic
    fixed_count = 0
    
    for primary_step in df_fixed['Primary_RMS_Step'].unique():
        if primary_step in rms_step_mapping:
            mask = df_fixed['Primary_RMS_Step'] == primary_step
            mapping = rms_step_mapping[primary_step]
            
            # Set active columns (current step) to empty string (indicating completion of that step)
            for col in mapping['active']:
                if col in df_fixed.columns:
                    # Replace '?' with empty string for active steps
                    before_count = (df_fixed.loc[mask, col] == '?').sum()
                    df_fixed.loc[mask, col] = df_fixed.loc[mask, col].replace('?', '')
                    fixed_count += before_count
            
            # Set completed columns to empty string (steps that have been completed)
            for col in mapping['completed']:
                if col in df_fixed.columns:
                    before_count = (df_fixed.loc[mask, col] == '?').sum()
                    df_fixed.loc[mask, col] = df_fixed.loc[mask, col].replace('?', '')
                    fixed_count += before_count
            
            # Set inactive columns to empty string (steps not yet reached or not applicable)
            for col in mapping['inactive']:
                if col in df_fixed.columns:
                    before_count = (df_fixed.loc[mask, col] == '?').sum()
                    df_fixed.loc[mask, col] = df_fixed.loc[mask, col].replace('?', '')
                    fixed_count += before_count
    
    # Also fix RMS_Step_Label missing values
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
    
    # Fix RMS_Step_Label
    for primary_step, label in rms_step_label_mapping.items():
        mask = (df_fixed['Primary_RMS_Step'] == primary_step) & (df_fixed['RMS_Step_Label'] == '?')
        before_count = mask.sum()
        df_fixed.loc[mask, 'RMS_Step_Label'] = label
        fixed_count += before_count
    
    print(f"\nFixed {fixed_count} missing values")
    print(f"Remaining missing values ('?'): {(df_fixed == '?').sum().sum()}")
    
    # Save the fixed CSV
    output_path = '/home/runner/work/uae_telecom_recommender_systems/uae_telecom_recommender_systems/network_infrastructure_risk_register_fixed.csv'
    df_fixed.to_csv(output_path, index=False)
    
    # Replace the original file
    df_fixed.to_csv(file_path, index=False)
    
    print(f"\nFixed data saved to:")
    print(f"- Original file updated: {file_path}")
    print(f"- Backup copy created: {output_path}")
    
    # Validate the results
    print("\nValidation Summary:")
    print(f"Total records: {len(df_fixed)}")
    print(f"Total missing values remaining: {(df_fixed == '?').sum().sum()}")
    
    # Show summary by Primary_RMS_Step
    print("\nSummary by Primary RMS Step:")
    for step in sorted(df_fixed['Primary_RMS_Step'].unique()):
        count = (df_fixed['Primary_RMS_Step'] == step).sum()
        print(f"  {step}: {count} records")
    
    return df_fixed

if __name__ == "__main__":
    fixed_df = fix_missing_data()
    print("\nData cleaning completed successfully!")