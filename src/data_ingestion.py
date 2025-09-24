"""
Data Ingestion Module for UAE Telecom Risk Recommender System

This module provides functionality to load and perform basic cleaning
on the telecom risk register data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """Pipeline for loading and basic cleaning of telecom risk data."""
    
    def __init__(self, data_path: str):
        """
        Initialize the data ingestion pipeline.
        
        Args:
            data_path (str): Path to the CSV data file
        """
        self.data_path = Path(data_path)
        self.raw_data: Optional[pd.DataFrame] = None
        self.cleaned_data: Optional[pd.DataFrame] = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the CSV data file.
        
        Returns:
            pd.DataFrame: Raw loaded data
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
            pd.errors.EmptyDataError: If the file is empty
        """
        try:
            logger.info(f"Loading data from {self.data_path}")
            self.raw_data = pd.read_csv(self.data_path)
            logger.info(f"Successfully loaded {len(self.raw_data)} rows and {len(self.raw_data.columns)} columns")
            return self.raw_data
            
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_path}")
            raise
        except pd.errors.EmptyDataError:
            logger.error(f"Data file is empty: {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def basic_cleaning(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Perform basic data cleaning operations.
        
        Args:
            data (Optional[pd.DataFrame]): Data to clean. Uses self.raw_data if None.
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        if data is None:
            data = self.raw_data
            
        if data is None:
            raise ValueError("No data available. Please load data first.")
            
        logger.info("Starting basic data cleaning")
        cleaned_data = data.copy()
        
        # Basic cleaning operations
        initial_rows = len(cleaned_data)
        
        # Remove completely empty rows
        cleaned_data = cleaned_data.dropna(how='all')
        logger.info(f"Removed {initial_rows - len(cleaned_data)} completely empty rows")
        
        # Clean text fields - strip whitespace
        text_columns = ['Risk_Title', 'Risk_Description', 'Risk_Category', 'Sub_Category', 
                       'Sector', 'Department_Unit', 'Project_Name', 'Risk_Owner', 
                       'Stakeholder_Group', 'Qualitative_Notes', 'Mitigation_Strategy', 
                       'Contingency_Plan', 'Risk_Status']
        
        for col in text_columns:
            if col in cleaned_data.columns:
                cleaned_data[col] = cleaned_data[col].astype(str).str.strip()
                cleaned_data[col] = cleaned_data[col].replace('nan', np.nan)
        
        # Clean categorical columns
        if 'Risk_Status' in cleaned_data.columns:
            cleaned_data['Risk_Status'] = cleaned_data['Risk_Status'].str.title()
            
        if 'Probability_Rating' in cleaned_data.columns:
            cleaned_data['Probability_Rating'] = cleaned_data['Probability_Rating'].str.title()
            
        if 'Impact_Rating' in cleaned_data.columns:
            cleaned_data['Impact_Rating'] = cleaned_data['Impact_Rating'].str.title()
        
        # Convert Risk_Score to numeric
        if 'Risk_Score' in cleaned_data.columns:
            cleaned_data['Risk_Score'] = pd.to_numeric(cleaned_data['Risk_Score'], errors='coerce')
        
        # Handle date columns
        date_columns = ['Date_Identified', 'Review_Date']
        for col in date_columns:
            if col in cleaned_data.columns:
                # Convert week-based dates to more standard format
                cleaned_data[col] = cleaned_data[col].astype(str)
        
        self.cleaned_data = cleaned_data
        logger.info(f"Data cleaning completed. Final dataset: {len(cleaned_data)} rows")
        
        return cleaned_data
    
    def get_data_summary(self, data: Optional[pd.DataFrame] = None) -> dict:
        """
        Get summary statistics and information about the data.
        
        Args:
            data (Optional[pd.DataFrame]): Data to summarize. Uses cleaned_data if None.
            
        Returns:
            dict: Summary information
        """
        if data is None:
            data = self.cleaned_data if self.cleaned_data is not None else self.raw_data
            
        if data is None:
            raise ValueError("No data available")
        
        summary = {
            'total_rows': len(data),
            'total_columns': len(data.columns),
            'missing_values': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.to_dict(),
            'unique_risk_categories': data['Risk_Category'].nunique() if 'Risk_Category' in data.columns else 0,
            'unique_sectors': data['Sector'].nunique() if 'Sector' in data.columns else 0,
            'risk_status_counts': data['Risk_Status'].value_counts().to_dict() if 'Risk_Status' in data.columns else {},
            'probability_rating_counts': data['Probability_Rating'].value_counts().to_dict() if 'Probability_Rating' in data.columns else {},
            'impact_rating_counts': data['Impact_Rating'].value_counts().to_dict() if 'Impact_Rating' in data.columns else {}
        }
        
        return summary
    
    def save_cleaned_data(self, output_path: str) -> None:
        """
        Save cleaned data to CSV file.
        
        Args:
            output_path (str): Path to save the cleaned data
        """
        if self.cleaned_data is None:
            raise ValueError("No cleaned data available. Please run basic_cleaning first.")
            
        self.cleaned_data.to_csv(output_path, index=False)
        logger.info(f"Cleaned data saved to {output_path}")


def main():
    """Main function to demonstrate data ingestion pipeline."""
    # Example usage
    data_path = "network_infrastructure_risk_register.csv"
    
    # Initialize pipeline
    pipeline = DataIngestionPipeline(data_path)
    
    try:
        # Load and clean data
        raw_data = pipeline.load_data()
        cleaned_data = pipeline.basic_cleaning()
        
        # Get summary
        summary = pipeline.get_data_summary()
        
        print("Data Summary:")
        print(f"Total rows: {summary['total_rows']}")
        print(f"Total columns: {summary['total_columns']}")
        print(f"Unique risk categories: {summary['unique_risk_categories']}")
        print(f"Unique sectors: {summary['unique_sectors']}")
        print("\nRisk status distribution:")
        for status, count in summary['risk_status_counts'].items():
            print(f"  {status}: {count}")
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")


if __name__ == "__main__":
    main()