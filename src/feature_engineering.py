"""
Feature Engineering Module for UAE Telecom Risk Recommender System

This module provides functionality to encode categorical features and
extract sentiment from risk descriptions for machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import re
import logging
from typing import Dict, List, Tuple, Optional
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class FeatureEngineeringPipeline:
    """Pipeline for feature engineering on telecom risk data."""
    
    def __init__(self):
        """Initialize the feature engineering pipeline."""
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: StandardScaler = StandardScaler()
        self.tfidf_vectorizer: TfidfVectorizer = TfidfVectorizer(
            max_features=100, 
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.engineered_features: Optional[pd.DataFrame] = None
        
    def clean_text(self, text: str) -> str:
        """
        Clean text data for sentiment analysis.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or text == 'nan':
            return ""
            
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_sentiment_features(self, data: pd.DataFrame, text_column: str = 'Risk_Description') -> pd.DataFrame:
        """
        Extract sentiment features from text descriptions.
        
        Args:
            data (pd.DataFrame): Input data
            text_column (str): Column containing text to analyze
            
        Returns:
            pd.DataFrame: Data with added sentiment features
        """
        logger.info(f"Extracting sentiment features from {text_column}")
        
        if text_column not in data.columns:
            logger.warning(f"Column {text_column} not found. Skipping sentiment analysis.")
            return data
        
        data_copy = data.copy()
        
        # Clean text
        data_copy['cleaned_description'] = data_copy[text_column].apply(self.clean_text)
        
        # Extract sentiment using TextBlob
        sentiments = []
        polarities = []
        subjectivities = []
        
        for text in data_copy['cleaned_description']:
            if text:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                # Categorize sentiment
                if polarity > 0.1:
                    sentiment = 'positive'
                elif polarity < -0.1:
                    sentiment = 'negative'
                else:
                    sentiment = 'neutral'
                    
                sentiments.append(sentiment)
                polarities.append(polarity)
                subjectivities.append(subjectivity)
            else:
                sentiments.append('neutral')
                polarities.append(0.0)
                subjectivities.append(0.0)
        
        data_copy['sentiment_category'] = sentiments
        data_copy['sentiment_polarity'] = polarities
        data_copy['sentiment_subjectivity'] = subjectivities
        
        # Create TF-IDF features for text
        try:
            tfidf_features = self.tfidf_vectorizer.fit_transform(data_copy['cleaned_description']).toarray()
            tfidf_df = pd.DataFrame(
                tfidf_features, 
                columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])],
                index=data_copy.index
            )
            data_copy = pd.concat([data_copy, tfidf_df], axis=1)
            logger.info(f"Added {tfidf_features.shape[1]} TF-IDF features")
        except Exception as e:
            logger.warning(f"Could not create TF-IDF features: {str(e)}")
        
        return data_copy
    
    def encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using label encoding and one-hot encoding.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with encoded categorical features
        """
        logger.info("Encoding categorical features")
        
        data_copy = data.copy()
        
        # Define categorical columns for different encoding strategies
        label_encode_columns = ['Risk_Owner', 'Stakeholder_Group', 'Department_Unit', 'Project_Name']
        onehot_encode_columns = ['Risk_Category', 'Sub_Category', 'Sector', 'Project_Phase', 
                                'Risk_Status', 'Probability_Rating', 'Impact_Rating']
        
        # Label encoding for high cardinality categorical features
        for col in label_encode_columns:
            if col in data_copy.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                
                # Handle missing values
                data_copy[col] = data_copy[col].fillna('Unknown')
                data_copy[f'{col}_encoded'] = self.label_encoders[col].fit_transform(data_copy[col])
                logger.info(f"Label encoded {col}: {len(self.label_encoders[col].classes_)} unique values")
        
        # One-hot encoding for lower cardinality categorical features
        for col in onehot_encode_columns:
            if col in data_copy.columns:
                # Handle missing values
                data_copy[col] = data_copy[col].fillna('Unknown')
                
                # Create dummy variables
                dummies = pd.get_dummies(data_copy[col], prefix=f'{col}')
                data_copy = pd.concat([data_copy, dummies], axis=1)
                logger.info(f"One-hot encoded {col}: {len(dummies.columns)} categories")
        
        return data_copy
    
    def create_risk_score_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features based on risk scores and ratings.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            pd.DataFrame: Data with additional risk score features
        """
        logger.info("Creating risk score features")
        
        data_copy = data.copy()
        
        # Convert ratings to numeric if they're categorical
        rating_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
        
        for col in ['Probability_Rating', 'Impact_Rating']:
            if col in data_copy.columns:
                data_copy[f'{col}_numeric'] = data_copy[col].map(rating_mapping)
                data_copy[f'{col}_numeric'] = data_copy[f'{col}_numeric'].fillna(2)  # Default to medium
        
        # Create combined risk features
        if 'Probability_Rating_numeric' in data_copy.columns and 'Impact_Rating_numeric' in data_copy.columns:
            data_copy['risk_score_calculated'] = data_copy['Probability_Rating_numeric'] * data_copy['Impact_Rating_numeric']
            
        # Risk score validation (if original risk score exists)
        if 'Risk_Score' in data_copy.columns:
            data_copy['Risk_Score'] = pd.to_numeric(data_copy['Risk_Score'], errors='coerce')
            data_copy['Risk_Score'] = data_copy['Risk_Score'].fillna(data_copy['risk_score_calculated'])
            
            # Risk score categories
            data_copy['risk_level'] = pd.cut(
                data_copy['Risk_Score'], 
                bins=[0, 3, 6, 9], 
                labels=['Low', 'Medium', 'High'], 
                include_lowest=True
            )
        
        # Create project phase numeric encoding
        phase_mapping = {'Initiating': 1, 'Planning': 2, 'Executing': 3, 'Monitoring': 4, 'Closing': 5}
        if 'Project_Phase' in data_copy.columns:
            data_copy['project_phase_numeric'] = data_copy['Project_Phase'].map(phase_mapping)
            data_copy['project_phase_numeric'] = data_copy['project_phase_numeric'].fillna(2)
        
        return data_copy
    
    def scale_numerical_features(self, data: pd.DataFrame, 
                                numerical_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            data (pd.DataFrame): Input data
            numerical_columns (Optional[List[str]]): Columns to scale. Auto-detect if None.
            
        Returns:
            pd.DataFrame: Data with scaled numerical features
        """
        logger.info("Scaling numerical features")
        
        data_copy = data.copy()
        
        if numerical_columns is None:
            # Auto-detect numerical columns
            numerical_columns = data_copy.select_dtypes(include=[np.number]).columns.tolist()
            
            # Exclude ID columns and encoded categorical columns
            exclude_patterns = ['_ID', '_id', 'Risk_ID']
            numerical_columns = [col for col in numerical_columns 
                               if not any(pattern in col for pattern in exclude_patterns)]
        
        # Filter for existing columns
        numerical_columns = [col for col in numerical_columns if col in data_copy.columns]
        
        if numerical_columns:
            scaled_features = self.scaler.fit_transform(data_copy[numerical_columns])
            
            # Create scaled feature names
            scaled_columns = [f'{col}_scaled' for col in numerical_columns]
            
            # Add scaled features to dataframe
            scaled_df = pd.DataFrame(scaled_features, columns=scaled_columns, index=data_copy.index)
            data_copy = pd.concat([data_copy, scaled_df], axis=1)
            
            logger.info(f"Scaled {len(numerical_columns)} numerical features")
        
        return data_copy
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            data (pd.DataFrame): Raw input data
            
        Returns:
            pd.DataFrame: Data with engineered features
        """
        logger.info("Starting complete feature engineering pipeline")
        
        # Step 1: Extract sentiment features
        data = self.extract_sentiment_features(data)
        
        # Step 2: Create risk score features
        data = self.create_risk_score_features(data)
        
        # Step 3: Encode categorical features
        data = self.encode_categorical_features(data)
        
        # Step 4: Scale numerical features
        data = self.scale_numerical_features(data)
        
        self.engineered_features = data
        logger.info(f"Feature engineering completed. Final dataset shape: {data.shape}")
        
        return data
    
    def get_feature_importance_data(self, data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Get information about engineered features for model training.
        
        Args:
            data (Optional[pd.DataFrame]): Engineered data. Uses self.engineered_features if None.
            
        Returns:
            Dict: Feature information
        """
        if data is None:
            data = self.engineered_features
            
        if data is None:
            raise ValueError("No engineered features available")
        
        # Identify different types of features
        sentiment_features = [col for col in data.columns if 'sentiment' in col.lower()]
        tfidf_features = [col for col in data.columns if 'tfidf' in col.lower()]
        encoded_features = [col for col in data.columns if '_encoded' in col or any(cat in col for cat in ['Risk_Category_', 'Sector_', 'Project_Phase_'])]
        scaled_features = [col for col in data.columns if '_scaled' in col]
        
        feature_info = {
            'total_features': len(data.columns),
            'sentiment_features': len(sentiment_features),
            'tfidf_features': len(tfidf_features),
            'encoded_categorical_features': len(encoded_features),
            'scaled_numerical_features': len(scaled_features),
            'feature_names': {
                'sentiment': sentiment_features,
                'tfidf': tfidf_features,
                'encoded_categorical': encoded_features,
                'scaled_numerical': scaled_features
            }
        }
        
        return feature_info


def main():
    """Main function to demonstrate feature engineering pipeline."""
    # This would typically be called after data ingestion
    print("Feature Engineering Pipeline - Ready for integration with data ingestion")
    
    # Example of initializing the pipeline
    pipeline = FeatureEngineeringPipeline()
    logger.info("Feature engineering pipeline initialized")


if __name__ == "__main__":
    main()