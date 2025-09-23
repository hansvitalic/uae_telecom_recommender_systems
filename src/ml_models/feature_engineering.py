"""
Feature Engineering Module
Prepares features for machine learning models from risk data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
import config

class FeatureEngineer:
    """
    Handles feature engineering for ML risk prediction models
    """
    
    def __init__(self, risks_df: pd.DataFrame = None):
        """Initialize with risk data"""
        self.risks_df = risks_df
        if self.risks_df is None:
            self.risks_df = pd.read_csv(config.RISK_REGISTER_CSV)
        
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def prepare_base_features(self) -> pd.DataFrame:
        """Prepare base features from raw risk data"""
        features_df = self.risks_df.copy()
        
        # Create numerical mappings
        features_df['Probability_Numeric'] = features_df['Probability Rating'].map(config.PROBABILITY_WEIGHTS)
        features_df['Impact_Numeric'] = features_df['Impact Rating'].map(config.IMPACT_WEIGHTS)
        
        # Create binary features
        features_df['Has_Mitigation'] = (features_df['Mitigation Strategy'].notna()).astype(int)
        features_df['Has_Contingency'] = (features_df['Contingency Plan'].notna()).astype(int)
        features_df['Is_Open'] = (features_df['Status'] == 'Open').astype(int)
        
        # Create risk level categories
        features_df['Risk_Level'] = self._categorize_risk_level(features_df)
        
        # Create composite features
        features_df['Risk_Urgency'] = features_df['Probability_Numeric'] * features_df['Impact_Numeric']
        features_df['Mitigation_Score'] = self._calculate_mitigation_score(features_df)
        
        return features_df
    
    def _categorize_risk_level(self, df: pd.DataFrame) -> pd.Series:
        """Categorize risks into levels based on probability and impact"""
        conditions = [
            (df['Probability Rating'] == 'High') & (df['Impact Rating'] == 'High'),
            ((df['Probability Rating'] == 'High') & (df['Impact Rating'] == 'Medium')) |
            ((df['Probability Rating'] == 'Medium') & (df['Impact Rating'] == 'High')),
            (df['Probability Rating'] == 'Medium') & (df['Impact Rating'] == 'Medium'),
        ]
        choices = ['Critical', 'High', 'Medium']
        
        return pd.Series(np.select(conditions, choices, default='Low'), index=df.index)
    
    def _calculate_mitigation_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate mitigation effectiveness score"""
        # Simple scoring based on presence and quality of mitigation
        scores = np.zeros(len(df))
        
        for i, row in df.iterrows():
            score = 0
            
            # Base score for having mitigation
            if pd.notna(row['Mitigation Strategy']):
                score += 2
                
                # Bonus for strategy length (proxy for detail)
                strategy_length = len(str(row['Mitigation Strategy']))
                if strategy_length > 50:
                    score += 1
            
            # Bonus for contingency plan
            if pd.notna(row['Contingency Plan']):
                score += 1
            
            # Bonus for documented lessons learned
            if pd.notna(row['Lessons Learned']):
                score += 0.5
            
            scores[i] = score
        
        return pd.Series(scores, index=df.index)
    
    def create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create categorical features using one-hot encoding"""
        categorical_columns = [
            'Risk Category', 'Sub-Category', 'Sector', 'Project Phase',
            'Process Group', 'Status', 'Primary RMS Step'
        ]
        
        # Create a copy for feature engineering
        feature_df = df.copy()
        
        # One-hot encode categorical variables
        for col in categorical_columns:
            if col in feature_df.columns:
                # Create dummy variables
                dummies = pd.get_dummies(feature_df[col], prefix=col, drop_first=True)
                feature_df = pd.concat([feature_df, dummies], axis=1)
        
        return feature_df
    
    def create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from text fields"""
        feature_df = df.copy()
        
        text_columns = ['Risk Title', 'Risk Description', 'Mitigation Strategy', 'Contingency Plan']
        
        for col in text_columns:
            if col in feature_df.columns:
                # Text length features
                feature_df[f'{col}_Length'] = feature_df[col].astype(str).str.len()
                
                # Word count features
                feature_df[f'{col}_Word_Count'] = feature_df[col].astype(str).str.split().str.len()
                
                # Keyword presence features
                keywords = self._get_risk_keywords()
                for keyword in keywords:
                    feature_df[f'{col}_Has_{keyword}'] = feature_df[col].astype(str).str.contains(
                        keyword, case=False, na=False
                    ).astype(int)
        
        return feature_df
    
    def _get_risk_keywords(self) -> List[str]:
        """Get relevant risk keywords for feature engineering"""
        return [
            'delay', 'failure', 'shortage', 'conflict', 'rejection', 'collapse',
            'flood', 'damage', 'dispute', 'approval', 'permit', 'compliance',
            'safety', 'quality', 'environment', 'urgent', 'critical', 'high'
        ]
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from date information"""
        feature_df = df.copy()
        
        # Extract week numbers (simplified approach)
        date_columns = ['Date Identified', 'Review Date']
        
        for col in date_columns:
            if col in feature_df.columns:
                # Extract week numbers
                week_numbers = feature_df[col].astype(str).str.extract(r'Week (\d+)')[0]
                feature_df[f'{col}_Week'] = pd.to_numeric(week_numbers, errors='coerce')
        
        # Create relative timing features
        if 'Date Identified_Week' in feature_df.columns and 'Review Date_Week' in feature_df.columns:
            feature_df['Review_Lag'] = feature_df['Review Date_Week'] - feature_df['Date Identified_Week']
        
        # Create project timeline features
        phase_order = ['Requirements', 'Planning', 'Implementation', 'Execution', 'Monitoring', 'Closing']
        if 'Project Phase' in feature_df.columns:
            feature_df['Phase_Order'] = feature_df['Project Phase'].map(
                {phase: i for i, phase in enumerate(phase_order)}
            )
        
        return feature_df
    
    def create_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated features based on groupings"""
        feature_df = df.copy()
        
        # Category-level aggregations
        category_stats = df.groupby('Risk Category')['Risk Score'].agg(['mean', 'count', 'std']).reset_index()
        category_stats.columns = ['Risk Category', 'Category_Avg_Score', 'Category_Risk_Count', 'Category_Score_Std']
        feature_df = feature_df.merge(category_stats, on='Risk Category', how='left')
        
        # Phase-level aggregations
        phase_stats = df.groupby('Project Phase')['Risk Score'].agg(['mean', 'count']).reset_index()
        phase_stats.columns = ['Project Phase', 'Phase_Avg_Score', 'Phase_Risk_Count']
        feature_df = feature_df.merge(phase_stats, on='Project Phase', how='left')
        
        # Owner-level aggregations
        owner_stats = df.groupby('Risk Owner')['Risk Score'].agg(['mean', 'count']).reset_index()
        owner_stats.columns = ['Risk Owner', 'Owner_Avg_Score', 'Owner_Risk_Count']
        feature_df = feature_df.merge(owner_stats, on='Risk Owner', how='left')
        
        return feature_df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables"""
        feature_df = df.copy()
        
        # Probability-Impact interactions
        feature_df['Prob_Impact_Product'] = feature_df['Probability_Numeric'] * feature_df['Impact_Numeric']
        feature_df['Prob_Impact_Sum'] = feature_df['Probability_Numeric'] + feature_df['Impact_Numeric']
        
        # Category-Phase interactions
        if 'Risk Category' in feature_df.columns and 'Project Phase' in feature_df.columns:
            feature_df['Category_Phase'] = feature_df['Risk Category'] + '_' + feature_df['Project Phase']
        
        # Status-Mitigation interactions
        feature_df['Status_Mitigation_Score'] = feature_df['Is_Open'] * feature_df['Mitigation_Score']
        
        return feature_df
    
    def select_ml_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Select relevant features for ML models"""
        # Define feature categories
        numerical_features = [
            'Probability_Numeric', 'Impact_Numeric', 'Risk Score', 'Has_Mitigation',
            'Has_Contingency', 'Is_Open', 'Risk_Urgency', 'Mitigation_Score'
        ]
        
        categorical_features = [col for col in df.columns if any(cat in col for cat in [
            'Risk Category_', 'Project Phase_', 'Status_', 'Primary RMS Step_'
        ])]
        
        text_features = [col for col in df.columns if any(suffix in col for suffix in [
            '_Length', '_Word_Count', '_Has_'
        ])]
        
        temporal_features = [col for col in df.columns if any(suffix in col for suffix in [
            '_Week', 'Review_Lag', 'Phase_Order'
        ])]
        
        aggregated_features = [col for col in df.columns if any(suffix in col for suffix in [
            'Category_Avg_Score', 'Phase_Avg_Score', 'Owner_Avg_Score',
            'Category_Risk_Count', 'Phase_Risk_Count', 'Owner_Risk_Count'
        ])]
        
        interaction_features = [col for col in df.columns if any(pattern in col for pattern in [
            'Prob_Impact_', 'Category_Phase', 'Status_Mitigation_Score'
        ])]
        
        # Combine all feature types
        selected_features = (
            numerical_features + categorical_features + text_features + 
            temporal_features + aggregated_features + interaction_features
        )
        
        # Filter for existing columns and remove duplicates
        available_features = []
        seen_features = set()
        for col in selected_features:
            if col in df.columns and col not in seen_features:
                available_features.append(col)
                seen_features.add(col)
        
        # Select unique columns from dataframe
        selected_df = df[available_features].copy()
        
        return selected_df, available_features
    
    def prepare_ml_dataset(self) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """Prepare complete ML dataset with all features"""
        print("🔧 Starting feature engineering...")
        
        # Step 1: Prepare base features
        print("  ✓ Creating base features...")
        features_df = self.prepare_base_features()
        
        # Step 2: Create categorical features
        print("  ✓ Creating categorical features...")
        features_df = self.create_categorical_features(features_df)
        
        # Step 3: Create text features
        print("  ✓ Creating text features...")
        features_df = self.create_text_features(features_df)
        
        # Step 4: Create temporal features
        print("  ✓ Creating temporal features...")
        features_df = self.create_temporal_features(features_df)
        
        # Step 5: Create aggregated features
        print("  ✓ Creating aggregated features...")
        features_df = self.create_aggregated_features(features_df)
        
        # Step 6: Create interaction features
        print("  ✓ Creating interaction features...")
        features_df = self.create_interaction_features(features_df)
        
        # Step 7: Select ML features
        print("  ✓ Selecting ML features...")
        ml_df, feature_names = self.select_ml_features(features_df)
        
        # Step 8: Handle missing values
        print("  ✓ Handling missing values...")
        ml_df = self._handle_missing_values(ml_df)
        
        # Create feature metadata
        feature_metadata = {
            "total_features": len(feature_names),
            "feature_categories": {
                "numerical": len([f for f in feature_names if any(num in f for num in ['_Numeric', 'Score', 'Urgency'])]),
                "categorical": len([f for f in feature_names if '_' in f and not any(suf in f for suf in ['_Length', '_Week'])]),
                "text_derived": len([f for f in feature_names if any(suf in f for suf in ['_Length', '_Word_Count', '_Has_'])]),
                "temporal": len([f for f in feature_names if any(suf in f for suf in ['_Week', 'Phase_Order'])]),
                "aggregated": len([f for f in feature_names if any(suf in f for suf in ['_Avg_', '_Count', '_Std'])]),
                "interaction": len([f for f in feature_names if any(pat in f for pat in ['Prob_Impact_', 'Category_Phase'])])
            },
            "feature_names": feature_names,
            "data_shape": ml_df.shape,
            "missing_values": ml_df.isnull().sum().sum()
        }
        
        print(f"  ✓ Generated {len(feature_names)} features from {len(self.risks_df)} risks")
        
        return ml_df, feature_metadata
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        df_clean = df.copy()
        
        # For numerical columns, fill with median
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            null_count = df_clean[col].isnull().sum()
            if null_count > 0:  # Check if there are any null values
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # For categorical columns, fill with mode or 'Unknown'
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            null_count = df_clean[col].isnull().sum()
            if null_count > 0:  # Check if there are any null values
                mode_value = df_clean[col].mode()
                fill_value = mode_value[0] if len(mode_value) > 0 else 'Unknown'
                df_clean[col].fillna(fill_value, inplace=True)
        
        return df_clean
    
    def create_target_variables(self) -> Dict[str, pd.Series]:
        """Create target variables for different prediction tasks"""
        targets = {}
        
        # Binary classification targets
        targets['is_critical'] = (
            (self.risks_df['Probability Rating'] == 'High') & 
            (self.risks_df['Impact Rating'] == 'High')
        ).astype(int)
        
        targets['is_high_risk'] = (self.risks_df['Risk Score'] >= 9).astype(int)
        targets['needs_immediate_attention'] = targets['is_critical']  # Alias for clarity
        
        # Multi-class classification targets
        risk_levels = []
        for _, risk in self.risks_df.iterrows():
            prob = risk['Probability Rating']
            impact = risk['Impact Rating']
            
            if prob == 'High' and impact == 'High':
                risk_levels.append('Critical')
            elif (prob == 'High' and impact == 'Medium') or (prob == 'Medium' and impact == 'High'):
                risk_levels.append('High')
            elif prob == 'Medium' and impact == 'Medium':
                risk_levels.append('Medium')
            else:
                risk_levels.append('Low')
        
        targets['risk_level'] = pd.Series(risk_levels)
        
        # Regression targets
        targets['risk_score'] = self.risks_df['Risk Score']
        targets['probability_numeric'] = self.risks_df['Probability Rating'].map(config.PROBABILITY_WEIGHTS)
        targets['impact_numeric'] = self.risks_df['Impact Rating'].map(config.IMPACT_WEIGHTS)
        
        return targets
    
    def export_ml_dataset(self, output_path: str = None) -> str:
        """Export the prepared ML dataset"""
        if output_path is None:
            output_path = config.ML_DATASET_FILE
        
        # Prepare the dataset
        ml_df, metadata = self.prepare_ml_dataset()
        targets = self.create_target_variables()
        
        # Combine features and targets
        final_df = ml_df.copy()
        for target_name, target_values in targets.items():
            final_df[f'target_{target_name}'] = target_values
        
        # Add original risk IDs for reference
        final_df['Risk_ID'] = self.risks_df['Risk ID']
        
        # Export to CSV
        final_df.to_csv(output_path, index=False)
        
        # Export metadata
        metadata_path = str(output_path).replace('.csv', '_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"✓ ML dataset exported to {output_path}")
        print(f"✓ Metadata exported to {metadata_path}")
        
        return str(output_path)

def main():
    """Main execution function for feature engineering"""
    engineer = FeatureEngineer()
    
    print("🔧 Starting Feature Engineering...")
    print("=" * 50)
    
    # Prepare ML dataset
    ml_df, metadata = engineer.prepare_ml_dataset()
    
    # Export dataset
    dataset_path = engineer.export_ml_dataset()
    
    print(f"✓ Feature engineering complete")
    print(f"✓ Generated {metadata['total_features']} features")
    print(f"✓ Dataset shape: {metadata['data_shape']}")
    print(f"✓ Dataset exported to: {dataset_path}")
    
    return ml_df, metadata

if __name__ == "__main__":
    main()