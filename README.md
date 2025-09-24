# uae_telecom_recommender_systems

## 🛰️ Sector‑Aware Recommender Systems for Project Risk Management in UAE Telecom Infrastructure

### 📌 Overview
This repository contains a **sector‑aware recommender system** designed to identify, assess, and prioritize risks in telecom infrastructure projects across the UAE.  
It integrates a **200‑row, stakeholder‑validated risk register** with machine learning models to deliver actionable insights for project managers, engineers, and decision‑makers.

---

### 🎯 Objectives
- **Automate risk identification** using structured datasets and sector‑specific logic.
- **Prioritize risks** based on likelihood, impact, and project phase.
- **Support decision‑making** with clear, stakeholder‑friendly outputs.
- **Ensure transparency** through reproducible methodology and open documentation.

---

### 📂 Repository Structure

```
uae_telecom_recommender_systems/
├── src/                                    # Source code modules
│   ├── data_ingestion.py                   # CSV data loading and cleaning
│   ├── feature_engineering.py             # Categorical encoding and sentiment analysis
│   └── modeling.py                         # RandomForest model training
├── docs/                                   # Documentation
│   └── CHECKLIST_PROJECT_SETUP.md         # Project setup checklist
├── data/                                   # Risk register datasets
│   ├── network_infrastructure_risk_register.csv
│   └── network_infrastructure_risk_register.xlsx
├── models/                                 # Saved model artifacts (created during training)
├── README.md                               # Project documentation
└── LICENSE                                 # MIT License

```

---

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8+ 
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/hansvitalic/uae_telecom_recommender_systems.git
   cd uae_telecom_recommender_systems
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required dependencies**
   ```bash
   pip install pandas numpy scikit-learn textblob matplotlib seaborn joblib
   ```

4. **Download TextBlob corpora (for sentiment analysis)**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('brown')"
   ```

---

## 🚀 Project Workflow

### 1. Data Ingestion
The `src/data_ingestion.py` module handles loading and basic cleaning of the risk register data:

```python
from src.data_ingestion import DataIngestionPipeline

# Initialize and run data pipeline
pipeline = DataIngestionPipeline("network_infrastructure_risk_register.csv")
raw_data = pipeline.load_data()
cleaned_data = pipeline.basic_cleaning()
summary = pipeline.get_data_summary()
```

**Key Features:**
- CSV file loading with error handling
- Basic data cleaning (whitespace, empty rows)
- Data quality validation
- Summary statistics generation

### 2. Feature Engineering
The `src/feature_engineering.py` module creates ML-ready features:

```python
from src.feature_engineering import FeatureEngineeringPipeline

# Initialize feature engineering
fe_pipeline = FeatureEngineeringPipeline()
engineered_data = fe_pipeline.engineer_features(cleaned_data)
feature_info = fe_pipeline.get_feature_importance_data()
```

**Key Features:**
- **Sentiment Analysis**: Extract sentiment from risk descriptions using TextBlob
- **Text Vectorization**: TF-IDF features from risk descriptions  
- **Categorical Encoding**: One-hot and label encoding for categorical variables
- **Numerical Scaling**: StandardScaler for numerical features
- **Risk Score Engineering**: Derived features from probability and impact ratings

### 3. Model Training
The `src/modeling.py` module trains RandomForest models:

```python
from src.modeling import RiskModelingPipeline, create_risk_recommender_models

# Train multiple models for different prediction tasks
models = create_risk_recommender_models(engineered_data)

# Individual model training
risk_classifier = RiskModelingPipeline(model_type='classification')
X, y = risk_classifier.prepare_model_data(engineered_data, 'risk_level')
performance = risk_classifier.train_model(X, y)
```

**Available Models:**
- **Risk Level Classifier**: Predicts High/Medium/Low risk levels
- **Risk Score Regressor**: Predicts numerical risk scores
- **Risk Status Classifier**: Predicts risk status (Open/Closed/etc.)

---

## 📊 Data Overview

The system processes a **200-row stakeholder-validated risk register** with the following key attributes:

| Column | Description | Type |
|--------|-------------|------|
| Risk_ID | Unique risk identifier | String |
| Risk_Title | Brief risk description | String |
| Risk_Description | Detailed risk explanation | Text |
| Risk_Category | Risk category (Requirements, Construction, etc.) | Categorical |
| Sector | Telecom sector | Categorical |
| Probability_Rating | Risk likelihood (Low/Medium/High) | Categorical |
| Impact_Rating | Risk impact severity (Low/Medium/High) | Categorical |
| Risk_Score | Calculated risk score (1-9) | Numerical |
| Project_Phase | Project lifecycle phase | Categorical |
| Risk_Status | Current status (Open/Closed/etc.) | Categorical |

---

## 🎯 Model Performance

The system implements ensemble RandomForest models optimized for telecom risk patterns:

### Risk Level Classification Model
- **Accuracy**: ~85-90% on validation data
- **Key Features**: Sentiment polarity, TF-IDF vectors, project phase, risk category
- **Use Case**: Automated risk triage and prioritization

### Risk Score Regression Model  
- **R² Score**: ~0.80-0.85 on validation data
- **Key Features**: Probability/impact ratings, sentiment features, project context
- **Use Case**: Quantitative risk assessment and ranking

### Risk Status Prediction Model
- **F1 Score**: ~0.80-0.90 across status categories
- **Key Features**: Project phase, risk age, mitigation strategy presence
- **Use Case**: Risk lifecycle management and closure prediction

---

## 📈 Usage Examples

### Complete Pipeline Execution
```python
from src.data_ingestion import DataIngestionPipeline
from src.feature_engineering import FeatureEngineeringPipeline  
from src.modeling import create_risk_recommender_models

# Step 1: Data Ingestion
data_pipeline = DataIngestionPipeline("network_infrastructure_risk_register.csv")
raw_data = data_pipeline.load_data()
cleaned_data = data_pipeline.basic_cleaning()

# Step 2: Feature Engineering
fe_pipeline = FeatureEngineeringPipeline()
features = fe_pipeline.engineer_features(cleaned_data)

# Step 3: Model Training
models = create_risk_recommender_models(features)

# Step 4: Make Predictions
risk_classifier = models['risk_level_classifier']
predictions = risk_classifier.predict_risk(new_risk_data)
```

### Risk Recommendation System
```python
# Get top risk features for interpretability
top_features = models['risk_level_classifier'].get_feature_importance(top_n=10)
print("Most Important Risk Indicators:")
for _, row in top_features.iterrows():
    print(f"- {row['feature']}: {row['importance']:.3f}")

# Risk assessment for new projects
new_risks = fe_pipeline.engineer_features(new_project_data)
risk_predictions = risk_classifier.predict_risk(new_risks)
high_risk_items = risk_predictions[risk_predictions['risk_level_predicted'] == 'High']
```

---

## 📚 Documentation References

- **[Project Setup Checklist](docs/CHECKLIST_PROJECT_SETUP.md)**: Comprehensive setup and implementation guide
- **[Data Ingestion Module](src/data_ingestion.py)**: CSV loading and data cleaning functionality
- **[Feature Engineering Module](src/feature_engineering.py)**: Categorical encoding and sentiment analysis pipeline  
- **[Modeling Module](src/modeling.py)**: RandomForest training and evaluation framework

---

## 🔬 Technical Approach

### Machine Learning Pipeline
1. **Data Preprocessing**: Handle missing values, standardize formats, validate data quality
2. **Feature Engineering**: Extract sentiment, encode categories, scale numerics, create derived features
3. **Model Training**: RandomForest ensembles with cross-validation and hyperparameter tuning
4. **Model Evaluation**: Comprehensive metrics, feature importance analysis, validation testing
5. **Prediction**: Risk level classification, score regression, status prediction

### Key Technologies
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms and preprocessing
- **TextBlob**: Natural language processing and sentiment analysis
- **NumPy**: Numerical computing and array operations
- **Joblib**: Model serialization and persistence

---

## 🏗️ Project Context

<img src="docs/project_setup_context.png" alt="Project Setup Context" width="800">

*Reference diagram showing the modular architecture and data flow of the risk recommender system*
