# Advanced Diabetes Data Processing Pipeline

## Overview

A comprehensive data science pipeline for analyzing diabetes patient data using advanced preprocessing, intelligent categorical encoding, and statistical modeling techniques. This project focuses on developing robust predictive models for hospital readmission risk assessment in diabetic patients.

## Project Objectives

- **Clinical Prediction**: Develop models to predict hospital readmission risk for diabetic patients
- **Data Quality**: Implement robust preprocessing pipeline maintaining clinical data integrity  
- **Advanced Encoding**: Apply state-of-the-art categorical encoding strategies optimized for healthcare data
- **Statistical Analysis**: Support comprehensive statistical modeling and machine learning approaches
- **Reproducibility**: Ensure full reproducibility with documented methodologies and version control

## Technical Architecture

```
Progetto_mott/
├── database/
│   └── diabetic_data.csv                           # Raw dataset (101,766 patients)
├── outputs/
│   ├── cleaned_data/
│   │   ├── diabetes_clean_no_encoding.csv          # [37,764 × 40] Preprocessed data
│   │   ├── diabetes_regression_ready.csv           # [30,211 × 21] ML-ready with feature selection
│   │   └── diabetes_ml_arrays.npz                  # NumPy arrays for ML models
│   ├── encoded/
│   │   ├── diabetes_smart_onehot.csv               # [37,764 × 84] Intelligent one-hot encoding
│   │   ├── diabetes_robust_target.csv              # [37,764 × 40] Cross-validation target encoding
│   │   ├── diabetes_advanced_encoding.csv          # [37,764 × 53] Binary + hashing methods
│   │   ├── diabetes_ensemble_encoding.csv          # [37,764 × 76] Adaptive optimal encoding
│   │   └── encoding_analysis_report.csv            # [29 × 7] Detailed encoding analysis
│   └── regression_results/
│       ├── model_comparison_report.csv             # [3 × 4] Model performance comparison
│       ├── feature_importance_report.csv           # [37 × 2] Feature importance rankings
│       └── regression_recommendations.txt          # Clinical recommendations and insights
├── diabetes_data_cleaner.py                        # Core preprocessing pipeline
├── encoding_pipeline.py                            # Advanced encoding pipeline  
├── diabetes_regression_analysis.py                 # Complete regression analysis
├── requirements.txt                                # Dependencies specification
└── README.md                                       # This documentation
```

## Installation & Setup

### Prerequisites

Ensure Python 3.8+ is installed, then install dependencies:

```bash
# Install core requirements
pip install -r requirements.txt

# Or install manually
pip install pandas>=2.0.0 numpy>=1.24.0 scikit-learn>=1.3.0 category-encoders>=2.6.0
```

### Quick Start

```bash
# 1. Clone/download the project
cd Progetto_mott

# 2. Run complete pipeline
python diabetes_data_cleaner.py        # Step 1: Data preprocessing
python encoding_pipeline.py            # Step 2: Advanced encoding
python diabetes_regression_analysis.py # Step 3: Regression analysis
```

## Complete Usage Guide

### Step 1: Data Preprocessing 

**Execute:** `python diabetes_data_cleaner.py`

**What it does:**
- Loads raw diabetes dataset (101,766 hospital encounters)
- Removes non-predictive administrative columns
- Handles missing values with authentic data approach (no imputation)
- Eliminates patient duplicates (keeps first admission per patient)
- Converts age ranges to numerical midpoints
- Creates binary readmission target variable

**Files Generated:**

#### `outputs/cleaned_data/diabetes_clean_no_encoding.csv`
- **Size**: 37,764 patients × 40 variables
- **Content**: Clean data ready for encoding (categorical variables preserved as text)
- **Use Case**: Input for encoding pipeline
- **Key Features**: No missing values, no patient duplicates, clinical variables preserved

#### `outputs/cleaned_data/diabetes_regression_ready.csv`  
- **Size**: 30,211 patients × 21 variables
- **Content**: ML-ready with feature selection and scaling applied
- **Use Case**: Direct use in traditional ML pipelines
- **Key Features**: Top 20 features selected, standardized numerical values

#### `outputs/cleaned_data/diabetes_ml_arrays.npz`
- **Content**: Pre-split NumPy arrays (X_train, X_test, y_train, y_test)
- **Use Case**: Direct loading into ML models without preprocessing
- **Format**: Compressed NumPy binary format

**How to read:**
```python
import pandas as pd
import numpy as np

# Read cleaned data
df = pd.read_csv('outputs/cleaned_data/diabetes_clean_no_encoding.csv')
print(f"Dataset shape: {df.shape}")
print(f"Target distribution: {df['readmitted_binary'].value_counts()}")

# Load ML arrays
arrays = np.load('outputs/cleaned_data/diabetes_ml_arrays.npz')
X_train, y_train = arrays['X_train'], arrays['y_train']
```

### Step 2: Advanced Encoding

**Execute:** `python encoding_pipeline.py`

**What it does:**
- Analyzes categorical variables for optimal encoding strategy
- Applies 4 different encoding approaches in parallel
- Generates comprehensive analysis report with recommendations
- Optimizes memory usage and prevents overfitting

**Files Generated:**

#### `outputs/encoded/diabetes_smart_onehot.csv`
- **Size**: 37,764 patients × 84 variables  
- **Strategy**: Intelligent one-hot encoding for variables ≤10 categories
- **Use Case**: Statistical analysis requiring maximum interpretability
- **Features**: Rare category consolidation, memory-optimized dtypes
- **Best For**: ANOVA, chi-square tests, clinical interpretation

#### `outputs/encoded/diabetes_robust_target.csv` **RECOMMENDED FOR ML**
- **Size**: 37,764 patients × 40 variables
- **Strategy**: Cross-validation target encoding (prevents overfitting)  
- **Use Case**: Machine learning with high-cardinality categorical variables
- **Features**: 5-fold CV encoding, out-of-fold predictions
- **Best For**: Random Forest, Gradient Boosting, Logistic Regression

#### `outputs/encoded/diabetes_advanced_encoding.csv`
- **Size**: 37,764 patients × 53 variables
- **Strategy**: Binary + hashing encoding for dimensionality reduction
- **Use Case**: Memory-constrained environments, large-scale deployment
- **Features**: Hash-based encoding, binary representations
- **Best For**: Production systems, embedded applications

#### `outputs/encoded/diabetes_ensemble_encoding.csv`
- **Size**: 37,764 patients × 76 variables
- **Strategy**: Adaptive optimal encoding (different strategy per variable)
- **Use Case**: General-purpose optimal encoding
- **Features**: Automatic strategy selection based on variable characteristics
- **Best For**: Exploratory analysis, benchmark comparisons

#### `outputs/encoded/encoding_analysis_report.csv`
- **Size**: 29 variables × 7 metrics
- **Content**: Detailed analysis of each categorical variable
- **Use Case**: Understanding encoding decisions and variable characteristics

**How to read:**
```python
import pandas as pd

# For statistical analysis (most interpretable)
df_stats = pd.read_csv('outputs/encoded/diabetes_smart_onehot.csv')

# For machine learning (best performance)
df_ml = pd.read_csv('outputs/encoded/diabetes_robust_target.csv')

# Read encoding analysis
analysis = pd.read_csv('outputs/encoded/encoding_analysis_report.csv')
print("Encoding strategies used:")
print(analysis[['Variable', 'Unique_Values', 'Recommended_Strategy']])
```

**Columns in analysis report:**
- `Variable`: Categorical variable name
- `Unique_Values`: Number of unique categories
- `Missing_%`: Percentage of missing values  
- `Most_Frequent_%`: Percentage of most common category
- `Rare_Categories`: Count of categories with <20 occurrences
- `Recommended_Strategy`: Optimal encoding method
- `Memory_MB`: Memory usage in megabytes

### Step 3: Regression Analysis

**Execute:** `python diabetes_regression_analysis.py`

**What it does:**
- Loads optimal encoded dataset for regression
- Tests 3 different regression models with cross-validation
- Analyzes feature importance and model interpretability  
- Generates comprehensive performance reports and clinical recommendations

**Files Generated:**

#### `outputs/regression_results/model_comparison_report.csv`
- **Size**: 3 models × 4 metrics
- **Content**: Performance comparison of all tested models
- **Use Case**: Model selection and performance benchmarking

**Columns explained:**
- `Model`: Model name (Logistic Regression, Ridge Logistic, Random Forest)
- `AUC_Test`: Area Under ROC Curve on test set (0.5-1.0, higher is better)
- `AUC_CV_Mean`: Average AUC from 5-fold cross-validation
- `AUC_CV_Std`: Standard deviation of CV scores (lower = more stable)

#### `outputs/regression_results/feature_importance_report.csv`
- **Size**: 37 features × 2 metrics
- **Content**: Feature importance rankings from Random Forest model
- **Use Case**: Understanding which clinical variables are most predictive

**Columns explained:**
- `feature`: Variable name from the dataset
- `importance`: Importance score (0-1, higher = more predictive)

**Clinical interpretation of top features:**
- `num_lab_procedures`: Number of laboratory tests → complexity of care
- `medical_specialty_target_enc`: Medical specialty → specialized care patterns
- `num_medications`: Number of medications → comorbidity burden
- `time_in_hospital`: Length of stay → illness severity
- `age`: Patient age → frailty and risk factors

#### `outputs/regression_results/regression_recommendations.txt`
- **Content**: Clinical insights, model interpretation, and next steps
- **Use Case**: Clinical decision support and model implementation guidance

**How to read and interpret:**
```python
import pandas as pd

# Read model comparison
models = pd.read_csv('outputs/regression_results/model_comparison_report.csv')
best_model = models.loc[models['AUC_Test'].idxmax()]
print(f"Best model: {best_model['Model']} (AUC: {best_model['AUC_Test']:.4f})")

# Read feature importance
features = pd.read_csv('outputs/regression_results/feature_importance_report.csv')
print("Top 5 most predictive features:")
print(features.head())

# Clinical risk interpretation
with open('outputs/regression_results/regression_recommendations.txt', 'r') as f:
    recommendations = f.read()
    print(recommendations)
```

**AUC Score Interpretation:**
- **0.9-1.0**: Excellent prediction capability
- **0.8-0.9**: Good prediction capability  
- **0.7-0.8**: Fair prediction capability
- **0.6-0.7**: Poor but potentially useful
- **0.5-0.6**: Little to no prediction capability

**Typical results from this pipeline:**
- **Expected AUC**: 0.62-0.68 (fair to good for real medical data)
- **Best Model**: Usually Random Forest
- **Key Predictors**: Laboratory procedures, medications, hospital stay duration

## Data Quality Metrics

### Raw Dataset Analysis
- **Initial Records**: 101,766 hospital encounters
- **Initial Variables**: 50 features (mix of numerical and categorical)
- **Missing Data**: Varies by column (0-98% missing)
- **Patient Duplicates**: 30,248 duplicate encounters removed

### Processed Dataset Quality  
- **Final Patients**: 37,764 unique patients (71,518 → 37,764 after encoding pipeline)
- **Final Variables**: 29 categorical + 8 numerical features
- **Missing Data**: Completely eliminated via strategic dropna approach
- **Data Integrity**: 100% authentic medical records (no imputation bias)

### Categorical Variable Distribution
- **Total Categorical**: 29 variables analyzed
- **Binary Variables (2 categories)**: 16 variables → Binary encoding
- **Low Cardinality (3-10 categories)**: 12 variables → One-hot encoding  
- **High Cardinality (>10 categories)**: 1 variable → Target/Hashing encoding

## Advanced Technical Features

### Cross-Validation Target Encoding
Prevents overfitting in target encoding through out-of-fold predictions:

```python
# Implementation in encoding_pipeline.py
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(df):
    target_means = df.iloc[train_idx].groupby(col)[target_col].mean()
    df.loc[val_idx, f'{col}_target_enc'] = df.loc[val_idx, col].map(target_means)
```

### Memory Optimization
Reduces memory usage by ~75% through intelligent data types:

```python
# Automatic optimization in encoding pipeline
df_encoded = pd.get_dummies(df, dtype=np.int8)  # Instead of int64
```

### Adaptive Strategy Selection
Automatically chooses optimal encoding based on variable characteristics:

```python
# Strategy selection logic
if unique_count <= 2: strategy = 'binary'
elif unique_count <= 10: strategy = 'onehot'  
elif unique_count <= 50: strategy = 'target_encoding'
else: strategy = 'hashing'
```

## Clinical Applications

### Readmission Risk Prediction
- **Primary Use**: Identify patients at high risk of 30-day readmission
- **Clinical Value**: Enable targeted interventions and follow-up programs
- **Implementation**: Integrate predictions into electronic health records

### Population Health Management
- **Risk Stratification**: Categorize patients by readmission risk levels
- **Resource Allocation**: Prioritize high-risk patients for care management
- **Quality Metrics**: Monitor and improve hospital readmission rates

### Clinical Decision Support
- **Real-Time Alerts**: Flag high-risk patients during discharge planning
- **Care Pathways**: Recommend appropriate post-discharge care intensity
- **Performance Monitoring**: Track prediction accuracy over time

## Best Practices for Different Use Cases

### For Statistical Analysis
```python
# Use most interpretable encoding
df = pd.read_csv('outputs/encoded/diabetes_smart_onehot.csv')
# Maintains clinical meaning, suitable for hypothesis testing
```

### For Machine Learning Development
```python
# Use robust target encoding
df = pd.read_csv('outputs/encoded/diabetes_robust_target.csv')
# Best balance of performance and overfitting prevention
```

### For Production Deployment
```python
# Use memory-efficient encoding
df = pd.read_csv('outputs/encoded/diabetes_advanced_encoding.csv')  
# Optimized for computational efficiency and storage
```

### For Exploratory Analysis
```python
# Use adaptive ensemble encoding
df = pd.read_csv('outputs/encoded/diabetes_ensemble_encoding.csv')
# Each variable optimally encoded for exploration
```

## Performance Benchmarks

| Encoding Strategy | File Size | Variables | Memory Usage | ML Performance | Interpretability |
|------------------|-----------|-----------|--------------|----------------|------------------|
| Smart One-Hot    | 4.8 MB    | 84        | High         | Good           | Excellent        |
| Robust Target    | 4.1 MB    | 40        | Low          | Excellent      | Good             |
| Advanced Methods | 4.0 MB    | 53        | Low          | Good           | Fair             |
| Ensemble Optimal | 5.9 MB    | 76        | Moderate     | Excellent      | Good             |

## Troubleshooting Guide

### Common Issues and Solutions

#### Memory Errors
```bash
# Reduce memory usage
python -c "import pandas as pd; pd.set_option('mode.chained_assignment', None)"
# Or use advanced encoding method for smaller memory footprint
```

#### Encoding Failures  
- **Issue**: New categorical values in production data
- **Solution**: Ensure consistent preprocessing pipeline and data validation

#### Missing Dependencies
```bash
# Install missing packages
pip install --upgrade category-encoders scikit-learn pandas numpy
```

#### File Path Errors
- **Issue**: Output directories don't exist
- **Solution**: Scripts automatically create directories, but ensure write permissions

#### Performance Issues
- **Issue**: Slow processing on large datasets
- **Solution**: Use `diabetes_advanced_encoding.csv` for memory-constrained environments

## File Format Specifications

### CSV Files
- **Encoding**: UTF-8
- **Separator**: Comma (,)
- **Header**: First row contains column names
- **Missing Values**: Eliminated during preprocessing (no NaN values)
- **Data Types**: Automatically inferred by pandas

### NumPy Files (.npz)
- **Format**: Compressed NumPy binary
- **Contents**: Multiple arrays stored in single file
- **Access**: Use `np.load()` to access individual arrays
- **Advantage**: Faster loading, smaller file size than CSV

### Text Files (.txt)
- **Format**: Plain text with structured sections
- **Content**: Human-readable reports and recommendations
- **Purpose**: Clinical interpretation and documentation

## Contributing

### Development Guidelines
- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Maintain backward compatibility
- Add unit tests for new features

### Testing
```bash
# Run tests (if implemented)
pytest tests/

# Validate pipeline integrity
python diabetes_data_cleaner.py
python encoding_pipeline.py  
python diabetes_regression_analysis.py
```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@misc{diabetes_encoding_pipeline,
  title={Advanced Diabetes Data Processing Pipeline},
  author={Healthcare Analytics Team},
  year={2025},
  publisher={GitHub},
  url={https://github.com/your-repo/diabetes-pipeline}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original dataset from UCI Machine Learning Repository  
- Inspired by best practices in healthcare data preprocessing
- Built with modern data science libraries and methodologies

---

**Contact**: For questions or collaboration opportunities, please open an issue on GitHub.

**Version**: 2.1.0 | **Last Updated**: January 2025