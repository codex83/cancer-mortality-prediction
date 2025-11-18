# Cancer Mortality Prediction with Model Monitoring

A comprehensive machine learning pipeline for predicting cancer mortality rates at the county level, featuring advanced model monitoring and data drift detection using Evidently AI.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)](https://scikit-learn.org/)
[![Evidently](https://img.shields.io/badge/Evidently-0.4.33-green.svg)](https://evidentlyai.com/)

## 🎯 Project Overview

This project implements an end-to-end machine learning solution for predicting county-level cancer mortality rates (deaths per 100,000 population) using demographic, socioeconomic, and health-related features. The system includes comprehensive model monitoring capabilities to detect data drift and track model performance degradation over time.

### Key Features

- **🤖 Advanced ML Pipeline**: Complete data preprocessing, model training, and evaluation workflow
- **📊 Model Monitoring**: Integrated Evidently AI for comprehensive drift detection and performance tracking
- **🔬 Scenario Analysis**: Systematic testing of model robustness under data drift conditions
- **📈 Interactive Reports**: Automated generation of HTML dashboards for drift visualization
- **🏗️ Production-Ready**: Modular architecture with separated concerns and reusable components

## 📋 Table of Contents

- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Monitoring & Drift Detection](#monitoring--drift-detection)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 📊 Dataset

**Source**: Cancer Registry Dataset  
**Records**: 3,047 US counties  
**Target Variable**: `TARGET_deathRate` - Cancer mortality rate per 100,000 population  
**Features**: 32 demographic, socioeconomic, and health-related variables including:

- Economic indicators (median income, poverty percentage)
- Demographics (age, household size, race distribution)
- Education (high school, bachelor's degree percentages)
- Healthcare access (insurance coverage rates)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/cancer-mortality-prediction.git
cd cancer-mortality-prediction

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import pandas, sklearn, evidently; print('✓ All dependencies installed successfully!')"
```

## 💻 Usage

### Option 1: Jupyter Notebook (Recommended)

Interactive execution with visualizations:

```bash
jupyter notebook cancer_mortality_prediction.ipynb
```

Then run all cells or execute step-by-step to see outputs and visualizations.

### Option 2: Python Scripts

Automated batch execution:

```bash
# Using the modular approach
cd pyfiles
python main.py
```

### What Gets Generated

After execution, the following outputs are created:

```
cancer-mortality-prediction/
├── models/
│   └── cancer_model.pkl                    # Trained model
├── predictions/
│   ├── predictions_scenario_A.csv          # Economic decline scenario
│   ├── predictions_scenario_AB.csv         # Economic + poverty scenario
│   └── predictions_scenario_ABC.csv        # Multi-factor drift scenario
└── evidently_reports/
    ├── data_drift_report_*.html            # Drift detection reports
    ├── model_performance_report_*.html     # Performance tracking
    ├── feature_drift_report_*.html         # Feature-level analysis
    └── scenario_comparison.csv             # Metrics summary
```

## 📁 Project Structure

```
cancer-mortality-prediction/
├── cancer_mortality_prediction.ipynb   # Main Jupyter notebook
├── pyfiles/                            # Modular Python implementation
│   ├── data_loader.py                  # Data loading & preprocessing
│   ├── model_trainer.py                # Model training & evaluation
│   ├── model_monitoring.py             # Evidently AI integration
│   ├── scenario_testing.py             # Drift scenario testing
│   └── main.py                         # Orchestration script
├── data/
│   └── cancer_reg.csv                  # Dataset
├── models/                             # Saved models (generated)
├── predictions/                        # Prediction outputs (generated)
├── evidently_reports/                  # Monitoring reports (generated)
├── requirements.txt                    # Python dependencies
├── .gitignore                         # Git ignore rules
└── README.md                          # This file
```

## 🔧 Model Architecture

### Algorithm: Gradient Boosting Regressor

**Rationale**: Chosen for its excellent performance on tabular data, ability to capture non-linear relationships, and robustness to outliers.

**Hyperparameters**:
```python
n_estimators=200
learning_rate=0.1
max_depth=5
min_samples_split=5
min_samples_leaf=2
```

### Feature Engineering

- **Scaling**: StandardScaler normalization
- **Missing Values**: Median imputation for numerical features
- **Feature Selection**: Dropped non-predictive categorical features (Geography, binnedInc)

### Evaluation Metrics

- **RMSE**: Root Mean Squared Error (primary metric)
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error

## 📈 Monitoring & Drift Detection

### Evidently AI Integration

The project uses [Evidently AI](https://evidentlyai.com/) for comprehensive model monitoring:

1. **Data Drift Detection**: Statistical tests (Kolmogorov-Smirnov, Chi-squared) to detect distribution changes
2. **Model Performance Tracking**: Continuous evaluation of regression metrics
3. **Feature-Level Analysis**: Individual feature drift scores and visualizations
4. **Prediction Drift**: Monitoring changes in model output distribution

### Drift Scenarios Tested

| Scenario | Modifications | Purpose |
|----------|--------------|---------|
| **A** | Decrease median income by $40,000 | Economic sensitivity |
| **AB** | A + Increase poverty by 20 percentage points | Compounding socioeconomic stress |
| **ABC** | AB + Increase household size by 2 | Multi-dimensional demographic shift |

### Opening Monitoring Reports

```bash
# Navigate to reports directory
cd evidently_reports

# Open any HTML report in your browser
open data_drift_report_A_*.html  # macOS
# or
xdg-open data_drift_report_A_*.html  # Linux
# or simply double-click the file on Windows
```

## 📊 Results

### Baseline Performance

| Metric | Value |
|--------|-------|
| RMSE | ~20.0 |
| MAE | ~15.0 |
| R² | ~0.50 |
| MAPE | ~9% |

*Note: Exact values may vary based on random seed*

### Top 5 Most Important Features

1. `medIncome` - Median household income
2. `povertyPercent` - Percentage below poverty line
3. `PctBachDeg25_Over` - Bachelor's degree attainment
4. `PctUnemployed16_Over` - Unemployment rate
5. `PctPrivateCoverage` - Private insurance coverage

### Drift Impact

As expected, model performance degrades progressively:
- **Scenario A**: Moderate drift, slight performance decrease
- **Scenario AB**: Significant drift, noticeable degradation
- **Scenario ABC**: Severe drift, substantial performance impact

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Areas for Improvement

- [ ] Hyperparameter optimization (GridSearchCV/RandomizedSearchCV)
- [ ] Additional feature engineering (interaction terms, polynomial features)
- [ ] Ensemble methods (stacking multiple models)
- [ ] Real-time monitoring dashboard
- [ ] API for model serving
- [ ] Docker containerization
- [ ] CI/CD pipeline integration

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Cancer Registry Data
- **Monitoring Framework**: [Evidently AI](https://evidentlyai.com/)
- **ML Framework**: [scikit-learn](https://scikit-learn.org/)

## 📧 Contact

**Hritik Jhaveri**

- GitHub: [@codex83](https://github.com/codex83)
- LinkedIn: [hritik-jhaveri](https://linkedin.com/in/hritik-jhaveri)
- Email: htj2@uchicago.edu

## ⭐ Star This Repository

If you find this project useful, please consider giving it a star! It helps others discover the project and motivates continued development.

---

**Built with ❤️ using Python, scikit-learn, and Evidently AI**
