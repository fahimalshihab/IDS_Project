# Data Science Final Project

Three independent R scripts demonstrating the complete data science lifecycle on real-world datasets. All scripts use the same network traffic dataset from Google Drive for consistency.

## Projects Overview

### 1. Classification - Network Traffic Classification
**Script:** `C-G01-final-classification.R` (142 lines)

**Task:** Binary classification of network traffic (Benign vs Malicious) using Random Forest

**Implementation:**
- **Data Collection (A):** Loads network traffic dataset from Google Drive (26,800 rows, 78 features) with caching for faster reruns
- **Data Understanding (B):** Explores data dimensions, class distribution, missing values, and generates EDA visualizations
- **Data Preprocessing (C):** Converts multi-class to binary, handles missing/infinite values with median imputation, normalizes features using z-score standardization
- **Modeling (D):** 70/30 train-test split, Random Forest with 50 trees
- **Evaluation (E):** Confusion matrix, accuracy, precision, recall, F1-score, feature importance analysis

**Results:** 98.66% accuracy, 98.37% precision, 97.36% recall, 97.86% F1-score

**Outputs:**
- `classification_plots.pdf` - EDA visualizations (class distribution, feature histograms, boxplots)
- `classification_results.pdf` - Model performance (confusion matrix heatmap, feature importance)

---

### 2. Regression - Network Flow Prediction
**Script:** `C-G01-final-regression.R` (125 lines)

**Task:** Regression analysis to predict network flow characteristics using Linear Regression

**Implementation:**
- **Data Collection (A):** Uses same network traffic dataset from Google Drive with caching
- **Data Understanding (B):** Analyzes dataset structure and generates distribution visualizations
- **Data Preprocessing (C):** Converts categorical labels to numeric, handles missing/infinite values, normalizes features with z-score standardization
- **Modeling (D):** 70/30 train-test split, Linear Regression model
- **Evaluation (E):** RMSE, MAE, R² metrics, coefficient analysis, residual diagnostics

**Results:** RMSE: 1.4084, MAE: 1.0283, R²: 0.4935

**Outputs:**
- `regression_plots.pdf` - EDA visualizations (feature distributions)
- `regression_results.pdf` - Model diagnostics (actual vs predicted, residuals, Q-Q plot, residual histogram)

---

### 3. Clustering - Network Traffic Segmentation
**Script:** `C-G01-final-clustering.R` (133 lines)

**Task:** Unsupervised clustering to segment network traffic patterns using K-Means

**Implementation:**
- **Data Collection (A):** Uses network traffic dataset from Google Drive (sampled to 5,000 rows for performance)
- **Data Understanding (B):** Explores data structure and generates feature distribution visualizations
- **Data Preprocessing (C):** Handles missing/infinite values, normalizes features, samples data for computational efficiency
- **Modeling (D):** Optimal cluster selection using Elbow Method and Silhouette Analysis (tests K=2-8), trains K-Means with best K
- **Evaluation (E):** Silhouette score, variance explained, cluster size analysis, cluster profile statistics

**Results:** 2 optimal clusters, Silhouette Score: 0.9491, Variance Explained: 21.58%

**Outputs:**
- `clustering_plots.pdf` - EDA visualizations (feature distributions)
- `clustering_results.pdf` - Clustering analysis (cluster sizes, silhouette plot, elbow curve, PCA visualization)

---

## Requirements

Install required R packages:
```r
install.packages(c("randomForest", "cluster"))
```

## Usage

Run each script independently in R or RStudio:
```r
source("C-G01-final-classification.R")
source("C-G01-final-regression.R")
source("C-G01-final-clustering.R")
```

Or from command line:
```bash
Rscript C-G01-final-classification.R
Rscript C-G01-final-regression.R
Rscript C-G01-final-clustering.R
```

## Dataset

All scripts use the same network traffic dataset hosted on Google Drive:
- **Size:** 26,800 rows × 78 features
- **Type:** Network flow statistics (packet sizes, timing, protocols, etc.)
- **Access:** Automatically downloaded and cached on first run
- **Cache Location:** `/tmp/test_dataset.csv`

## Data Science Lifecycle (A-E)

Each script implements all five stages:
- **A. Data Collection** - Automated dataset loading from Google Drive with caching
- **B. Data Understanding** - Exploratory analysis with statistical summaries and visualizations
- **C. Data Preprocessing** - Cleaning, normalization, feature engineering, missing value handling
- **D. Modeling** - Training machine learning models appropriate to each task
- **E. Evaluation** - Performance metrics, model interpretation, and result visualizations

## Output Files

All scripts generate PDF visualizations:

| Script | EDA Plots | Results Plots |
|--------|-----------|---------------|
| Classification | `classification_plots.pdf` | `classification_results.pdf` |
| Regression | `regression_plots.pdf` | `regression_results.pdf` |
| Clustering | `clustering_plots.pdf` | `clustering_results.pdf` |

**Note:** PDF files are not committed to the repository. Run the scripts to generate them locally.

## Requirements

```r
install.packages(c("randomForest", "kagglehub", "cluster"))
```

## Usage

Run each script independently:
```r
source("C-G01-final-classification.R")
source("C-G01-final-regression.R")
source("C-G01-final-clustering.R")
```

## Lifecycle (A-E)

Each script implements:
- **A. Data Collection** - Automated loading
- **B. Data Understanding** - EDA with visualizations
- **C. Data Preprocessing** - Cleaning, normalization, feature engineering
- **D. Modeling** - ML model training
- **E. Evaluation** - Metrics and interpretation

## Outputs

- Console metrics and results
- EDA visualizations (histograms, boxplots, distributions)
- Results visualizations (confusion matrix, residual plots, cluster plots)
- All visualizations saved as PDF files
