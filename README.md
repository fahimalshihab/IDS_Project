# Data Science Final Project

Three independent R scripts demonstrating the complete data science lifecycle on real-world datasets.

## Projects

### 1. Classification - Network Traffic Classification
**File:** `C-G01-final-classification.R`

**Overview:** Classifies network traffic as Benign or Malicious using Random Forest algorithm.

**Code Explanation:**
- **Data Collection (A):** Downloads network traffic dataset from Google Drive (26,800 rows, 78 features). Checks for cached version to avoid re-downloading.
- **Data Understanding (B):** Displays dataset dimensions, class distribution, and missing value count. Generates visualizations: class distribution bar chart, histograms of first 4 features, boxplots for outlier detection.
- **Data Preprocessing (C):** Converts multi-class labels to binary (most common class = 0, rest = 1). Keeps only numeric features. Handles missing/infinite values with median imputation. Normalizes features using z-score standardization. Removes incomplete rows.
- **Modeling (D):** Splits data 70/30 train/test. Trains Random Forest with 50 trees.
- **Evaluation (E):** Generates confusion matrix. Calculates accuracy, precision, recall, F1-score. Displays top 10 important features. Creates visualizations: confusion matrix heatmap and feature importance bar chart.

**Performance:** ~98% accuracy, 98% precision, 97% recall

---

### 2. Regression - House Price Prediction
**File:** `C-G01-final-regression.R`

**Overview:** Predicts median house values in California using Linear Regression.

**Code Explanation:**
- **Data Collection (A):** Downloads California Housing dataset from Kaggle using kagglehub package.
- **Data Understanding (B):** Shows dataset dimensions and missing values. Creates EDA visualizations: house value distribution, income distribution, income vs price scatter plot, price boxplot.
- **Data Preprocessing (C):** Separates target (median_house_value) from features. Removes categorical column (ocean_proximity). Imputes missing values with median. Engineers new feature (rooms_per_household = total_rooms / households). Normalizes all features using z-score. Removes incomplete rows.
- **Modeling (D):** Splits data 70/30 train/test. Trains Linear Regression model.
- **Evaluation (E):** Predicts on test set. Calculates RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), R² (coefficient of determination). Shows top 10 influential features. Creates visualizations: actual vs predicted scatter, residuals plot, residual distribution histogram, Q-Q plot for normality check.

**Metrics:** RMSE, MAE, R²

---

### 3. Clustering - Customer Segmentation
**File:** `C-G01-final-clustering.R`

**Overview:** Segments mall customers into groups based on demographics and spending behavior using K-Means.

**Code Explanation:**
- **Data Collection (A):** Downloads Mall Customers dataset from Kaggle.
- **Data Understanding (B):** Displays dataset size and missing values. Creates EDA visualizations: gender distribution, age histogram, income histogram, spending score histogram.
- **Data Preprocessing (C):** Removes ID column. Encodes Gender as binary (Male=1, Female=0). Normalizes all features using z-score. Removes incomplete rows.
- **Modeling (D):** Finds optimal number of clusters using Elbow Method (within-cluster sum of squares) and Silhouette Method. Selects best K based on highest silhouette score. Trains final K-Means model with optimal clusters.
- **Evaluation (E):** Shows cluster sizes and distribution. Calculates Silhouette Score (cluster quality) and Variance Explained. Displays cluster profiles with mean values for each feature per cluster. Creates visualizations: cluster size bar chart, silhouette values plot, elbow curve, PCA-reduced 2D cluster plot.

**Metrics:** Silhouette Score, Variance Explained

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
