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

---

## Frequently Asked Questions (FAQ)

### Project Overview Questions

**Q1: What did you implement in this project?**
- We implemented three different machine learning tasks on network traffic data:
  1. **Classification:** Identifying malicious vs benign network traffic using Random Forest
  2. **Regression:** Predicting network flow characteristics using Linear Regression
  3. **Clustering:** Discovering patterns and grouping similar network traffic using K-Means

**Q2: Why did you choose these three specific tasks?**
- To demonstrate the three main paradigms of machine learning:
  - **Supervised Classification:** Learning from labeled data to categorize
  - **Supervised Regression:** Learning from labeled data to predict continuous values
  - **Unsupervised Clustering:** Finding patterns in unlabeled data

**Q3: What algorithms did you use for each task and why?**
- **Classification → Random Forest:** Handles high-dimensional data well, robust to overfitting, provides feature importance, ensemble method with high accuracy
- **Regression → Linear Regression:** Simple, interpretable, shows direct relationships between features and target, computationally efficient
- **Clustering → K-Means:** Fast, scalable, easy to understand, works well with normalized data, widely used in practice

**Q4: What dataset did you use and why?**
- Network traffic dataset with 26,800 flows and 78 features. Chosen because:
  - Real-world data from actual network traffic
  - Large enough for meaningful analysis
  - Rich features (packet sizes, timing, protocols)
  - Suitable for all three tasks (classification, regression, clustering)
  - Publicly accessible via Google Drive

---

### General Questions

**Q5: What tools and technologies did you use?**
- **Language:** R programming language
- **Packages:** randomForest (classification), cluster (clustering), base R (regression and visualization)
- **IDE:** RStudio or command-line Rscript
- **Visualization:** Base R graphics, PDF output
- **Data Source:** Google Drive for dataset hosting

**Q6: What is the main objective of this project?**
- To demonstrate the complete data science lifecycle (A-E) on real-world network traffic data using three different machine learning paradigms: supervised classification, supervised regression, and unsupervised clustering.

**Q7: Why did you use the same dataset for all three tasks?**
- Using the same dataset ensures consistency and allows direct comparison of different analytical approaches. Network traffic data is versatile enough to support classification (identifying malicious traffic), regression (predicting flow characteristics), and clustering (discovering traffic patterns).

**Q8: What is the data science lifecycle (A-E)?**
- **A. Data Collection:** Gathering data from sources (Google Drive in our case)
- **B. Data Understanding:** Exploring data characteristics, distributions, relationships through EDA
- **C. Data Preprocessing:** Cleaning, normalizing, handling missing values, feature engineering
- **D. Modeling:** Selecting and training appropriate algorithms for the task
- **E. Evaluation:** Measuring performance using appropriate metrics and interpreting results

**Q9: How large is the dataset and what does it contain?**
- 26,800 network flow records with 78 features including packet counts, byte sizes, timing information (IAT - Inter-Arrival Time), flow duration, protocol information, and traffic labels.

**Q10: What outputs does each script produce?**
- Each script generates:
  - Console output with metrics and results
  - Two PDF files: one for exploratory data analysis (EDA) plots and one for model results/evaluation visualizations
  - Classification: 98.66% accuracy, confusion matrix, feature importance
  - Regression: RMSE, MAE, R², residual plots
  - Clustering: Silhouette score, cluster profiles, PCA visualization

---

### Classification Questions

**Q11: Why did you choose Random Forest for classification?**
- Random Forest is robust to overfitting, handles high-dimensional data well, provides feature importance rankings, and works effectively with both numerical features. It's an ensemble method that combines multiple decision trees for better accuracy.

**Q6: What is binary classification and why convert multi-class to binary?**
- Binary classification predicts between two classes. We converted the multi-class labels (7 different attack types) to binary (Benign=0, Malicious=1) to simplify the problem and achieve higher accuracy by focusing on detecting any type of attack vs normal traffic.

**Q7: What do the evaluation metrics mean?**
- **Accuracy (98.66%):** Percentage of correctly classified instances
- **Precision (98.37%):** Of all predicted malicious traffic, 98.37% were actually malicious (low false positives)
- **Recall (97.36%):** Of all actual malicious traffic, 97.36% were correctly detected (low false negatives)
- **F1-Score (97.86%):** Harmonic mean of precision and recall, balancing both metrics

**Q8: What is a confusion matrix?**
- A table showing true positives, true negatives, false positives, and false negatives. It visualizes classification performance across classes.

**Q9: What is feature importance in Random Forest?**
- It measures how much each feature contributes to reducing impurity (Gini index) across all trees. Features with higher importance are more influential in making predictions. In our case, `Init_Win_bytes_forward` was most important.

**Q10: What does 50 trees (ntree=50) mean?**
- Random Forest builds 50 independent decision trees. Each tree votes on the classification, and the majority vote determines the final prediction. More trees generally improve accuracy but increase computation time.

---

### Regression Questions

**Q11: Why use Linear Regression?**
- Linear Regression is interpretable, computationally efficient, and works well for understanding linear relationships between features and target variables. It provides coefficients showing each feature's contribution.

**Q12: What does the warning "rank-deficient fit" mean?**
- Some features are perfectly correlated (multicollinearity), making the model unable to estimate unique coefficients for all predictors. This doesn't prevent predictions but some coefficients may be unreliable.

**Q13: What do the regression metrics mean?**
- **RMSE (1.4084):** Root Mean Squared Error - average prediction error in the same units as the target. Lower is better.
- **MAE (1.0283):** Mean Absolute Error - average absolute difference between predicted and actual values. More robust to outliers than RMSE.
- **R² (0.4935):** Coefficient of determination - 49.35% of variance in the target is explained by the model. Ranges from 0 to 1, where 1 is perfect prediction.

**Q14: Why is R² only 0.49? Is that bad?**
- R² of 0.49 means moderate predictive power. In complex real-world network traffic data with many confounding factors, this is reasonable. It indicates the model captures significant patterns but there's room for improvement with feature engineering or non-linear models.

**Q15: What are residuals and why analyze them?**
- Residuals are differences between actual and predicted values. We analyze them to check assumptions: they should be normally distributed, have constant variance (homoscedasticity), and show no patterns. Our Q-Q plot and residual histogram help verify these assumptions.

**Q16: What is feature engineering?**
- Creating new features from existing ones to improve model performance. In the preprocessing, we create derived features by handling ratios and interactions between variables.

---

### Clustering Questions

**Q17: Why use K-Means clustering?**
- K-Means is simple, fast, scalable, and effective for spherical clusters. It's intuitive (groups similar points together) and works well with normalized data. It's the most popular clustering algorithm for exploratory analysis.

**Q18: How do you determine the optimal number of clusters?**
- We use two methods:
  - **Elbow Method:** Plot within-cluster sum of squares (WSS) vs K. Choose K where the curve "bends" (diminishing returns)
  - **Silhouette Method:** Measures how similar points are to their own cluster vs other clusters. Choose K with highest average silhouette score.

**Q19: What is the Silhouette Score (0.9491)?**
- Ranges from -1 to 1. Values close to 1 indicate points are well-matched to their clusters and poorly matched to neighboring clusters. Our score of 0.95 indicates excellent clustering quality.

**Q20: What does "Variance Explained: 21.58%" mean?**
- The clustering explains 21.58% of total variance in the data. This is the ratio of between-cluster variance to total variance. Higher percentages indicate more distinct, separated clusters.

**Q21: Why sample to 5,000 rows for clustering?**
- K-Means computational complexity is O(n*k*i*d) where n=samples, k=clusters, i=iterations, d=dimensions. With 26,800 rows and 78 dimensions, computation becomes slow. Sampling maintains statistical properties while improving speed.

**Q22: What is PCA visualization?**
- Principal Component Analysis reduces 78 dimensions to 2D for visualization. PC1 and PC2 capture the most variance, allowing us to visualize cluster separation in 2D space.

**Q23: What are cluster profiles?**
- Average feature values for each cluster, showing what makes each cluster unique. For example, Cluster 1 might represent normal traffic patterns while Cluster 2 represents unusual patterns.

---

### Data Preprocessing Questions

**Q24: Why normalize/standardize features?**
- Z-score normalization ((x - mean) / std) scales features to mean=0, std=1. This prevents features with large ranges from dominating distance calculations in algorithms like K-Means and helps gradient descent converge faster.

**Q25: How do you handle missing and infinite values?**
- We use median imputation: replace missing/infinite values with the median of valid values. Median is robust to outliers. If all values are invalid, we use 0.

**Q26: Why use 70/30 train-test split?**
- 70% for training provides enough data to learn patterns. 30% for testing ensures reliable performance evaluation. This is a common standard balancing bias-variance tradeoff.

**Q27: What is the difference between train and test sets?**
- **Train set:** Used to fit/train the model (learn patterns)
- **Test set:** Held-out data used only for evaluation (measure generalization to unseen data)
- Never train on test data to avoid overfitting and optimistic performance estimates.

**Q28: Why set seed (set.seed(42))?**
- Ensures reproducibility. Random operations (train-test split, K-Means initialization) will produce the same results every run, making debugging easier and results verifiable.

---

### Technical Implementation Questions

**Q29: Why cache the dataset in /tmp/?**
- Downloading 24MB from Google Drive takes time. Caching avoids repeated downloads during testing and development, speeding up subsequent runs.

**Q30: What packages did you use and why?**
- **randomForest:** Implements Random Forest algorithm for classification
- **cluster:** Provides silhouette analysis and clustering utilities
- **Base R:** Used for data manipulation, visualization (base graphics), statistical functions, and linear regression (lm function)

**Q31: Why use PDF for visualizations instead of PNG?**
- PDFs are vector graphics (scalable without quality loss), support multiple pages, smaller file sizes for complex plots, and preferred in academic/professional settings.

**Q32: Could you use other algorithms?**
- Yes! Alternatives include:
  - **Classification:** SVM, Neural Networks, Gradient Boosting, Naive Bayes
  - **Regression:** Ridge/Lasso Regression, Random Forest Regressor, SVR, Neural Networks
  - **Clustering:** DBSCAN, Hierarchical Clustering, Gaussian Mixture Models
- We chose simpler, interpretable algorithms suitable for demonstrating the data science lifecycle.

**Q33: What would you do to improve the models?**
- Feature selection (remove irrelevant features)
- Feature engineering (create interaction terms, polynomial features)
- Hyperparameter tuning (grid search, cross-validation)
- Try ensemble methods or deep learning
- Collect more diverse training data
- Address class imbalance if present

---

## Key Takeaways

1. **Complete Lifecycle:** Successfully implemented all stages (A-E) for three different ML paradigms
2. **High Performance:** Achieved 98.66% accuracy in classification, demonstrating effective malicious traffic detection
3. **Interpretability:** Used interpretable models with feature importance and cluster profiles for explainability
4. **Automation:** Fully automated pipeline from data loading to visualization generation
5. **Best Practices:** Proper train-test splitting, normalization, reproducibility (set.seed), and comprehensive evaluation metrics

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
