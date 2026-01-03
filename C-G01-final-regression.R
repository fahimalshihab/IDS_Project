################################################################################
# Regression Project - House Price Prediction
# Dataset: California Housing (Kaggle)
################################################################################

cat("REGRESSION PROJECT\n\n")
set.seed(42)

# A. DATA COLLECTION
cat("Loading dataset...\n")
if (!require("kagglehub", quietly = TRUE)) {
  install.packages("kagglehub", repos = "http://cran.rstudio.com/", quiet = TRUE)
  library(kagglehub)
}

path <- kagglehub::dataset_download("camnugent/california-housing-prices")
csv_file <- list.files(path, pattern = "\\.csv$", full.names = TRUE)[1]
df_data <- read.csv(csv_file)
cat("Dataset loaded\n\n")

# B. DATA UNDERSTANDING
cat("Dataset:", nrow(df_data), "rows,", ncol(df_data), "columns\n")
cat("Missing values:", sum(is.na(df_data)), "\n\n")

pdf("regression_plots.pdf", width = 10, height = 8)
par(mfrow = c(2, 2))
hist(df_data$median_house_value, main = "House Value Distribution", 
     xlab = "Price", col = "steelblue", breaks = 30)
hist(df_data$median_income, main = "Income Distribution", 
     xlab = "Income", col = "lightgreen", breaks = 30)
plot(df_data$median_income, df_data$median_house_value, 
     main = "Income vs Price", xlab = "Income", ylab = "Price", 
     pch = 20, col = rgb(0, 0, 1, 0.3))
boxplot(df_data$median_house_value, main = "Price Boxplot", 
        ylab = "Price", col = "coral")
dev.off()
par(mfrow = c(1, 1))

# C. DATA PREPROCESSING
cat("Preprocessing...\n")
y <- df_data$median_house_value
X <- df_data[, !(names(df_data) %in% c("median_house_value", "ocean_proximity"))]

for (col in names(X)) {
  if (sum(is.na(X[[col]])) > 0) {
    X[[col]][is.na(X[[col]])] <- median(X[[col]], na.rm = TRUE)
  }
}

if ("total_rooms" %in% names(X) && "households" %in% names(X)) {
  X$rooms_per_hh <- X$total_rooms / X$households
  X$rooms_per_hh[is.infinite(X$rooms_per_hh)] <- 0
}

X_scaled <- as.data.frame(scale(X))
valid <- complete.cases(X_scaled) & !is.na(y)
X <- X_scaled[valid, ]
y <- y[valid]
cat("Preprocessed:", nrow(X), "rows,", ncol(X), "features\n\n")

# D. MODELING
cat("Splitting data (70/30)...\n")
n <- nrow(X)
train_idx <- sample(1:n, size = floor(0.7 * n))
X_train <- X[train_idx, ]
X_test <- X[-train_idx, ]
y_train <- y[train_idx]
y_test <- y[-train_idx]
cat("Train:", length(y_train), "| Test:", length(y_test), "\n\n")

cat("Training model...\n")
train_data <- data.frame(X_train, price = y_train)
model <- lm(price ~ ., data = train_data)
cat("Model trained\n\n")

# E. EVALUATION
y_pred <- predict(model, X_test)

rmse <- sqrt(mean((y_test - y_pred)^2))
mae <- mean(abs(y_test - y_pred))
r2 <- cor(y_test, y_pred)^2

cat("=== RESULTS ===\n")
cat(sprintf("RMSE: %.2f\n", rmse))
cat(sprintf("MAE:  %.2f\n", mae))
cat(sprintf("R²:   %.4f\n\n", r2))

cat("Top 10 Features:\n")
coefs <- coef(model)[-1]
coefs_sorted <- sort(abs(coefs), decreasing = TRUE)
print(head(coefs_sorted, 10))

pdf("regression_results.pdf", width = 10, height = 8)

par(mfrow = c(2, 2))
plot(y_test, y_pred, main = "Actual vs Predicted", 
     xlab = "Actual Price", ylab = "Predicted Price", 
     pch = 20, col = rgb(0, 0, 1, 0.3))
abline(0, 1, col = "red", lwd = 2)

residuals <- y_test - y_pred
plot(y_pred, residuals, main = "Residuals Plot", 
     xlab = "Predicted", ylab = "Residuals", 
     pch = 20, col = rgb(0, 0, 0, 0.3))
abline(h = 0, col = "red", lwd = 2)

hist(residuals, main = "Residuals Distribution", 
     xlab = "Residuals", col = "lightblue", breaks = 30)

qqnorm(residuals, pch = 20, col = rgb(0, 0, 0, 0.3))
qqline(residuals, col = "red", lwd = 2)

dev.off()
par(mfrow = c(1, 1))
cat("\nVisualizations saved\n")
