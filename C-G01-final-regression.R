################################################################################
# Regression Project - House Price Prediction
# Dataset: California Housing (Google Drive)
################################################################################

cat("REGRESSION PROJECT\n\n")
set.seed(42)

# A. DATA COLLECTION
cat("Loading dataset...\n")
if (file.exists("/tmp/test_dataset.csv")) {
  df_data <- read.csv("/tmp/test_dataset.csv")
} else {
  file_id <- "1t4dovWScXCYnduCb8pSFF-XTQJA4iBwV"
  url <- paste0("https://drive.google.com/uc?export=download&id=", file_id)
  temp_file <- tempfile(fileext = ".csv")
  download.file(url, temp_file, mode = "wb", quiet = TRUE)
  df_data <- read.csv(temp_file)
}
cat("Dataset loaded\n\n")

# B. DATA UNDERSTANDING
cat("Dataset:", nrow(df_data), "rows,", ncol(df_data), "columns\n")
cat("Missing values:", sum(is.na(df_data)), "\n\n")

pdf("regression_plots.pdf", width = 10, height = 8)
par(mfrow = c(2, 2))
numeric_cols <- names(df_data)[sapply(df_data, is.numeric)][1:4]
for (col in numeric_cols) {
  hist(df_data[[col]], main = paste("Distribution:", col), 
       xlab = col, col = "steelblue", breaks = 30)
}
dev.off()
par(mfrow = c(1, 1))

# C. DATA PREPROCESSING
cat("Preprocessing...\n")
numeric_cols <- sapply(df_data, is.numeric)
if ("Label" %in% names(df_data)) {
  y <- as.numeric(factor(df_data$Label))
  X <- df_data[, numeric_cols & names(df_data) != "Label"]
} else {
  y <- df_data[[ncol(df_data)]]
  X <- df_data[, -ncol(df_data)]
  X <- X[, sapply(X, is.numeric)]
}

for (col in names(X)) {
  col_vals <- X[[col]]
  finite_vals <- col_vals[is.finite(col_vals)]
  if (length(finite_vals) > 0) {
    replacement <- median(finite_vals, na.rm = TRUE)
  } else {
    replacement <- 0
  }
  col_vals[is.na(col_vals) | is.infinite(col_vals)] <- replacement
  X[[col]] <- col_vals
}

X_scaled <- as.data.frame(lapply(X, function(col) {
  col_sd <- sd(col, na.rm = TRUE)
  col_mean <- mean(col, na.rm = TRUE)
  if (col_sd > 0) (col - col_mean) / col_sd else col - col_mean
}))

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
cat(sprintf("RMSE: %.4f\n", rmse))
cat(sprintf("MAE:  %.4f\n", mae))
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
