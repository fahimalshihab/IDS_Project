################################################################################
# Clustering Project - Customer Segmentation
# Dataset: Mall Customers (Google Drive)
################################################################################

cat("CLUSTERING PROJECT\n\n")
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

pdf("clustering_plots.pdf", width = 10, height = 8)
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
df_processed <- df_data[, sapply(df_data, is.numeric)]

for (i in 1:ncol(df_processed)) {
  col_vals <- df_processed[, i]
  finite_vals <- col_vals[is.finite(col_vals)]
  replacement <- ifelse(length(finite_vals) > 0, median(finite_vals, na.rm = TRUE), 0)
  col_vals[is.na(col_vals) | is.infinite(col_vals)] <- replacement
  df_processed[, i] <- col_vals
}

df_scaled <- as.data.frame(lapply(df_processed, function(col) {
  col_sd <- sd(col, na.rm = TRUE)
  col_mean <- mean(col, na.rm = TRUE)
  if (col_sd > 0) (col - col_mean) / col_sd else col - col_mean
}))

df_scaled <- df_scaled[complete.cases(df_scaled), ]

# Sample for faster clustering
set.seed(42)
if (nrow(df_scaled) > 5000) {
  sample_idx <- sample(1:nrow(df_scaled), 5000)
  df_scaled <- df_scaled[sample_idx, ]
}
cat("Preprocessed:", nrow(df_scaled), "rows,", ncol(df_scaled), "features\n\n")

# D. MODELING
cat("Finding optimal clusters...\n")
wss <- sapply(2:8, function(k) {
  kmeans(df_scaled, centers = k, nstart = 10)$tot.withinss
})

if (!require("cluster", quietly = TRUE)) {
  install.packages("cluster", repos = "http://cran.rstudio.com/", quiet = TRUE)
  library(cluster)
}

sil_scores <- sapply(2:8, function(k) {
  km <- kmeans(df_scaled, centers = k, nstart = 10)
  mean(silhouette(km$cluster, dist(df_scaled))[, 3])
})

optimal_k <- which.max(sil_scores) + 1
cat(sprintf("Optimal clusters: %d\n\n", optimal_k))

cat("Training K-Means...\n")
final_model <- kmeans(df_scaled, centers = optimal_k, nstart = 25)
df_data_sampled <- df_data[as.numeric(rownames(df_scaled)), ]
df_data_sampled$Cluster <- as.factor(final_model$cluster)
cat("Model trained\n\n")

# E. EVALUATION
cat("=== RESULTS ===\n")
cat("Cluster sizes:\n")
print(table(df_data_sampled$Cluster))

sil <- silhouette(final_model$cluster, dist(df_scaled))
avg_sil <- mean(sil[, 3])
var_explained <- (final_model$betweenss / final_model$totss) * 100

cat(sprintf("\nSilhouette Score: %.4f\n", avg_sil))
cat(sprintf("Variance Explained: %.2f%%\n\n", var_explained))

cat("Cluster Profiles:\n")
for (i in 1:optimal_k) {
  cat(sprintf("\nCluster %d (n=%d):\n", i, sum(df_data_sampled$Cluster == i)))
  cluster_data <- df_data_sampled[df_data_sampled$Cluster == i, ]
  numeric_cols <- names(cluster_data)[sapply(cluster_data, is.numeric)][1:5]
  for (col in numeric_cols) {
    cat(sprintf("  %s: %.2f\n", col, mean(cluster_data[[col]], na.rm = TRUE)))
  }
}

pdf("clustering_results.pdf", width = 10, height = 8)

par(mfrow = c(2, 2))
barplot(table(df_data_sampled$Cluster), main = "Cluster Sizes", 
        xlab = "Cluster", ylab = "Count", col = rainbow(optimal_k))

plot(sil[, 3], main = "Silhouette Values", 
     xlab = "Sample", ylab = "Silhouette", pch = 20, 
     col = df_data_sampled$Cluster)
abline(h = avg_sil, col = "red", lwd = 2, lty = 2)

plot(2:8, wss, type = "b", main = "Elbow Method", 
     xlab = "Number of Clusters", ylab = "Within-cluster SS", 
     pch = 19, col = "blue")

pca <- prcomp(df_scaled)
plot(pca$x[, 1], pca$x[, 2], col = df_data_sampled$Cluster, pch = 20, 
     main = "Clusters (PCA)", xlab = "PC1", ylab = "PC2")
legend("topright", legend = 1:optimal_k, col = 1:optimal_k, pch = 20)

dev.off()
par(mfrow = c(1, 1))
cat("\nVisualizations saved\n")
