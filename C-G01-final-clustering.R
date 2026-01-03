################################################################################
# Clustering Project - Customer Segmentation
# Dataset: Mall Customers (Google Drive)
################################################################################

cat("CLUSTERING PROJECT\n\n")
set.seed(42)

# A. DATA COLLECTION
cat("Loading dataset...\n")
if (file.exists("/tmp/mall_customers.csv")) {
  df_data <- read.csv("/tmp/mall_customers.csv")
} else {
  file_id <- "1Ew4bXJ0nJ5TqNZDcDNZYT7MoWZN9VnmF"
  url <- paste0("https://drive.google.com/uc?export=download&id=", file_id)
  temp_file <- tempfile(fileext = ".csv")
  download.file(url, temp_file, mode = "wb", quiet = TRUE)
  df_data <- read.csv(temp_file)
  write.csv(df_data, "/tmp/mall_customers.csv", row.names = FALSE)
}
cat("Dataset loaded\n\n")

# B. DATA UNDERSTANDING
cat("Dataset:", nrow(df_data), "rows,", ncol(df_data), "columns\n")
cat("Missing values:", sum(is.na(df_data)), "\n\n")

pdf("clustering_plots.pdf", width = 10, height = 8)
par(mfrow = c(2, 2))
if ("Gender" %in% names(df_data)) {
  barplot(table(df_data$Gender), main = "Gender Distribution", col = c("pink", "lightblue"))
}
if ("Age" %in% names(df_data)) {
  hist(df_data$Age, main = "Age Distribution", xlab = "Age", col = "steelblue", breaks = 20)
}
if ("Annual.Income..k.." %in% names(df_data)) {
  hist(df_data$Annual.Income..k.., main = "Income Distribution", 
       xlab = "Income", col = "lightgreen", breaks = 20)
}
if ("Spending.Score..1.100." %in% names(df_data)) {
  hist(df_data$Spending.Score..1.100., main = "Spending Score Distribution", 
       xlab = "Score", col = "coral", breaks = 20)
}
dev.off()
par(mfrow = c(1, 1))

# C. DATA PREPROCESSING
cat("Preprocessing...\n")
df_processed <- df_data[, !(names(df_data) %in% c("CustomerID"))]

if ("Gender" %in% names(df_processed)) {
  df_processed$Gender_Encoded <- ifelse(df_processed$Gender == "Male", 1, 0)
  df_processed <- df_processed[, !(names(df_processed) == "Gender")]
}

df_scaled <- as.data.frame(scale(df_processed))
df_scaled <- df_scaled[complete.cases(df_scaled), ]
cat("Preprocessed:", nrow(df_scaled), "rows,", ncol(df_scaled), "features\n\n")

# D. MODELING
cat("Finding optimal clusters...\n")
wss <- sapply(1:10, function(k) {
  kmeans(df_scaled, centers = k, nstart = 25)$tot.withinss
})

if (!require("cluster", quietly = TRUE)) {
  install.packages("cluster", repos = "http://cran.rstudio.com/", quiet = TRUE)
  library(cluster)
}

sil_scores <- sapply(2:10, function(k) {
  km <- kmeans(df_scaled, centers = k, nstart = 25)
  mean(silhouette(km$cluster, dist(df_scaled))[, 3])
})

optimal_k <- which.max(sil_scores) + 1
cat(sprintf("Optimal clusters: %d\n\n", optimal_k))

cat("Training K-Means...\n")
final_model <- kmeans(df_scaled, centers = optimal_k, nstart = 25)
df_data$Cluster <- as.factor(final_model$cluster)
cat("Model trained\n\n")

# E. EVALUATION
cat("=== RESULTS ===\n")
cat("Cluster sizes:\n")
print(table(df_data$Cluster))

sil <- silhouette(final_model$cluster, dist(df_scaled))
avg_sil <- mean(sil[, 3])
var_explained <- (final_model$betweenss / final_model$totss) * 100

cat(sprintf("\nSilhouette Score: %.4f\n", avg_sil))
cat(sprintf("Variance Explained: %.2f%%\n\n", var_explained))

cat("Cluster Profiles:\n")
for (i in 1:optimal_k) {
  cat(sprintf("\nCluster %d (n=%d):\n", i, sum(df_data$Cluster == i)))
  cluster_data <- df_data[df_data$Cluster == i, ]
  numeric_cols <- names(cluster_data)[sapply(cluster_data, is.numeric)]
  for (col in numeric_cols) {
    if (!grepl("CustomerID", col)) {
      cat(sprintf("  %s: %.2f\n", col, mean(cluster_data[[col]], na.rm = TRUE)))
    }
  }
}

pdf("clustering_results.pdf", width = 10, height = 8)

par(mfrow = c(2, 2))
barplot(table(df_data$Cluster), main = "Cluster Sizes", 
        xlab = "Cluster", ylab = "Count", col = rainbow(optimal_k))

plot(sil[, 3], main = "Silhouette Values", 
     xlab = "Sample", ylab = "Silhouette", pch = 20, 
     col = df_data$Cluster)
abline(h = avg_sil, col = "red", lwd = 2, lty = 2)

plot(wss, type = "b", main = "Elbow Method", 
     xlab = "Number of Clusters", ylab = "Within-cluster SS", 
     pch = 19, col = "blue")

pca <- prcomp(df_scaled)
plot(pca$x[, 1], pca$x[, 2], col = df_data$Cluster, pch = 20, 
     main = "Clusters (PCA)", xlab = "PC1", ylab = "PC2")
legend("topright", legend = 1:optimal_k, col = 1:optimal_k, pch = 20)

dev.off()
par(mfrow = c(1, 1))
cat("\nVisualizations saved\n")
