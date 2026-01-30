# Load required libraries
library(jsonlite)

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("Please provide the path to the sample JSON file")
}
sample_path <- args[1]

# Load outlier results
outlier_results <- fromJSON("tmp/outlier_results.json")

# Print outlier detection results first
cat("\nOutlier Detection Results:\n")
cat("------------------------\n")
for (cls in names(outlier_results)) {
  cat(sprintf("\n%s:\n", cls))
  cat(sprintf("  Is Outlier: %s\n", outlier_results[[cls]]$is_outlier))
  cat(sprintf("  Isolation Forest Score: %.4f\n", outlier_results[[cls]]$iso_forest_score))
  cat(sprintf("  LOF Score: %.4f\n", outlier_results[[cls]]$lof_score))
}

# Check if sample is an outlier for all classes
is_outlier_all <- all(sapply(outlier_results, function(x) x$is_outlier))

if (is_outlier_all) {
  cat("\nSample is an outlier for all classes. Skipping prediction.\n")
  quit(status = 1)
} else {
  cat("\nSample is not an outlier for all classes. Proceeding with prediction.\n")
}

# Load model and predictions
# This model file is sourced from the original AIPAL repository
# https://github.com/VincentAlcazer/AIPAL
# Licensed under MIT
res_list <- readRDS("aipal_validation/r/221003_Final_model_res_list.rds")
model <- res_list$final_model

# Read sample data
sample_data <- jsonlite::fromJSON(sample_path)

# Prediction function
predict_sample <- function(new_data) {
  # Calculate Monocytes_percent if not present
  if (!"Monocytes_percent" %in% names(new_data)) {
    new_data$Monocytes_percent <- new_data$Monocytes_G_L * 100 / new_data$WBC_G_L
  }

  # Make prediction
  prediction <- predict(model, newdata = new_data, type = "prob", na.action = na.pass)

  # Get predicted class and probability
  pred_class <- colnames(prediction)[which.max(prediction)]
  pred_prob <- max(prediction)

  # Create result list
  result <- list(
    prediction = prediction,
    predicted_class = pred_class,
    predicted_probability = pred_prob
  )

  return(result)
}

# Run prediction
result <- predict_sample(sample_data)

# Print results
cat("\nPrediction Results:\n")
cat("------------------\n")
cat(sprintf("Predicted Class: %s\n", result$predicted_class))
cat(sprintf("Predicted Probability: %.4f\n", result$predicted_probability))

cat("\nClass Probabilities:\n")
cat("------------------\n")
print(result$prediction)

# Combined Interpretation
cat("\nCombined Analysis:\n")
cat("------------------\n")

# Get the predicted class
pred_class <- result$predicted_class
pred_prob <- result$predicted_probability

# Check if the predicted class is an outlier
pred_class_outlier <- outlier_results[[pred_class]]$is_outlier
pred_class_iso_score <- outlier_results[[pred_class]]$iso_forest_score
pred_class_lof_score <- outlier_results[[pred_class]]$lof_score

# Print interpretation
cat(sprintf("\nPredicted Class (%s):\n", pred_class))
cat(sprintf("  Probability: %.4f\n", pred_prob))
cat(sprintf("  Outlier Status: %s\n", ifelse(pred_class_outlier, "Outlier", "Normal")))
cat(sprintf("  Isolation Forest Score: %.4f\n", pred_class_iso_score))
cat(sprintf("  LOF Score: %.4f\n", pred_class_lof_score))

# Additional analysis for other classes
cat("\nOther Classes Analysis:\n")
for (cls in names(outlier_results)) {
  if (cls != pred_class) {
    cat(sprintf("\n%s:\n", cls))
    cat(sprintf("  Probability: %.4f\n", result$prediction[cls]))
    cat(sprintf("  Outlier Status: %s\n", ifelse(outlier_results[[cls]]$is_outlier, "Outlier", "Normal")))
    cat(sprintf("  Isolation Forest Score: %.4f\n", outlier_results[[cls]]$iso_forest_score))
    cat(sprintf("  LOF Score: %.4f\n", outlier_results[[cls]]$lof_score))
  }
}

# Overall assessment
cat("\nOverall Assessment:\n")
if (pred_class_outlier) {
  cat("  ⚠️  Warning: The predicted class is marked as an outlier.\n")
  cat("  Consider this prediction with caution.\n")
} else {
  cat("  ✓ The predicted class is within normal range.\n")
}

# Check for competing predictions
high_prob_classes <- names(result$prediction)[result$prediction > 0.3]
if (length(high_prob_classes) > 1) {
  cat("\n  ⚠️  Note: Multiple classes have significant probabilities (>0.3):\n")
  for (cls in high_prob_classes) {
    cat(sprintf("    - %s: %.4f\n", cls, result$prediction[cls]))
  }
}
