# Load required libraries
library(dplyr)
library(tidyr)
library(yaml)

sessionInfo()

# Get command line arguments - config file path is optional
args <- commandArgs(trailingOnly = TRUE)
config_file <- if (length(args) > 0 && file.exists(args[1])) args[1] else "aipal_validation/config/config_training.yaml"

print(paste("Using config file:", config_file))

# Load model, predictions, config
config <- yaml.load_file(config_file)
res_list <- readRDS("aipal_validation/r/221003_Final_model_res_list.rds")
model <- res_list$final_model

# Handle different directory structures
root_dir <- config$root_dir
if (root_dir == ".") {
  # For synthetic data (current directory)
  root_dir <- getwd()
} else {
  # For production data (/data structure)
  root_dir <- sub("^/", "", root_dir)
  root_dir <- normalizePath(file.path(getwd(), "..", root_dir))
}

working_dir <- file.path(root_dir, config$run_id, config$task)
print(paste("Working directory:", working_dir))
print(paste("Looking for samples.csv at:", file.path(working_dir, "samples.csv")))

new_data <- read.csv(file.path(working_dir, "samples.csv"))

print(paste("Loaded", nrow(new_data), "samples with", ncol(new_data), "columns"))

# Check and replace NaN and Inf values with NA
new_data[sapply(new_data, is.numeric)] <- sapply(new_data[sapply(new_data, is.numeric)], function(x) replace(x, is.nan(x) | is.infinite(x), NA))

# Ensure required columns exist when some features were dropped (e.g., Monocytes_G_L)
if (!("Monocytes_G_L" %in% names(new_data))) {
  new_data$Monocytes_G_L <- NA_real_
}
if (!("WBC_G_L" %in% names(new_data))) {
  new_data$WBC_G_L <- NA_real_
}

# Add calculated column for Monocytes_percent (robust to missing inputs)
new_data$Monocytes_percent <- NA_real_
if ("Monocytes_G_L" %in% names(new_data) && "WBC_G_L" %in% names(new_data)) {
  suppressWarnings({
    new_data$Monocytes_percent <- new_data$Monocytes_G_L * 100 / new_data$WBC_G_L
  })
}

# Replace any resulting Inf or NaN in calculations (if any)
new_data$Monocytes_percent[is.infinite(new_data$Monocytes_percent) | is.nan(new_data$Monocytes_percent)] <- NA

# Convert 'Lymphocytes_G_L' to numeric (only relevant as fraction dataset does not include this observation)
if ("Lymphocytes_G_L" %in% names(new_data)) {
  new_data$Lymphocytes_G_L <- as.numeric(new_data$Lymphocytes_G_L)
}

# Count number of missing values in each column
missing_values <- sapply(new_data, function(x) sum(is.na(x)))
print("Missing values per column:")
print(missing_values)

# Replace completely empty columns with NA and assign a consistent type (e.g., numeric)
new_data <- new_data %>% mutate(across(everything(), ~ ifelse(is.na(.) & all(is.na(.)), NA_real_, .)))

# Prediction function
predict_type <- function(new_data) {
  prediction <- predict(model, newdata = new_data, type = "prob", na.action = na.pass)
  return(prediction)
}

# Execute prediction for all rows
new_data$prediction <- predict_type(new_data)

# Save new_data with predictions to a new CSV file
output_file <- file.path(working_dir, "predict.csv")
print(paste0("Saving predictions to ", output_file))
write.csv(new_data, file = output_file, row.names = FALSE)

print("Prediction completed successfully!")
