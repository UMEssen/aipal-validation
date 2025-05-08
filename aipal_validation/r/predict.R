# Load required libraries
library(dplyr)
library(tidyr)
library(yaml)

sessionInfo()

# Load model, predictions, config
config <- yaml.load_file("aipal_validation/config/config_training.yaml")
res_list <- readRDS("aipal_validation/r/221003_Final_model_res_list.rds")
model <- res_list$final_model
root_dir <- sub("^/", "", config$root_dir)
root_dir <- normalizePath(file.path(getwd(), "..", root_dir))

working_dir <- file.path(root_dir, config$run_id, config$task)
new_data <- read.csv(file.path(working_dir, "/samples.csv"))

# Check and replace NaN and Inf values with NA
new_data[sapply(new_data, is.numeric)] <- sapply(new_data[sapply(new_data, is.numeric)], function(x) replace(x, is.nan(x) | is.infinite(x), NA))

# Add calculated column for Monocytes_percent
new_data$Monocytes_percent <- new_data$Monocytes_G_L * 100 / new_data$WBC_G_L

# Replace any resulting Inf or NaN in calculations (if any)
new_data$Monocytes_percent[is.infinite(new_data$Monocytes_percent) | is.nan(new_data$Monocytes_percent)] <- NA

# Convert 'Lymphocytes_G_L' to numeric (only relevant as fraction dataset does not include this observation)
new_data$Lymphocytes_G_L <- as.numeric(new_data$Lymphocytes_G_L)

# Count number of missing values in each column
missing_values <- sapply(new_data, function(x) sum(is.na(x)))
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
print(paste0("Saving predictions to ", working_dir, "/predict.csv"))
write.csv(new_data, file = paste0(working_dir, "/predict.csv"), row.names = FALSE)
