# Load required libraries
library(xgboost)
library(dplyr)
library(caret) # For createDataPartition
library(yaml)  # For loading config

# --- Configuration ---
message("Loading configuration...")
config <- yaml.load_file("aipal_validation/config/config_training.yaml")
root_dir <- sub("^/", "", config$root_dir)
root_dir <- normalizePath(file.path(getwd(), "..", root_dir))

# When task is 'retrain' and step is 'all_cohorts'
# CLI sets config$run_id to 'all_cohorts' and config$task to 'retrain'
working_dir <- file.path(root_dir, "all_cohorts", "retrain")
data_file_path <- file.path(working_dir, "samples.csv")
output_dir <- "r"  # Save outputs to the r/ directory
model_output_name <- "retrained_xgb_model.rds"
train_split_output_name <- "pediatric_train_samples.csv"
test_split_output_name <- "pediatric_test_samples.csv"
predictions_output_name <- "predict.csv"

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# --- Load Data ---
message("Loading data from: ", data_file_path)
if (!file.exists(data_file_path)) {
  stop(paste0("Input file does not exist: ", data_file_path,
              "\nWorking directory was: ", working_dir,
              "\nCurrent directory is: ", getwd()))
}
full_data <- read.csv(data_file_path, colClasses = "character", stringsAsFactors = FALSE)

# Convert columns to appropriate types
# Identify numeric columns based on header in samples.csv
# ID,age,sex,class,MCV_fL,PT_percent,LDH_UI_L,MCHC_g_L,WBC_G_L,Fibrinogen_g_L,Monocytes_G_L,Platelets_G_L,Lymphocytes_G_L,Monocytes_percent,prediction.base.ALL,prediction.base.AML,prediction.base.APL,city_country
numeric_cols <- c("age", "MCV_fL", "PT_percent", "LDH_UI_L", "MCHC_g_L", "WBC_G_L",
                  "Fibrinogen_g_L", "Monocytes_G_L", "Platelets_G_L", "Lymphocytes_G_L")

for (col_name in numeric_cols) {
  if (col_name %in% names(full_data)) {
    full_data[[col_name]] <- suppressWarnings(as.numeric(as.character(full_data[[col_name]])))
  }
}
message("Data loaded. Dimensions: ", paste(dim(full_data), collapse = " x "))

# --- Filter for Pediatric Cohort ---
message("Filtering for pediatric cohort (age < 18)...")
pediatric_data <- full_data %>%
  filter(!is.na(age) & age < 18)

if (nrow(pediatric_data) == 0) {
  stop("No pediatric samples found (age < 18). Please check the 'age' column and data.")
}
message("Pediatric data dimensions: ", paste(dim(pediatric_data), collapse = " x "))

# --- Preprocessing ---
message("Starting preprocessing...")
# Calculate Monocytes_percent
pediatric_data <- pediatric_data %>%
  mutate(Monocytes_percent = ifelse(WBC_G_L > 0, (Monocytes_G_L * 100) / WBC_G_L, NA_real_))

# Handle NaN, Inf globally for numeric columns by converting to NA
for (col_name in names(pediatric_data)) {
  if (is.numeric(pediatric_data[[col_name]])) {
    pediatric_data[[col_name]][is.nan(pediatric_data[[col_name]]) | is.infinite(pediatric_data[[col_name]])] <- NA
  }
}

# --- Define Target Variable ---
message("Defining target variable from 'class' column...")
if (!"class" %in% names(pediatric_data)) {
  stop("The required 'class' column for target variable is not found in the data.")
}

# Assuming 'class' column contains strings like "ALL", "AML", "APL" (case-insensitive)
# Mapping these to 0-indexed classes for XGBoost (0: ALL, 1: AML, 2: APL)
pediatric_data$pediatric_class_target <- dplyr::case_when(
  toupper(pediatric_data$class) == "ALL" ~ 0L,
  toupper(pediatric_data$class) == "AML" ~ 1L,
  toupper(pediatric_data$class) == "APL" ~ 2L,
  TRUE ~ NA_integer_
)

# Check for unmapped target values
if (any(is.na(pediatric_data$pediatric_class_target) & !is.na(pediatric_data$class))) {
  unmapped_classes <- unique(pediatric_data$class[is.na(pediatric_data$pediatric_class_target) & !is.na(pediatric_data$class)])
  warning(paste("Some values in 'class' column were not mapped to a target class:", paste(unmapped_classes, collapse=", "), ". These will be treated as NA targets."))
}

if (all(is.na(pediatric_data$pediatric_class_target))) {
  warning("Target variable 'pediatric_class_target' is all NA after mapping from 'class' column. Check 'class' column content and mapping logic. XGBoost training will likely fail.")
} else {
   message("Number of samples for each mapped target class (0=ALL, 1=AML, 2=APL):")
   print(table(pediatric_data$pediatric_class_target, useNA = "ifany"))
}

# Remove rows with NA in the target variable as they cannot be used for training/testing
pediatric_data_clean <- pediatric_data %>% filter(!is.na(pediatric_class_target))

if (nrow(pediatric_data_clean) == 0) {
  stop("No samples remaining after filtering for valid target variable from 'class' column. XGBoost training cannot proceed.")
}
message(paste("Dimensions after removing NA targets:", paste(dim(pediatric_data_clean), collapse = " x ")))


# --- Identify Predictor Features ---
message("Identifying and preparing predictor features...")
# Explicitly define the list of feature names to use for the model
# 'Monocytes_percent' is calculated during preprocessing.
model_feature_names <- c("age", "MCV_fL", "PT_percent", "LDH_UI_L", "MCHC_g_L",
                         "WBC_G_L", "Fibrinogen_g_L", "Monocytes_G_L",
                         "Platelets_G_L", "Lymphocytes_G_L", "Monocytes_percent")

# Ensure all selected features exist in the data and are numeric
final_features <- character(0)
for (feat_name in model_feature_names) {
  if (!feat_name %in% names(pediatric_data_clean)) {
    warning(paste0("Feature '", feat_name, "' is specified but not found in the processed data. It will be skipped."))
    next
  }
  if (!is.numeric(pediatric_data_clean[[feat_name]])) {
    original_class_type <- class(pediatric_data_clean[[feat_name]])[1]
    warning(paste0("Feature '", feat_name, "' is not numeric (type: ", original_class_type, "). XGBoost requires numeric features. Attempting conversion."))
    # Ensure conversion handles factors or characters appropriately
    pediatric_data_clean[[feat_name]] <- suppressWarnings(as.numeric(as.character(pediatric_data_clean[[feat_name]])))
    if (all(is.na(pediatric_data_clean[[feat_name]])) && !all(is.na(pediatric_data[[feat_name]][pediatric_data$ID %in% pediatric_data_clean$ID & !is.na(pediatric_data[[feat_name]])])) ) { # check if it became all NA due to conversion
        warning(paste0("Feature '", feat_name, "' became all NA after numeric conversion over the pediatric_data_clean subset. This might indicate issues with the data type or values for this feature."))
    }
  }
  if (all(is.na(pediatric_data_clean[[feat_name]]))) {
    warning(paste0("Feature '", feat_name, "' is entirely NA in the cleaned dataset. While XGBoost can handle NAs, this feature might not be useful."))
  }
  final_features <- c(final_features, feat_name)
}

# Filter out any features that might have been skipped (e.g. not found)
final_features <- final_features[final_features %in% names(pediatric_data_clean)]

if (length(final_features) == 0) {
  stop("No valid features available for training after selection and validation. Please check feature list and data.")
}
message("Final selected features for training: ", paste(final_features, collapse = ", "))

# The rest of the script uses 'features' variable, so assign final_features to it.
features <- final_features

# --- Split Data (80% Train, 20% Test) ---
# In the future we can do a stratified split and cross-validation
message("Splitting data into training (80%) and testing (20%) sets...")
set.seed(123) # for reproducibility
train_indices <- createDataPartition(pediatric_data_clean$pediatric_class_target, p = 0.8, list = FALSE, times = 1)
train_data <- pediatric_data_clean[train_indices, ]
test_data  <- pediatric_data_clean[-train_indices, ]

message("Training set dimensions: ", paste(dim(train_data), collapse = " x "))
message("Testing set dimensions: ", paste(dim(test_data), collapse = " x "))

# --- Save Split Data ---
train_file_path <- file.path(output_dir, train_split_output_name)
test_file_path <- file.path(output_dir, test_split_output_name)

message("Saving training data to: ", train_file_path)
write.csv(train_data, train_file_path, row.names = FALSE, na = "")
message("Saving testing data to: ", test_file_path)
write.csv(test_data, test_file_path, row.names = FALSE, na = "")

# --- Prepare Data for XGBoost ---
# XGBoost expects a numeric matrix for features and a numeric vector for the label
# Ensure target is 0-indexed factor/numeric
train_labels <- as.integer(train_data$pediatric_class_target) # Already 0,1,2 from earlier
test_labels  <- as.integer(test_data$pediatric_class_target)

# Create DMatrices
train_matrix <- as.matrix(train_data[, features, drop = FALSE])
mode(train_matrix) <- "numeric"

test_matrix <- as.matrix(test_data[, features, drop = FALSE])
mode(test_matrix) <- "numeric"

# Check for columns that are entirely NA after subsetting (if any row had NA in target and was removed)
dtrain <- xgb.DMatrix(data = train_matrix, label = train_labels, missing = NA)
dtest  <- xgb.DMatrix(data = test_matrix,  label = test_labels, missing = NA)

# --- Train XGBoost Model ---
message("Training XGBoost model...")
num_classes <- length(unique(pediatric_data_clean$pediatric_class_target))
message(paste("Number of classes detected:", num_classes))

if (num_classes <= 1) {
    stop("Not enough classes to train a multi-class model. Need at least 2.")
}

# XGBoost parameters
params <- list(
  booster = "gbtree",
  objective = "multi:softprob", # Output probabilities for each class
  num_class = num_classes,
  eta = 0.1,                   # learning rate
  max_depth = 6,               # max depth of tree
  subsample = 0.8,             # subsample ratio of the training instance
  colsample_bytree = 0.8,      # subsample ratio of columns when constructing each tree
  eval_metric = "mlogloss"     # evaluation metric
)

# Set number of rounds
nrounds <- 100 # Number of boosting rounds

# Train the model
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = nrounds,
  watchlist = list(train = dtrain, test = dtest),
  early_stopping_rounds = 10, # Stop if mlogloss doesn't improve for 10 rounds on test set
  print_every_n = 10
)

message("XGBoost model training complete.")
message("Best iteration: ", xgb_model$best_iteration)
message("Best mlogloss on test set: ", xgb_model$evaluation_log$test_mlogloss[xgb_model$best_iteration])


# --- Save Trained Model ---
model_file_path <- file.path(output_dir, model_output_name)
message("Saving trained XGBoost model to: ", model_file_path)
# RDS is generally preferred for R objects like models
saveRDS(xgb_model, file = model_file_path)
message("Model saved to:", model_file_path)

# --- Generate Predictions ---
message("Generating predictions for test and train sets...")

# Generate predictions for test set
test_predictions_prob <- predict(xgb_model, dtest, missing = NA)
test_predictions_matrix <- matrix(test_predictions_prob, ncol = num_classes, byrow = TRUE)

# Generate predictions for train set
train_predictions_prob <- predict(xgb_model, dtrain, missing = NA)
train_predictions_matrix <- matrix(train_predictions_prob, ncol = num_classes, byrow = TRUE)

# Function to create prediction dataframe with all original columns plus predictions
create_prediction_df <- function(data, prediction_matrix) {
  # Start with the complete data including all features
  df <- data

  # Add prediction columns
  df$prediction.ALL <- prediction_matrix[, 1]
  df$prediction.AML <- prediction_matrix[, 2]
  df$prediction.APL <- prediction_matrix[, 3]

  return(df)
}

# Create prediction dataframes
test_predictions_df <- create_prediction_df(test_data, test_predictions_matrix)
train_predictions_df <- create_prediction_df(train_data, train_predictions_matrix)

# Save test predictions (predict.csv)
predictions_file_path <- file.path(working_dir, predictions_output_name)
message("Saving test predictions to: ", predictions_file_path)
write.csv(test_predictions_df, predictions_file_path, row.names = FALSE)

# Save train predictions (predict_train.csv)
train_predictions_file_path <- file.path(working_dir, "predict_train.csv")
message("Saving train predictions to: ", train_predictions_file_path)
write.csv(train_predictions_df, train_predictions_file_path, row.names = FALSE)

message("Retraining script finished successfully.")
message(paste("Train split saved to:", train_file_path))
message(paste("Test split saved to:", test_file_path))
message(paste("Predictions saved to:", predictions_file_path))
