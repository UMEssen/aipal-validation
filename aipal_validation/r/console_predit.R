# Load required libraries
library(dplyr)
library(tidyr)
library(yaml)

# Load model
res_list <- readRDS("aipal_validation/r/221003_Final_model_res_list.rds")
model <- res_list$final_model

# function to predict a dataframe
predict_df <- function(new_df) {
  # Add calculated column for Monocytes_percent if not already present
  if (!"Monocytes_percent" %in% names(new_df)) {
    new_df$Monocytes_percent <- new_df$Monocytes_G_L * 100 / new_df$WBC_G_L
  }
  
  # Make prediction
  prediction <- predict(model, newdata = new_df, type = "prob", na.action = na.pass)
  
  # Return prediction
  return(prediction)
}

# Usage:
# 1. Create a dataframe with the required columns
# 2. Call predict_df(your_dataframe)
# 3. The output will be a matrix with probabilities for each class
