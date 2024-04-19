# Load required libraries
library(dplyr)
library(tidyr)

# Load model and predictions
res_list <- readRDS("r/221003_Final_model_res_list.rds")
model <- res_list$final_model

# Load new data from CSV file
new_data <- read.csv("data/V1/aipal/samples.csv")

# Check and replace NaN and Inf values with NA
new_data[sapply(new_data, is.numeric)] <- sapply(new_data[sapply(new_data, is.numeric)], function(x) replace(x, is.nan(x) | is.infinite(x), NA))

# Add calculated column for Monocytes_percent
new_data$Monocytes_percent <- new_data$Monocytes_G_L * 100 / new_data$WBC_G_L

# Replace any resulting Inf or NaN in calculations (if any)
new_data$Monocytes_percent[is.infinite(new_data$Monocytes_percent) | is.nan(new_data$Monocytes_percent)] <- NA

# Prediction function
predict_type <- function(new_data) {
  prediction <- predict(model, newdata = new_data, type = "prob", na.action = na.pass)
  return(prediction)
}

# Execute prediction for all rows
new_data$prediction <- predict_type(new_data)

# Save new_data with predictions to a new CSV file
write.csv(new_data, file = "data/V1/aipal/predict.csv", row.names = FALSE)
