# Load required libraries
library(dplyr)
library(tidyr)

# Load model and predictions
res_list <- readRDS("r/221003_Final_model_res_list.rds")
model <- res_list$final_model

# Simulated console input data
new_data <- data.frame(
  MCV_fL = 90.2,
  MCHC_g_L = 330,
  Platelets_G_L = 50,
  age = 55,
  WBC_G_L = 10,
  Monocytes_G_L = 6,
  PT_percent = 6,
  Fibrinogen_g_L = 6,
  LDH_UI_L = 250,
  Lymphocytes_G_L = NA_real_ #ALC_G_L, 5
)

new_data$Monocytes_percent <- new_data$Monocytes_G_L * 100 / new_data$WBC_G_L

# Prediction function
predict_type <- function(new_data) {
  prediction <- predict(model, newdata = new_data, type = "prob", na.action = na.pass)
  return(prediction)
}

# Execute prediction
prediction <- predict_type(new_data)
print(prediction)
