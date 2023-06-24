##################################
#Properties of gas_data variables#
##################################
#Date - UK Format Daily Short Date.
#P.therm - Gas Day Ahead Price in Pence Per Therm.
#Temp - Average Daily UK Temperature in Degrees Celsius.
#SN - 10 Year Seasonal Norm Temperature in Degrees Celsius.
#War - Boolean Variable for Russia's Invasion of Ukraine.
#COban - Boolean Variable for Europe's Ban on Russian Crude Oil.
#Pban - Boolean Variable for Europe's Ban on Russian Petroleum Products.

########################################
#Setting working directory instructions#
########################################
#Before running the following code it is important to first set the working directory to the folder where the original
#data file is stored. This can be done in the following ways:
#Navigate to: Session > Set Working Directory > Choose Directory
#Alternatively you can press: Ctrl + Shift + H
#Or you can type the code: setwd("") and type the folder directory into the quotation marks
setwd("C:\\Users\\Nikit\\Desktop\\Business Case for R-studio\\Proc Web Demo\\Gas Cost Model")

##############################################
#Installing and calling all relevant packages#
############################################## 
if (!("tidyverse" %in% installed.packages())) {
  install.packages("tidyverse")}
library(tidyverse)

if (!("caret" %in% installed.packages())) {
  install.packages("caret")}
library(caret)

if (!("xgboost" %in% installed.packages())) {
  install.packages("xgboost")}
library(xgboost)

if (!("ggplot2" %in% installed.packages())) {
  install.packages("ggplot2")}
library(ggplot2)

###############
#Data Cleaning#
###############

#Import Data Set:
original_data <- read.csv("gas_data.csv")

#############################
#Linear Regression Modelling#
#############################

#Split the dataset into test and train samples:
set.seed(123)
trainIndex <- createDataPartition(original_data$P.therm, p = 0.8, list = FALSE)
train_data <- original_data[trainIndex,]
test_data <- original_data[-trainIndex,]

#Create a control object for cross-validation of final models:
train_control <- trainControl(method = "cv", number = 10)

#Model 1 - Gas Prices against Temperature:
model1 <- train(P.therm ~ Temp, data = train_data, method = "lm", trControl = train_control)

#Model 2 - Gas Prices against Temperature and All Boolean Supply Shock variables:
model2 <- train(P.therm ~ Temp + War + COban + Pban, data = train_data, method = "lm", trControl = train_control)

#Model 3 - Gas Prices against Temperature and War in Ukraine:
model3 <- train(P.therm ~ Temp + War, data = train_data, method = "lm", trControl = train_control)

##########################
#Evaluating Linear Models#
##########################

#Predicting test data
pred1 <- predict(model1, newdata = test_data)
pred2 <- predict(model2, newdata = test_data)
pred3 <- predict(model3, newdata = test_data)

#Calculate RMSE
rmse1 <- RMSE(pred1, test_data$P.therm)
rmse2 <- RMSE(pred2, test_data$P.therm)
rmse3 <- RMSE(pred3, test_data$P.therm)

#Calculate R^2
rsq1 <- R2(pred1, test_data$P.therm)
rsq2 <- R2(pred2, test_data$P.therm)
rsq3 <- R2(pred3, test_data$P.therm)

#Display metrics
data.frame(Model = c("Model1", "Model2", "Model3"),
           RMSE = c(rmse1, rmse2, rmse3),
           R2 = c(rsq1, rsq2, rsq3))

#Model 2 shows the best results with the lowest RMSE value. This suggests that variables Pban and COban
#do not add value to the predictive model. A possible explanation for this is that the start of the Ukraine
#War signaled to market investors that a Russian Petroleum and Crude Oil export ban is likely to be
#actioned. Therefore, the gas price rise explained by the War variable already factored in these sanctions.

#Model 2 shows the best results but let's inspect the coefficients to further assess model validity.
model2_coefficients <- coef(model2$finalModel)
print(model2_coefficients)

#Model 3 Coefficients
model3_coefficients <- coef(model3$finalModel)
print(model3_coefficients)

#Final Linear Model - Gas Prices against Temperature and War in Ukraine on the full data set:
linear_model <- train(P.therm ~ Temp + War, data = original_data, method = "lm")
linear_model_coefficients <- coef(linear_model$finalModel)
print(linear_model_coefficients)

#########
#xGboost#
#########

#Prepare the data for xGBoost by converting it into a matrix:
train_data_matrix <- model.matrix(P.therm ~ Temp + War + COban + Pban, data = train_data)[,-1]
test_data_matrix <- model.matrix(P.therm ~ Temp + War + COban + Pban, data = test_data)[,-1]

#Train the test xGboost model:
set.seed(100)
xgb_params <- list(objective = "reg:squarederror", eval_metric = "rmse", eta = 0.01, max_depth = 6)

xgb_train_data <- xgb.DMatrix(data = train_data_matrix, label = train_data$P.therm)
xgb_test_data <- xgb.DMatrix(data = test_data_matrix, label = test_data$P.therm)

xgb_model <- xgb.train(params = xgb_params, data = xgb_train_data, nrounds = 1000, early_stopping_rounds = 10, watchlist = list(val = xgb_test_data), verbose = 0)

#Evaluate the performance of the test model:
xgb_pred <- predict(xgb_model, newdata = xgb_test_data)

xgb_rmse <- RMSE(xgb_pred, test_data$P.therm)
xgb_r2 <- R2(xgb_pred, test_data$P.therm)

data.frame(Model = c("XGBoost"), RMSE = c(xgb_rmse), R2 = c(xgb_r2))

#Train xGBoost model on the entire data set:

original_data_matrix <- model.matrix(P.therm ~ Temp + War + COban + Pban, data = original_data)[,-1]
xgb_original_data <- xgb.DMatrix(data = original_data_matrix, label = original_data$P.therm)
xgb_model_full <- xgb.train(params = xgb_params, data = xgb_original_data, nrounds = 1000)

#Extract the importance of each predictor variable in the final model:
importance_matrix <- xgb.importance(feature_names = colnames(train_data_matrix), model = xgb_model_full)
print(importance_matrix)

#Save the final model to use in the app:
saveRDS(xgb_model_full, "xgb_model_full.rds")

#############
#Forecasting#
#############

#Define Forecast Date Range:
start_date <- as.Date("2023-04-01")
end_date <- as.Date("2024-03-31")
date_seq <- seq(from = start_date, to = end_date, by = "day")

#Create a data frame for the latest year of data to run the forecast with:
last_date <- as.Date("2023-03-30")
start_date_prev_year <- last_date - 365  # 365 days before the last date
prev_year_data <- original_data[as.Date(original_data$Date, format = "%d/%m/%Y") > start_date_prev_year &
                                  as.Date(original_data$Date, format = "%d/%m/%Y") <= last_date, ]

#Create a new data set with the next full year's dates and average seasonal temps:
next_year_data <- prev_year_data
next_year_data$Date <- as.Date(next_year_data$Date, format = "%d/%m/%Y") + 365 # Add 1 year to the date

# Set the boolean variables to 1 (True)
next_year_data$War <- 1
next_year_data$COban <- 1
next_year_data$Pban <- 1

#Remove the P.therm variable
next_year_data$P.therm <- NULL

#Create a CSV file for next year data to reuse in later projects:
write.csv(next_year_data, file = "next_year_data.csv") #WARNING: Adds extra column to data set.

#Convert next_year_data into a matrix format suitable for XGBoost
next_year_data_matrix <- model.matrix(~ Temp + War + COban + Pban, data = next_year_data)[,-1]

#Convert the matrix to an xgb.DMatrix
xgb_next_year_data <- xgb.DMatrix(data = next_year_data_matrix)

#Generate the gas price forecast using the xgb_model_full
next_year_data$P.therm_forecast <- predict(xgb_model_full, newdata = xgb_next_year_data)

#Define Mark to Market value:
next_year_data$M2M  <- 294.02*0.5096 + next_year_data$P.therm_forecast * (1 - 0.5096)






















######
#Plot#
######
#Add a month and year column to next_year_data
next_year_data$Month <- format(next_year_data$Date, "%m")
next_year_data$Year <- format(next_year_data$Date, "%Y")

#Calculate the monthly average P.therm_forecast and M2M
monthly_avg <- next_year_data %>%
  group_by(Year, Month) %>%
  summarise(Avg_P.therm_forecast = mean(P.therm_forecast),
            Avg_M2M = mean(M2M))

#Convert the Year and Month columns to Date format
monthly_avg$Date <- as.Date(paste(monthly_avg$Year, monthly_avg$Month, "01", sep = "-"))

#Create the area line graph
ggplot() +
  # P.therm_forecast area and line
  geom_area(data = monthly_avg, aes(x = Date, y = Avg_P.therm_forecast, fill = "Gas Price Forecast"), alpha = 0.4) +
  geom_line(data = monthly_avg, aes(x = Date, y = Avg_P.therm_forecast, color = "Gas Price Forecast"), size = 1) +
  
  # M2M area and line
  geom_area(data = monthly_avg, aes(x = Date, y = Avg_M2M, fill = "Mark to Market"), alpha = 0.4) +
  geom_line(data = monthly_avg, aes(x = Date, y = Avg_M2M, color = "Mark to Market"), size = 1) +
  
  # Axis labels and title
  labs(x = "Date (2023/2024)", y = "Price (Pence per Therm)", title = "Monthly Average Gas Price Forecast and Mark to Market") +
  
  # Legend settings
  scale_fill_manual("Legend", values = c("Gas Price Forecast" = "#5b358c", "Mark to Market" = "#0091aa")) +
  scale_color_manual("Legend", values = c("Gas Price Forecast" = "#5b358c", "Mark to Market" = "#0091aa")) +
  
  # X-axis settings
  scale_x_date(labels = scales::date_format("%b"), breaks = scales::date_breaks("1 month")) +
  
  # Y-axis settings
  scale_y_continuous(breaks = seq(0, max(monthly_avg$Avg_M2M), by = 50)) +
  
  # Theme settings
  theme_minimal() +
  theme(legend.title = element_text(size = 12),
        legend.text = element_text(size = 10),
        plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
        legend.position = c(0.92, 0.92),  # Position of legend inside the top right corner
        panel.grid.major.y = element_line(color = "gray", size = 0.5),  # Add horizontal major gridlines
        panel.grid.major.x = element_blank(),# Remove vertical major gridlines
        panel.grid.minor = element_blank())  # Remove minor gridlines
