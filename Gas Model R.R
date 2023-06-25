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

#Adding Weekday, Month and Year features for later xGboost Modelling:
original_data$Date <- as.Date(original_data$Date, format ="%d/%m/%Y")
original_data$Weekday <- weekdays(original_data$Date)
original_data$Month <- format(original_data$Date, "%m")
original_data$Year <- format(original_data$Date, "%Y")

#Advanced models such Gradient Boosting Machines are able to incorporate 
#considerations of how market trading hours/days effect the day ahead price.
#The month feature should encourage the Machine Learning model to consider how
#seasonal differences influence gas prices.

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
#War signaled to market investors that a Russian Petroleum and Crude Oil export ban is likely to occur
#Therefore, the gas price rise explained by the War variable already factored in these sanctions.

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
#xGBoost#
#########

#Standardize P.therm
standardize <- function(x) {
  return((x - mean(x)) / sd(x))
}

inverse_standardize <- function(x, original_mean, original_sd) {
  return(x * original_sd + original_mean)
}

# Save original mean and standard deviation for inverse standardization later
original_mean <- mean(original_data$P.therm, na.rm = TRUE)
original_sd <- sd(original_data$P.therm, na.rm = TRUE)

#Standardize P.therm in the training and test data
train_data$P.therm <- standardize(train_data$P.therm)
test_data$P.therm <- standardize(test_data$P.therm)

#Prepare the data for xGBoost by converting it into a matrix
train_data_matrix <- model.matrix(P.therm ~ Temp + War + Weekday + Month + Year, data = train_data)[,-1]
test_data_matrix <- model.matrix(P.therm ~ Temp + War + Weekday + Month + Year, data = test_data)[,-1]

#Train the test xGBoost model
set.seed(100)
xgb_params <- list(objective = "reg:squarederror", eval_metric = "rmse", eta = 0.01, max_depth = 6)

xgb_train_data <- xgb.DMatrix(data = train_data_matrix, label = train_data$P.therm)
xgb_test_data <- xgb.DMatrix(data = test_data_matrix, label = test_data$P.therm)

xgb_model <- xgb.train(params = xgb_params, data = xgb_train_data, nrounds = 1000, early_stopping_rounds = 10, watchlist = list(val = xgb_test_data), verbose = 0)

# Train xGBoost model on the entire data set
original_data$P.therm <- standardize(original_data$P.therm)
original_data_matrix <- model.matrix(P.therm ~ Temp + War + Weekday + Month + Year, data = original_data)[,-1]
xgb_original_data <- xgb.DMatrix(data = original_data_matrix, label = original_data$P.therm)
xgb_model_full <- xgb.train(params = xgb_params, data = xgb_original_data, nrounds = 1000)

#############
#Forecasting# 
#############
#In normal forecasting, you would train the predictive models on lagged variables to later
#predict future outcomes with the most current data. However, in this example, we have
#quite a short snippet of static data which captures a particularly turbulent time for the
#market. Consequently, I decided to fabricate a fictional next year data set to make 
#forecasting predictions on.

# Define Forecast Date Range:
start_date <- as.Date("2023-04-01")
end_date <- as.Date("2024-03-31")
date_seq <- seq(from = start_date, to = end_date, by = "day")

# Create a data frame for the latest year of data to run the forecast with:
last_date <- as.Date("2023-03-30")
start_date_prev_year <- last_date - 365 # 365 days before the last date
prev_year_data <- original_data[as.Date(original_data$Date, format = "%d/%m/%Y") > start_date_prev_year &
                                  as.Date(original_data$Date, format = "%d/%m/%Y") <= last_date, ]

# Create a new data set with the next full year's dates:
next_year_data <- prev_year_data
next_year_data$Date <- as.Date(next_year_data$Date, format = "%d/%m/%Y") + 365 # Add 1 year to the date

# Set the boolean variables to 1 (True)
next_year_data$War <- 1

# Remove the P.therm, COban and Pban variables
next_year_data$P.therm <- NULL
next_year_data$COban <- NULL
next_year_data$Pban <- NULL

# Add the Weekday, Month and Year Features back in:
next_year_data$Weekday <- weekdays(next_year_data$Date)
next_year_data$Month <- format(next_year_data$Date, "%m")
next_year_data$Year <- format(next_year_data$Date, "%Y")

# One-hot encode the Weekday, Month, and Year columns
next_year_data$Weekday <- as.factor(next_year_data$Weekday)
next_year_data$Month <- as.factor(next_year_data$Month)
next_year_data$Year <- as.factor(next_year_data$Year)

# Get feature names from the model
model_features <- xgb_model_full$feature_names

# Exclude the Date column and convert the data frame to a matrix
next_year_data_matrix <- model.matrix(~ Temp + War + Weekday + Month - 1, data = next_year_data)

# Convert to data frame to easily manipulate columns
next_year_data_df <- as.data.frame(next_year_data_matrix)

# Add missing columns (if any) with 0 values
missing_cols <- setdiff(model_features, colnames(next_year_data_df))
for(col in missing_cols) {
  next_year_data_df[[col]] <- 0
}

# Select and reorder columns to match the model's feature names
next_year_data_df <- next_year_data_df[, model_features]

# Convert the data frame to an xgb.DMatrix
xgb_next_year_data <- xgb.DMatrix(data = as.matrix(next_year_data_df))

# Make predictions
next_year_data$P.therm_forecast_standardized <- predict(xgb_model_full, xgb_next_year_data)

# Inverse standardize the predictions to bring them to the original scale
next_year_data$P.therm_forecast <- inverse_standardize(next_year_data$P.therm_forecast_standardized, original_mean, original_sd)

# Define Mark to Market value
next_year_data$M2M <- 295*0.15 + next_year_data$P.therm_forecast * (1 - 0.15)

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
