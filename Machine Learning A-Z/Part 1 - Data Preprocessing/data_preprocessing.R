# Data Preprocessing
# Importing the dataset

#Set Working Directory as the Directory containing Data.csv
dataset = read.csv('Data.csv')

#Missing Data: Use mean of the remaining data

dataset$Age = ifelse(is.na(dataset$Age), 
				ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)), 
				dataset$Age)

dataset$Salary = ifelse(is.na(dataset$Salary), 
				ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)), 
				dataset$Salary)

# Encoding categorical data
#c is a vector in R
dataset$Country = factor(dataset$Country,
					levels = c('France', 'Spain', 'Germany'),
					labels = c(1, 2, 3))

dataset$Purchased = factor(dataset$Purchased,
					levels = c('No', 'Yes'),
					labels = c(0, 1))

# Split data into Training set and Test set 
# Once installed the package, comment
# install.packages('caTools')
library(caTools)
set.seed(123)
# In Python set for test set , in R we need to set for training set and so 80%
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# Category columns country and purchased are string values and so needs to exclude from the set to avoid the error.
training_set[, 2:3] = scale(training_set[, 2:3])
test_set[, 2:3] = scale(test_set[, 2:3])

