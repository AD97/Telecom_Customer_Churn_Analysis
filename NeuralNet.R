# Aadhithya Dinesh
# MIS 545 Section 02
# This R application will import a dataset of telecom customers and generate a 
# neural network model that will predict if a customer will leave the service or not.
# We will be importing the csv file, assigning data types, 
# building a supervised neural network model, and testing for model fit.

# install.packages("tidyverse")
# install.packages("neuralnet")

library(tidyverse)
library(neuralnet)
library(factoextra)
library(cluster)
library(gridExtra)

# set the working directory
setwd("~/MIS/Projects/DataMining/project/MIS545G18")

# read the csv file
churn <- read_csv(file = "./data/ModelChurn.csv",
                      col_types = "ciiiiiiiiiiiiiiiiiddi",
                      col_names = TRUE)

# drop the 11 rows which contain NULL values
churn <- drop_na(churn)

# print the churn tibble
print(churn)

# print the structure of churn tibble
print(str(churn))

# print the summary of churn tibble
print(summary(churn))



# scaling the tenure to a value between 0 and 1
churn <- churn %>%
  mutate(tenureScaled = (tenure - min(tenure))/
           (max(tenure) - min(tenure)))

# scaling the monthly charges to a value between 0 and 1
churn <- churn %>%
  mutate(MonthlyChargesScaled = (MonthlyCharges - min(MonthlyCharges))/
           (max(MonthlyCharges) - min(MonthlyCharges)))

# scaling the total charges to a value between 0 and 1
churn <- churn %>%
  mutate(TotalChargesScaled = (TotalCharges - min(TotalCharges))/
           (max(TotalCharges) - min(TotalCharges)))


set.seed(591)

# creating the training dataset
sampleSet <- sample(nrow(churn),
                    round(nrow(churn)*0.75),
                    replace = FALSE)


# splitting into 75% training dataset
churnTraining <- churn[sampleSet, ]

# loading the remaining 25% of the dataset for testing
churnTesting <- churn[-sampleSet, ]

# generating the neural network with 1 hidden layer
churnNeuralNet <- neuralnet(
  formula = Churn ~ tenureScaled + MultipleLines + InternetServiceNo + 
    SeniorCitizen + Dependents + 
    PaymentMethodElectronicCheck + 
    MonthlyChargesScaled + MultipleLines + TechSupport + ContractOneYear + ContractTwoYear,
  data = churnTraining,
  hidden = 1,
  act.fct = "logistic",
  linear.output = FALSE)

# displaying the neural network results
print(churnNeuralNet$result.matrix)

# using churnProbability to generate probablities on the testing dataset
churnProbability <- compute(churnNeuralNet, 
                            churnTesting)

# visualizing the neural network
plot(churnNeuralNet)

# displaying the results from the testing dataset on the console
print(churnProbability$net.result)

# converting probability predictions into 0 or 1 predictions
churnPrediction <- 
  ifelse(churnProbability$net.result > 0.5, 1, 0)

# displaying the predictions on the console
# print(churnPrediction)

# evaluating the model by forming a confusion matrix
churnConfusionMatrix <- table(churnTesting$Churn,
                              churnPrediction)

# displaying confusion matrix on the console
print(churnConfusionMatrix)

# calculating model predictive accuracy
predictiveAccuracy <- sum(diag(churnConfusionMatrix)) /
  nrow(churnTesting)

# displaying the predictive accuracy
print(predictiveAccuracy)




