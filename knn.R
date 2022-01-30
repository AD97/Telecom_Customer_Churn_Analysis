# needs to be run every time you start R and want to use %>%
library(dplyr)
library(caret)
library(tidyverse)
library(dplyr)
library(corrplot)
library(dummies)
library(corrplot)
library(olsrr) 
library(ggplot2)
library(gridExtra)
library(ggthemes)
library(class)

setwd("C:/Users/ngnai/OneDrive/Desktop/ECON 511A")
customerChurn <- read_csv("./data/ModelChurn.csv")

str(customerChurn)                  

summary(customerChurn)

customerChurn1 <- customerChurn %>% select(-customerID)

customerChurn1$TotalCharges[is.na(customerChurn1$TotalCharges)] <-
  mean(customerChurn1$TotalCharges,na.rm=TRUE)

customerChurnLabel <- customerChurn1 %>% select(Churn)
customerChurn1 <- customerChurn1 %>% select(-Churn)

summary(customerChurn1)

set.seed(1234)
sampleSet <- sample(nrow(customerChurn1),
                    round(nrow(customerChurn1)*0.70),
                    replace= FALSE)

Training <- customerChurn1[sampleSet,]
TrainingLabel <- customerChurnLabel[sampleSet,]

Testing <- customerChurn1[-sampleSet,]
TestingLabel <- customerChurnLabel[-sampleSet,]

# Generate k nearest neighbour
customerPrediction <- knn(train = Training,
                          test = Testing,
                          cl = TrainingLabel$Churn,
                          k = 70)
print(customerPrediction)

confusionMatrix <- table(TestingLabel$Churn,
                         customerPrediction)

print(confusionMatrix)

predictiveAccuracy <- sum(diag(confusionMatrix))/
  nrow(Testing)

print(predictiveAccuracy)