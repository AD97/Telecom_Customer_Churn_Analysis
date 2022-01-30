#install.packages("tidyverse")
#adding comment - naina
#install.packages("magrittr") # package installations are only needed the first time you use it
#install.packages("dplyr")    # alternative installation of the %>%
library(magrittr) # needs to be run every time you start R and want to use %>%
library(dplyr)

library(tidyverse)
library(dplyr)
library(corrplot)
library(dummies)
library(corrplot)
library(olsrr) 
library(ggplot2)
library(gridExtra)
library(ggthemes)


setwd("~/MIS/Projects/DataMining/project/MIS545G18")
customerChurn <- read_csv("./data/ModelChurn.csv")

str(customerChurn)                  

summary(customerChurn)

customerChurn1 <- customerChurn %>% select(-customerID)

customerChurn1$TotalCharges[is.na(customerChurn1$TotalCharges)] <-
  mean(customerChurn1$TotalCharges,na.rm=TRUE)

summary(customerChurn1)

set.seed(1234)
sampleSet <- sample(nrow(customerChurn1),
                    round(nrow(customerChurn1)*0.70),
                    replace= FALSE)

Training <- customerChurn1[sampleSet,]
Testing <- customerChurn1[-sampleSet,]

churnModel <- glm(data = customerChurn1,
                  family = binomial,
                  formula = Churn ~.)
summary(churnModel)

predictionModel <- predict(churnModel,
                           Testing,
                           type = "response")

predictionModel <- ifelse(predictionModel >= 0.5,1,0)

print(predictionModel)  

churnConfusionMatrix <- table(Testing$Churn,
                              predictionModel)
print(churnConfusionMatrix)

misClasificError <- mean(predictionModel != Testing$Churn)

print(1 - misClasificError)


predictiveAccuracy <- sum(diag(churnConfusionMatrix))/nrow(Testing)
print(predictiveAccuracy)