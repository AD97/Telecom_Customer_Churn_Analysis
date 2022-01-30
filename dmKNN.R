# Naina Gupta
# MIS 545 Section 02
# DecisionTree.R
# imported a dataset of telecom and generated a k-nearest neighbors model
# that will predict the customer will churn or not
# installed and loaded tidyverse,corrplot, class packages
# install.packages("tidyverse")
# install.packages("olsrr")
library(tidyverse)
library(corrplot)
library(class)

# set working directory 
setwd("~/MIS/Projects/DataMining/project/MIS545G18")
customerChurn <- read_csv("./data/ModelChurn.csv",
                          col_types = "cllllnllllllllllllllllnnl",
                          col_names = TRUE)
# display structure 
str(customerChurn)                  

# display summary 
summary(customerChurn)


# remove customerID from the data
customerChurn1 <- customerChurn %>% select(-customerID)

customerChurn1$TotalCharges[is.na(customerChurn1$TotalCharges)] <-
  mean(customerChurn1$TotalCharges,na.rm=TRUE)

customerChurnLabel <- customerChurn1 %>% select(Churn)
customerChurn1 <- customerChurn1 %>% select(-Churn)

summary(customerChurn1)
# Recreating displayAllHistograms() function
displayAllHistograms <- function(tibbleDataset) {
  tibbleDataset %>%
    keep(is.numeric) %>%
    gather() %>%
    ggplot() + geom_histogram(mapping = aes(x=value, fill=key),
                              color = "black") +
    facet_wrap(~ key, scales = "free") +
    theme_minimal ()
}

displayAllHistograms(customerChurn1)
set.seed(777)
sampleSet <- sample(nrow(customerChurn1),
                    round(nrow(customerChurn1)*0.75),
                    replace= FALSE)

churnTraining <- customerChurn1[sampleSet,]
churnTrainingLabel <- customerChurnLabel[sampleSet,]

churnTesting <- customerChurn1[-sampleSet,]
churnTestingLabel <- customerChurnLabel[-sampleSet,]

# Generate k nearest neighbour
customerPrediction <- knn(train = churnTraining,
                          test = churnTesting,
                          cl = churnTrainingLabel$Churn,
                          k = 70)
print(customerPrediction)
# Display summary of the predictions from the testing dataset
print(summary(customerPrediction))

# Evaluate the model by forming a confusion matrix
churnconfusionMatrix <- table(churnTestingLabel$Churn,
                         customerPrediction)

print(churnconfusionMatrix)

churnpredictiveAccuracy <- sum(diag(churnconfusionMatrix))/
  nrow(churnTesting)

print(churnpredictiveAccuracy)

# Create a matrix of k-values with their predictive accuracy
# Store the matrix into an object called kValueMatrix
kValueMatrix <- matrix(data = NA,
                       nrow = 0,
                       ncol =2)

colnames(kValueMatrix) <- c("k value", "Predictive accuracy")

# Loop through odd values of k from 1 up to the number of records in the
# training 
for(kValue in 1:nrow(churnTraining)) {
  if(kValue %% 2 != 0) {
    customerPrediction <- knn(train = churnTraining,
                           test = churnTesting,
                           cl = churnTrainingLabel$Churn,
                           k = kValue)
    churnconfusionMatrix <- table(churnTestingLabel$Churn,
                                  customerPrediction)
    predictiveAccuracy <- sum(diag(churnconfusionMatrix))/
      nrow(churnTesting)
    kValueMatrix <- rbind(kValueMatrix, c(kValue,churnpredictiveAccuracy))
  }
}

# dataset. With each pass through the loop, store the k-value along
# with its predictive accuracy

# Display the kValueMatrix on the console to determine the best k-value

print(kValueMatrix)
