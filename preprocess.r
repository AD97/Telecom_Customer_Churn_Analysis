library(tidyverse)
library(dummies)
library(corrplot)
library(smotefamily)
###########################
# Import data
##########################

# setwd()
setwd("~/MIS/Projects/DataMining/project/MIS545G18")

telecom <- read_csv(file = './data/WA_Fn-UseC_-Telco-Customer-Churn.csv',
                    col_types = 'cflffiffffffffffffnnf',
                    col_names = TRUE)


###########################
# Data Inspection 
##########################

# To see the data condition
summary(telecom)

# TotalCharges have na data, go deeper to see the row
checkNa <- telecom[is.na(telecom$TotalCharges),]
print(checkNa)

# All the na data have MonthlyCharges, but tenure is 0
# Dose 0 tenure cause na TotalCharges?
print(telecom[telecom$tenure == 0,c("customerID","tenure","TotalCharges")])
# Yes~!! They are the new costumer! Considering its tinny ratio , remove them!
telecomNoNA <- telecom[!is.na(telecom$TotalCharges),]

# Check again
summary(telecomNoNA)
# There are no na value any more!

# Remove ID
telecomNoNA$customerID <- NULL
# Check again
summary(telecomNoNA)


###########################
# Start the Visualization
##########################

# For the Visualizational purpose change SeniorCitizen to YES/NO
telecomVisual <- telecomNoNA %>% 
  mutate(SeniorCitizen = ifelse(SeniorCitizen ==  0, 'NO', 'YES'))

#========
# Numeric data analysis
#========
summary(telecomVisual)
telecomVisNumeric <- telecomVisual %>% 
  select(tenure,MonthlyCharges,TotalCharges)

# Display all boxplot
displayAllBoxplots <- function(tibbleDataset) {
  tibbleDataset %>% 
    keep(is.numeric) %>%
    gather() %>%
    ggplot() + 
    geom_boxplot(mapping = aes(x=value, fill=key),
                 color = "black",
                 outlier.shape = 1) +
    facet_wrap( ~ key, scales = "free") +
    theme_minimal() +
    coord_flip()
}
displayAllBoxplots(telecomVisual)
# -- No outlier in the data

# Display all histograms
displayAllHistograms <- function(tibbleDataset) {
  tibbleDataset %>% 
    keep(is.numeric) %>%
    gather() %>%
    ggplot() + 
    geom_histogram(mapping = aes(x=value, fill=key),
                   color = "black") +
    facet_wrap( ~ key, scales = "free") +
    theme_minimal()
}
displayAllHistograms(telecomVisual)
# -- Go to find the interesting points. 
# -- Might need add the range in monthlyCharges as a category when improving


# Display a correlation matrix
round(cor(telecomVisNumeric),2)

corrplot(cor(telecomVisNumeric),
         method = "number",
         type = "lower")
# --  Note: if your algorithm have to do extra preprocess, you have to care 
#           about the threshold of 0.7, then remove the variables.(See HW6) 

# Display scatterplots
telecomVisNumeric %>%
  ggplot() +
  geom_point(mapping = aes(y = tenure , x = Contract), color = "blue",
             size = 2) +
  labs(title = "xxxxx",
       x = "tenure", y = "MonthlyCharges")

#========
# Categorical data analysis
#========
summary(telecomVisual)
# -- Go to find the interesting points. 
summary(telecomVisual)
telecomVisual %>%
  ggplot() +
  geom_bar(mapping = aes(x =Churn, fill = InternetService), color = "black") +
  labs(title = "Customer Churn based on InternetService",
       x = "Churn", y = "number") 

telecomVisual %>%
  ggplot() +
  geom_bar(mapping = aes(x =Churn, fill = Contract), color = "black") +
  labs(title = "Customer Churn based on Contract duration",
       x = "Churn", y = "number") 

telecomVisual %>%
  ggplot() +
  geom_bar(mapping = aes(x =Churn, fill = OnlineSecurity), color = "black") +
  labs(title = "Customer Churn based on Online Security",
       x = "Churn", y = "number") 




#########################
# pre-process for moduel
#########################

#========
# Numeric data 
#========
# normalization
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}
telecomNorm <- telecomNoNA %>% 
  mutate(tenure = normalize(tenure),
         MonthlyCharges = normalize(MonthlyCharges),
         TotalCharges = normalize(TotalCharges))
summary(telecomNorm)

#========
# Categorical data 
#========
# gender SeniorCitizen Partner Dependents PhoneService Churn to 0 and 1
telecomYN <- telecomNorm %>% mutate(gender = ifelse(gender == 'Female', 0, 1),
                                    SeniorCitizen = ifelse(SeniorCitizen == 'FALSE', 0, 1),
                                    Partner = ifelse(Partner == 'No', 0, 1),
                                    Dependents = ifelse(Dependents == 'No', 0, 1),
                                    PhoneService = ifelse(PhoneService == 'No', 0, 1),
                                    Churn = ifelse(Churn == 'No', 0, 1)
)
summary(telecomYN)

# Dummy
telecomDF <- data.frame(telecomYN)
telecomLabel <- telecomDF %>% select(Churn)
telecomDF <- telecomDF %>% select(-Churn)
telecomDummy <- dummy.data.frame(data = telecomDF, sep = "_")

# Remove space from the colnames
newName <- sapply(colnames(telecomDummy),
                  function(x) gsub(" ","",x, fixed = TRUE))
newName <- sapply(newName,
                  function(x) gsub("(","",x, fixed = TRUE))
newName <- sapply(newName,
                  function(x) gsub(")","",x, fixed = TRUE))
colnames(telecomDummy) <- newName


# Remove duplicate dummy columns
contract_Month2month <- newName[str_detect(newName,'Contract_Month-to-month')]
PaymentMethod_Bank <- newName[str_detect(newName,'PaymentMethod_Bank')]
telecomDummy <- telecomDummy[, !(names(telecomDummy)) %in% c(newName[str_detect(newName,'_No')],
                                                             contract_Month2month
                                                             ,PaymentMethod_Bank)]

# Add the Churn back
telecomDummy <- cbind(telecomDummy,telecomLabel)
summary(telecomDummy)


# Randomly split the dataset into mobilePhoneTraining (75% of records) and 
# mobilePhoneTesting (25% of records) using 777 as the random seed
set.seed(777)
sampleSet <- sample(nrow(telecomDummy),
                    round(nrow(telecomDummy)*0.75),
                    replace= FALSE)

churnTraining <- telecomDummy[sampleSet,]
churnTesting <- telecomDummy[-sampleSet,]


# Deal with class imbalance using the SMOTE technique 
getImbalanceRatio <- function(column) {
  keys <- unique(column)
  firstNumber <- sum(churnTraining$Churn==keys[1])
  secndNumber <- sum(churnTraining$Churn==keys[2])
  if (firstNumber > secndNumber) {
    return(c(toString(keys[1]),firstNumber/secndNumber)) 
  } else {
    return(c(toString(keys[2]),secndNumber/firstNumber)) 
  }
}
getImbalanceRatio(churnTraining$Churn)

churnTrainingSmoted <- 
  tibble(SMOTE( X = data.frame(churnTraining),
                target = churnTraining$Churn,
                dup_size = 2.8)$data)

summary(churnTrainingSmoted$Churn)

# Change Churn back into factor and remove class created by SMOTE
churnTrainingSmoted$Churn <- as.factor(churnTrainingSmoted$Churn)
churnTesting$Churn <- as.factor(churnTesting$Churn)
churnTrainingSmoted$class <- NULL

#### Note: SMOTE effect 
##### KNN:  N (59, 0.7940842) | N + B (9, 0.705347) 
##### DTree: (0.005, 0.7906712) | N (xxxx) | N + B (xxx)
churnTraining$Churn <- as.factor(churnTraining$Churn)






###############
# KNN
##############
library(class)

churnTrainingSmotedLabel <- churnTrainingSmoted %>% select(Churn) 
churnTrainingSmoted <- churnTrainingSmoted %>% select(-Churn) 
print(churnTrainingSmoted)
churnTrainingLabel <- churnTraining %>% select(Churn) 
churnTraining <- churnTraining %>% select(-Churn) 

churnTestingLabel <- churnTesting %>%  select(Churn) 
churnTesting <- churnTesting %>% select(-Churn) 
#------------
# Non-SMOTE data
#------------
# Generate k nearest neighbour
customerPrediction <- knn(train = churnTraining,
                          test = churnTesting,
                          cl = churnTrainingLabel$Churn,
                          k = 137)
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
for(kValue in 1:1000) {
  if(kValue %% 2 != 0) {
    customerPrediction <- knn(train = churnTraining,
                              test = churnTesting,
                              cl = churnTrainingLabel$Churn,
                              k = kValue)
    churnconfusionMatrix <- table(churnTestingLabel$Churn,
                                  customerPrediction)
    predictiveAccuracy <- sum(diag(churnconfusionMatrix))/
      nrow(churnTesting)
    kValueMatrix <- rbind(kValueMatrix, c(kValue,predictiveAccuracy))
  }
}

# Display the kValueMatrix on the console to determine the best k-value
print(kValueMatrix)

# Display the best k value
maxAUC <- kValueMatrix[kValueMatrix[,"Predictive accuracy"] == 
                         max(kValueMatrix[,"Predictive accuracy"])]
maxAUCMatrix <- kValueMatrix[which(kValueMatrix == maxAUC[1]),]
print(maxAUCMatrix)

#------------
# SMOTE data
#------------
# Create a matrix of k-values with their predictive accuracy
# Store the matrix into an object called kValueMatrix
kValueMatrixSMOTE <- matrix(data = NA,
                            nrow = 0,
                            ncol =2)

colnames(kValueMatrixSMOTE) <- c("k value", "Predictive accuracy")

# Loop through odd values of k from 1 up to the number of records in the
# training 
for(kValue in 1:1000) {
  if(kValue %% 2 != 0) {
    customerPrediction <- knn(train = churnTrainingSmoted,
                              test = churnTesting,
                              cl = churnTrainingSmotedLabel$Churn,
                              k = kValue)
    churnconfusionMatrix <- table(churnTestingLabel$Churn,
                                  customerPrediction)
    predictiveAccuracy <- sum(diag(churnconfusionMatrix))/
      nrow(churnTesting)
    kValueMatrixSMOTE <- rbind(kValueMatrixSMOTE, c(kValue,predictiveAccuracy))
  }
}

# Display the kValueMatrixSMOTE on the console to determine the best k-value
print(kValueMatrixSMOTE)

# Display the best k value
SMOTEmaxAUC <- 
  kValueMatrixSMOTE[kValueMatrixSMOTE[,"Predictive accuracy"] == 
                      max(kValueMatrixSMOTE[,"Predictive accuracy"])]
SMOTEmaxAUCMatrix <- kValueMatrixSMOTE[which(kValueMatrixSMOTE == maxAUC[1]),]
print(SMOTEmaxAUCMatrix)

###############
# Logistic regression
##############

churnModel <- glm(data = churnTraining,
                  family = binomial,
                  formula = Churn ~.)
summary(churnModel)

predictionModel <- predict(churnModel,
                           churnTestingLabel,
                           type = "response")

predictionModel <- ifelse(predictionModel >= 0.5,1,0)

print(predictionModel)  

churnConfusionMatrix <- table(churnTestingLabel$Churn,
                              predictionModel)
print(churnConfusionMatrix)

misClasificError <- mean(predictionModel != churnTestingLabel$Churn)

print(1 - misClasificError)

###############
# Na??ve Bayes
##############

###############
# Decision tree
##############

#------------
# original data
#------------
telecomVisual

#------------
# Non SMOTE data (N)
#------------
churnTraining
churnTesting

#------------
# SMOTE data (N + B)
#------------
churnTrainingSmoted
churnTesting

###############
# Neural network
##############