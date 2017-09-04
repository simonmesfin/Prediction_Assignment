# Prediction_Assignment
Practical Machine Learning: Prediction Assignment 
### **Practical Machine Learning Project : Prediction Assignment Writeup**  

#### **1. Overview**  
This document is a report of the Peer Assessment project of Practical Machine Learning Course. The main goal of the project is to predict the manner in which 6 participants performed some exercise as described below. This is the “classe” variable in the training set. The machine learning algorithm described here is applied to 20 test cases available in the test data and the predictions are submitted in appropriate format to the Course Project Prediction Quiz for automated grading.  

#### **2. Background**  
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, the goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).  

Read more: http://groupware.les.inf.puc-rio.br/har#ixzz3xsbS5bVX  

#### **3. Data Loading and Exploratory Analysis** 

##### **3.1 Dataset Overview**  
The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv  

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv  

The data for this project come from http://groupware.les.inf.puc-rio.br/har. Full source:
Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. “Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human ’13)”. Stuttgart, Germany: ACM SIGCHI, 2013.  

Many Thanks!  

A short description of the datasets content from the authors’ website: “Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).  

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes. Participants were supervised by an experienced weight lifter to make sure the execution complied to the manner they were supposed to simulate. The exercises were performed by six male participants aged between 20-28 years, with little weight lifting experience. We made sure that all participants could easily simulate the mistakes in a safe and controlled manner by using a relatively light dumbbell (1.25kg)."  

##### **3.2 Preprocessing**  

First upload the R libraries necessary for analysis.
```{r warning=FALSE}
# Data Preprocessing
library(caret); library(lattice); library(ggplot2)
library(rpart)
library(knitr)
library(randomForest)
library(rattle)
library(ElemStatLearn)
library(rpart.plot)
library(corrplot)
set.seed(888) # For research reproducibility purpose
```
##### **3.2 Data Loading and Cleaning**  

The next step is loading the dataset from the URL provided above. The training dataset is then partinioned in 2 to create a Training set (70% of the data) for the modeling process and a Test set (with the remaining 30%) for the validations. The testing dataset is not changed and will only be used for the quiz results generation.  
```{r}
# set the URL for the download
TrainingUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
TestingUrl  <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# download the datasets
training <- read.csv(url(TrainingUrl))
testing  <- read.csv(url(TestingUrl))

# create a partition with the training dataset 
inTrain <- createDataPartition(training$classe, p = 0.7, list = FALSE)
Train <- training[inTrain, ]
Test <- training[-inTrain, ]
dim(Train)
dim(Test)
```
Both created datasets have 160 variables. Those variables have plenty of NA, that can be removed with the cleaning procedures below. The Near Zero variance (NZV) variables are also removed and the ID variables as well.  
```{r}
# remove variables with Nearly Zero Variance
NZV <- nearZeroVar(Train)
Train <- Train[, -NZV]
Test <- Test[, -NZV]
dim(Train)
dim(Test)
# remove variables that are mostly NA
AllNA <- sapply(Train, function(x) mean(is.na(x))) > 0.95
Train <- Train[, AllNA == FALSE]
Test <- Test[, AllNA == FALSE]
dim(Train)
dim(Test)
# remove identification only variables (columns 1 to 5)
Train <- Train[, -(1:5)]
Test  <- Test[, -(1:5)]
dim(Train)
dim(Test)
```
With the cleaning process above, the number of variables for the analysis has been reduced to 54 only.  

##### **3.4 Correlation Analysis**  
A correlation among variables is analysed before proceeding to the modeling procedures.
```{r setup}  
knitr::opts_chunk$set(fig.width = 16, fig.height = 12) 
corMatrix <- cor(Train[, -54])
corrplot(corMatrix, order = "FPC", method = "color", type = "lower", 
         tl.cex = 0.8, tl.col = rgb(0, 0, 0))
```  

The highly correlated variables are shown in dark colors in the graph above.  

### **4. Prediction Model Building**  

Three methods will be applied to model the regressions (in the Train dataset) and the best one (with higher accuracy when applied to the Test dataset) will be used for the quiz predictions. The methods are: Random Forests, Decision Tree and Generalized Boosted Model, as described below.  
A Confusion Matrix is plotted at the end of each analysis to better visualize the accuracy of the models.  

#### **4.1 Method: Random Forest**  
```{r}
# model fit
set.seed(888)
controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
modFitRandForest <- train(classe ~ ., data=Train, method="rf",
                          trControl=controlRF)
modFitRandForest$finalModel

# prediction on Test dataset
predictRandForest <- predict(modFitRandForest, newdata=Test)
confMatRandForest <- confusionMatrix(predictRandForest, Test$classe)
confMatRandForest

# plot matrix results
plot(confMatRandForest$table, col = confMatRandForest$byClass, 
     main = paste("Random Forest - Accuracy =",
                  round(confMatRandForest$overall['Accuracy'], 4)))
```  

#### **4.2 Method: Decision Trees**
```{r, warning=FALSE}  

# model fit
set.seed(888)
modFitDecTree <- rpart(classe ~ ., data=Train, method="class")
fancyRpartPlot(modFitDecTree)

# prediction on Test dataset
predictDecTree <- predict(modFitDecTree, newdata=Test, type="class")
confMatDecTree <- confusionMatrix(predictDecTree, Test$classe)
confMatDecTree

# plot matrix results
plot(confMatDecTree$table, col = confMatDecTree$byClass, 
     main = paste("Decision Tree - Accuracy =",
                  round(confMatDecTree$overall['Accuracy'], 4)))
```  

#### **4.3 Method: Generalized Boosted Model**
```{r results='hide', warning=FALSE}
library(gbm); library(survival); library(splines); library(parallel); library(plyr)  
# model fit
set.seed(888)
controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modFitGBM  <- train(classe ~ ., data=Train, method = "gbm",
                    trControl = controlGBM, verbose = FALSE)
modFitGBM$finalModel

# prediction on Test dataset
predictGBM <- predict(modFitGBM, newdata=Test)
confMatGBM <- confusionMatrix(predictGBM, Test$classe)
confMatGBM

# plot matrix results
plot(confMatGBM$table, col = confMatGBM$byClass, 
     main = paste("GBM - Accuracy =", round(confMatGBM$overall['Accuracy'], 4)))
```

### **5. Applying the Selected Model to the Test Data**
The accuracy of the 3 regression modeling methods above are:  

1. Random Forest : 0.998  
2. Decision Tree : 0.8318  
3. GBM : 0.9878  
Random Forest model will be applied to predict the 20 quiz results (testing dataset).  
```{r}
predictTEST <- predict(modFitRandForest, newdata=testing)
predictTEST
```
