Implementing a Random Forest Model
========================================================


** Purpose of Project **
This project is designed to build a prediction model using data from wearable devices. The predictions are which category of exercise an individual is doing based on the tracking data. As shown below the model created does successfully predict all 20 examples. 

**How Predictors Were Chosen**
  The initial data file contained 160 unique column variables (including one which defined the result). Initial predictor selection took the following form:
1) Eliminate columns that have N/A values.
2) Eliminate columns which blank values. 
3) Eliminate columns which contain non-measurement data (record ID's, time and date, names). 

These steps brought the number of predictors down to 50. 

The second process for reducing predictors was to run a stats package (???) against the results and eliminating any that had low variance (defined as variance under 2 for this data set). This left 16 predictors. 

As a final strategy, the findCorrelation() function in the caret package was run against the remaining predictors. It returned two predictors which were highly correlated to the rest and could be eliminated. 

**Building the Random Forest Model**

```r
data <- read.csv("pml-training3.csv")
library(caret)
```

```
## Warning: package 'caret' was built under R version 2.15.3
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 2.15.3
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 2.15.3
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(ggplot2)
data$classe <- factor(data$classe)
testIndex = createDataPartition(data$classe, p = 0.6, list = FALSE)
training <- data[-testIndex, ]
test <- data[testIndex, ]
trControl = trainControl(method = "cv", number = 10, allowParallel = TRUE, verboseIter = FALSE)
modfit <- train(classe ~ ., data = training, method = "rf", prox = FALSE, trControl = trControl)
```

```
## Warning: package 'e1071' was built under R version 2.15.3
```

```r
testclass <- predict(modfit, newdata = test)


cfMatrix <- confusionMatrix(data = testclass, test$classe)
```


**Running the Model on Test Data (partitioned from original data)**
The following is the confusion matrix from running the model against the test data. 

```r
cfMatrix
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3329   58    0    1    0
##          B   15 2169   43    0   15
##          C    3   36 2005   28    9
##          D    0   14    6 1893   20
##          E    1    2    0    8 2121
## 
## Overall Statistics
##                                         
##                Accuracy : 0.978         
##                  95% CI : (0.975, 0.981)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : < 2e-16       
##                                         
##                   Kappa : 0.972         
##  Mcnemar's Test P-Value : 1.14e-13      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.994    0.952    0.976    0.981    0.980
## Specificity             0.993    0.992    0.992    0.996    0.999
## Pos Pred Value          0.983    0.967    0.963    0.979    0.995
## Neg Pred Value          0.998    0.988    0.995    0.996    0.995
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.283    0.184    0.170    0.161    0.180
## Detection Prevalence    0.288    0.190    0.177    0.164    0.181
## Balanced Accuracy       0.994    0.972    0.984    0.988    0.989
```

**Results On A Set of Unknown Quantities**

A set of 20 test cases was provided as a final demonstration of the random forest implementation. 


```r
testdata <- read.csv("pml-testing.csv")
output <- predict(modfit, newdata = testdata)
output
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

