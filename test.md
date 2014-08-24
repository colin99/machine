Implementing a Random Forest Model
========================================================


**Purpose of Project**
This project is designed to build a prediction model using data from wearable devices. The predictions are which category of exercise an individual is doing based on the tracking data. As shown below the model created does successfully predict all 20 examples. 


```r
## The pml-training3.csv file contains the reduced number of columns (the
## final view used to train the model). The stats-test.csv file is a 50+
## variable version of it which is included here to illustrate the use of
## the describe() function in reducing the number of predictors.
options(warn = -1)
data <- read.csv("pml-training3.csv")
stats <- read.csv("stats-test.csv")
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(ggplot2)
library(psych)
```

```
## 
## Attaching package: 'psych'
## 
## The following object(s) are masked from 'package:ggplot2':
## 
##     %+%
```

```r
data$classe <- factor(data$classe)
testIndex = createDataPartition(data$classe, p = 0.6, list = FALSE)
training <- data[-testIndex, ]
test <- data[testIndex, ]
lowVariance <- describe(stats)

```


**How Predictors Were Chosen**
  The initial data file contained 160 unique column variables (including one which defined the result). Initial predictor selection took the following form:
1) Eliminate columns that have N/A values.
2) Eliminate columns which blank values. 
3) Eliminate columns which contain non-measurement data (record ID's, time and date, names). 

These steps brought the number of predictors down to 50. 

The second process for reducing predictors was to run a stats package (the describe command in psych package) against the results and eliminating any that had low variance (defined as variance under 2 for this data set). This left 16 predictors and the "classe" variable. 



```r
plot(lowVariance$sd, ylab = "variance")
```

![plot of chunk unnamed-chunk-2](figure/unnamed-chunk-2.png) 



As a final strategy, the findCorrelation() function in the caret package was run against the remaining predictors. It returned two predictors which were highly correlated to the rest and could be eliminated. 

**Building the Random Forest Model**

Once the potential set of predictors was determined a random forest model was created using the caret package. 60 percent of the training data was used to create an initial training set to build the model. The remaining 40 percent was used as a test data set for the resulting model. 


```r
trControl = trainControl(method = "cv", number = 10, allowParallel = TRUE, verboseIter = FALSE)
modfit <- train(classe ~ ., data = training, method = "rf", prox = FALSE, trControl = trControl)
testclass <- predict(modfit, newdata = test)


cfMatrix <- confusionMatrix(data = testclass, test$classe)
```

**Cross Validation Set**

The random forest implementation does cross validation on it's own when selecting the appropriate model parameters. The cross validation is determined by the 'number' and 'method' parameters in trainControl. In this case, the number of cross-validation is set to 10. 


```r
print(modfit, digits = 4)
```

```
## Random Forest 
## 
## 7846 samples
##   14 predictors
##    5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 7061, 7062, 7063, 7062, 7060, 7061, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa   Accuracy SD  Kappa SD
##   2     0.9749    0.9682  0.005485     0.006939
##   8     0.9757    0.9692  0.007309     0.009258
##   14    0.9713    0.9637  0.008241     0.01044 
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 8.
```



**Running the Model on Test Data (partitioned from original data)**

Following is the confusion matrix from running the model against the test data. It shows the overall accuracy of the model is 97.8 percent which compares favorably to the original training data used to generate the model (those statistics are shown in the previous section).

```r
print(cfMatrix, digits = 4)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3321   40    2    0    1
##          B   15 2199   34    2   19
##          C   12   27 2003   22    9
##          D    0   11   15 1899    3
##          E    0    2    0    7 2133
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9812          
##                  95% CI : (0.9786, 0.9836)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9763          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9919   0.9649   0.9752   0.9839   0.9852
## Specificity            0.9949   0.9926   0.9928   0.9971   0.9991
## Pos Pred Value         0.9872   0.9691   0.9662   0.9850   0.9958
## Neg Pred Value         0.9968   0.9916   0.9947   0.9969   0.9967
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2820   0.1867   0.1701   0.1613   0.1811
## Detection Prevalence   0.2857   0.1927   0.1760   0.1637   0.1819
## Balanced Accuracy      0.9934   0.9788   0.9840   0.9905   0.9921
```

**Results On A Set of Unknown Quantities**

A set of 20 test cases was provided as a final demonstration of the random forest implementation. In all 20 cases, the correct answer was generated by the model. 


```r
testdata <- read.csv("pml-testing.csv")
output <- predict(modfit, newdata = testdata)
output
```

```
##  [1] B A B A A C D B A A B C B A E E A B B B
## Levels: A B C D E
```

