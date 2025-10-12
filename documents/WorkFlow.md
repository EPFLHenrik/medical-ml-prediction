# WorkFlow

## Every week

1. Do the labs and upload on our branch
2. Save the links of interesting things we get to know
3. Keeping written track of our workflow for the report

## Program

1. 04/10 we will probably have all the "Step 2" functions implemented -> we could meet, test them and merge the final version in main!
2. We should divide the tasks regarding data processing and methods

## Things to do

1. Watch deeply the dataset and understand the context medical-wise (just a bit)
2. Cleaning data
3. Visualize data and feature process with different methods
4. Keeping different versions of features on different branches in order to test all of the potential solutions
5. checking the hypotesis for our methods + beware of overfitting
6. Feeding the dataset to our implemented functions + other that we find/try -> different branches for different methods in order to keep only the best one
7. Trying cross-validation locally
8. Competition
9. Informal report
10. Retrying everything (implementations count much more than expected), maybe GPT-check
11. Report

## 1 - Exploratory data analysis 

1. Cleaning in case of high frequency of NA - find continuous and categorical variables 
2. descriptive statistics: mean, median, variance, pencentiles
3. just visualizing: boxplots, hists, runchart, barchart, heat map -> find and managing outliers
4. correlation: scatterplots, np.corrcoef
5. Variable evaluation : R_squared -> NO DATA DREDGING, always remember the context 
6. Dimensionality reduction

## 2 - Feature processing

1. use EDA results -> keep different versions!
2. Scaling/Normalization : min-max scaling (GD is scale-sensitive), standardization (could help convergence)
3. Transformation: Log/sqrt to stabilize variance, polynomial features creation for capturing nonlinear relationships, Interaction features (eg height*weight=size)
4. Categorical encoding
5. Dimensionality reduction
6. Feature creation: discretization, periodic patterns (if applicable)

## 3 - Underfitting, Overfitting, convergence

1. More training examples / Smaller set of features - fixes high variance (overfitting)
2. Larger/different set of features - fixes high bias (underfitting)
3. Higher max-iterations / Newton's method - fixes the optimization algorithm
4. Different lambda_ (regularization) / Support vector machine: fixes the objective of optimizazion
5. Learning curves (error vs training set size)
6. Visualize y-y_pred
7. If least-squares/poly-regression : plot

## 4 - Improve the model

1. Confusion matrix : ROC curve, precision and recall
2. Misclassified points : why? Per-point error
3. Convergence, step length and method to descend