# Coronary Heart Disease Prediction

This repository contains our work for Project 1 in EPFL’s **CS-433 Machine Learning** course (Fall 2025).  
The goal of this project is to predict the risk of coronary heart disease (CHD) using health, lifestyle, and demographic data from the Behavioral Risk Factor Surveillance System (**BRFSS**) - a large-scale U.S. health survey with over 300,000 individuals.

## Project Overview

We implemented, analyzed, and compared several machine learning models to predict CHD and myocardial infarction outcomes.  
Through a series of notebooks, we carried out the full ML pipeline for CHD prediction, including:

- Exploratory data analysis  
- Data cleaning and preprocessing  
- Feature selection and dimensionality analysis  
- Model training and hyperparameter tuning  (weighted logistic regression, decision tree)
- Evaluation and interpretation of results
- Submission to the platform

## Authors

| Name | Email |
|------|--------|
| Emma Anastassova | [emma.anastassova@epfl.ch] |
| Henrik Gruber | [henrik.gruber@epfl.ch] |
| Andrea Saporito | [andrea.saporito@epfl.ch] |

## Project Structure

For more details please check the file <ins>repo_structure.txt</ins> in the base directory. This graph serves as a vast overview.

```
├── documents                   <- General Documents
│
├── notebooks                   <- Notebooks used in the project, ranging from EDA to final submission
│
├── plots                       <- Plots resulting from notebooks
│   ├── eda                         <- Plots from EDA phase
│   ├── eval                        <- Plots from Eval phase
│
├── results                     <- Folder storing the main results
│   ├── eda                         <- Results from EDA (e.g. feature importance)
│   ├── eval                        <- Results from evaluation (e.g. performance of models)
│   ├── submisson                   <- CSV Files to submit to the platform
│
├── utils                       <- Utilities that are used in multiple notebook files (e.g. Data preparation functions, DTs, log reg, etc.)
├──.gitignore                   <- Files ignored by GIT
├── CS433_PestoPasta_proj1.pdf  <- Project report as in the handin
├── implementations.py          <- File with mandatory functions for the handin
├── README.md                   <- Info for the readers
├── repo_structure.txt          <- More detailed repository structure
└── run.py                      <- Python file that reproduces the best submission from the platform
```

## Navigating the repository
We focused on making this repository easy to navigate by simplifying the structure as much as possible.

### Exploratory data analysis  
The Exploratory data analysis can be found in notebooks 1-3. It heavily relies on utility functions that can be found in utils/all_functions.py, utils/eda_helpers.py and utils/helpers.py.
### Data cleaning and preprocessing  
Data cleaning happens mostly in the utility files, where we have summarized the functions that were the results of our EDA. These functions are used throughout all the other parts of the projects as the first step, to get from the full dataset to the one we were eventually working with. Functions are found in utils/all_functions.py, utils/eda_helpers.py and utils/helpers.py.
### Feature selection and dimensionality analysis  
Feature selection happens in notebook 2 and can be found in files utils/all_functions.py, utils/eda_helpers.py and utils/helpers.py. It's outcomes are also used as data preparation in subsequent notebooks.
### Model training and hyperparameter tuning  (weighted logistic regression, decision tree)
Model training can be found in notebooks 4-6, which also use utils/all_functions.py together with the implemented utils/adecision_tree.py and utils/arandom_forest.py. Other methods for training can be found in utils/sampling_methods.py
### Evaluation and interpretation of results
Evaluation happens in notebook 7, but also utilizing many functions as described above
### Submission to the platform
Submission to the platform happens in the last notebook (notebook 8).