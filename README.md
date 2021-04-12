# Predicting Stroke

<p>
  <a href="https://github.com/mdalton87" target="_blank">
    <img alt="Matthew" src="https://img.shields.io/github/followers/mdalton87?label=Follow_Matt&style=social" />
  </a>
  <a href="https://github.com/mdalton87/stroke_prediction/commit/main" target="_blank">
    <img alt="Last: Push" src="https://img.shields.io/github/last-commit/matt-and-alicia/zillow-clustering-project" />
  </a>  
    
    
<p>


## Project Description

According to the World Health Organization (WHO) stroke is the 2<sup>nd</sup> leading cause of death globally, responsible for approximately 11% of total deaths.

This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relavant information about the patient.


## Planning
- Create trello board for organization: https://trello.com/b/GOZGfZhv/stroke-prediction
- Define Goals
- Complete Deliverables
- How to get to the end?

### Goals

- Complete first complete pipeline project from start to finish
- Find a model that predicts the likelihood of a person having a stroke
- Learn a new technique during this project


### Deliverables
- A completed notebook full of visuals, commented code with markdown and comments, and machine learning models.
- python files that contain acquire and preparation functions
- python files that contain functions for splitting and explorating the data
- README file to explain the process of this project

### How?
- Begin by selecting and acquiring the data set
    - I chose a data set that contains over 5100 records of patient data of stroke indicators.
- Examine the data for missing values and obvious outliers 
- Prepare the data for exploration and statistical tests
- Explore the univariate, bivariate, and multivariate relationships.
- Run Stats tests to verify that the features are acceptable to be modeled
- Create a model baseline
- Run various models with different hyperparameters for the best results
- Select and test the best performing model. 

### Project Predictions / Hypothesis
- Heart disease will be a driver of stroke
- Decision tree will be my best model due to the large amount of binary features
- Age will be a significant factor of my model
- The dataset is too imbalaced to get an accurate prediction


## Wrangling the Data
- functions for wrangle contained in wrangle.py

### Changes to df:
- set index to id
- made ever_married into binary variable
- replaced 'Unknown' in smoking_status as 'never_smoked'
- created dumm variables of residence_type and gender
- impute knn for bmi using 'age', 'avg_glucose_level', 'heart_disease', 'hypertension'
- created current smoker feature
- created age_bin and gluc_bin

## Exploration
functions for exploration contained in explore.py

### Explore Univariate 
#### Univariate Takeaways
- Age is pretty even across the board
- Most work is in private sector
- Avg. glucose and bmi have a right skew, I assume they are related

### Explore Bivariate
#### Bivariate Takeaways
- Good features:
    - hypertension
    - heart disease
    - ever married
    - age
    - glucose
- Bad features:
    - residency
    - gender
    - current smoker
- Need more info:
    - bmi
    - ever_smoked...

## Statistical Analysis
### χ<sup>2</sup> Test
- The χ<sup>2</sup> Test can be used to compare two categorical variables and lets us test the hypothesis that one group is indenpendent of another.
- The χ<sup>2</sup> Test returns the χ<sup>2</sup> statistic and the p-value
    - The χ<sup>2</sup> statistic
        - a single number that tells you how much difference exists between your observed counts and the counts you would expect if there were no relationship at all in the population.
    - the p-value:
        - The probability of obtaining test results at least as extreme as the results actually observed, under the assumption that the null hypothesis is correct


### T-Test
- A T-test allows me to compare a categorical and a continuous variable by comparing the mean of the continuous variable by subgroups based on the categorical variable
- The t-test returns the t-statistic and the p-value:
    - t-statistic: 
        - Is the ratio of the departure of the estimated value of a parameter from its hypothesized value to its standard error. It is used in hypothesis testing via Student's t-test. 
        - It is used in a t-test to determine if you should support or reject the null hypothesis
        - t-statistic of 0 = H<sub>0</sub>
    - the p-value:
        - The probability of obtaining test results at least as extreme as the results actually observed, under the assumption that the null hypothesis is correct
        
### Stats Summary

#### χ<sup>2</sup> Results
- heart_disease, hypertension, and ever_married all rejected the null hypothesis
- It is now assumed that there is a dependency of each variable and stroke.

#### T-test Results
- a two sample one tail t-test was performed on age of those who had a stroke and those who did not have a stroke.
- the null hypothesis was rejected.
- the t-test proved that the age of those who have not had a stroke was significantly less than the age of those who have had a stroke.

- a two sample two tail t-test was performed on average glucose levels of those who had a stroke and those who did not have a stroke.
- the null hypothesis was rejected.

## Modeling
### What am I looking for?
- In these models I will be looking to the ones that produce the highest Recall or Sensitivity. 
- I need the model that produce as many True Positives are False Negatives as possible. 
- Accuracy in this case will not produce the best predictions since it will not capture most people who will have a stroke.

### Model Selection Tools
- During this project I stumbled upon some helpful tool in selecting the hyperparameters for each model. 
- This tool is the GridSearchCV from sklearn.model_selection. 
    - This tool takes in a model, a dictionary of parameters, and a scoring parameter.
    - With a for loop it is easy to see what this tool does
    
### Selected Model
#### KNN Model had the best fit
- Hyperparameters:
    - algorithm='brute' 
    - n_neighbors=1
    - weights='distance'
    
### Model Summary
- My models performed pretty poorly
- The imbalanced data set did not provide enough Stroke positive people to analyze thus making it difficult to see what is happeneing

## Conclusion

With the current dataset, predicting stroke is extremely difficult. When I completed modeling of this data, I realized that finding a really good solution is problematic for a couple reasons.
1. The dataset is far too small. If stroke is the worlds 2<sup>nd</sup> leading cause of death, there should be much more informaiton available. 
2. This dataset is far too imbalanced for a good machine learning algorithm to analyze.
    a. When imblearn is applied the dataset drops from 5000 records to 300. 

What can be done to make this project better?
Collect more stroke victim data, perhaps conducting a large study to gather more patients' data and more data points like family history, blood disorders, etc.