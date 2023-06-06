# Machine Learning Sub-Featue Selection with Genetic Algorithm

## Introduction and Motivation

- Many applications of machine learning arise from complex relationships between features
- Feature selection, is the process of finding the most relevant input for a model
- Problem: requires lots of computational work and, if the number of features is big, becomes impracticable
- The goal: select the most relevant sub-feature

![image](https://github.com/voidlotus/ml-subfeature-selection-genetic-algorithm/assets/64185555/44eefebf-3916-4227-bf41-5f902b56fc5f)

## Initialization

- Create and initialize the individuals in the population (random)
- Dataset composed of 9 features, each Individual is then composed of 9 elements.
- Design variables are the inclusion (1) or the exclusion (0) of the input variable.
Example: 
![image](https://github.com/voidlotus/ml-subfeature-selection-genetic-algorithm/assets/64185555/92957873-23b8-4f10-bd6a-187fee8c2388)

## Fitness Assignment
- Logistic Regression (Classification task)
- Evaluate Accuracy
- K-Fold Cross Validation (k = 5)

![image](https://github.com/voidlotus/ml-subfeature-selection-genetic-algorithm/assets/64185555/06240aee-4103-4300-a64d-c97415d2ded3)

## Selection and Crossover
- Conduct tournament
- Crossover 
  - Generate a random number between 0 and 1 (for each variable) 
  - If number < 0.5, switch variables

![image](https://github.com/voidlotus/ml-subfeature-selection-genetic-algorithm/assets/64185555/4c7f673d-59ad-4445-8bf3-e55b439ce8e2)

## Mutation
- Self Adaptive mutation
- Generate a random number between 0 and 1  
- Flip variable if generated number < mutation rate

![image](https://github.com/voidlotus/ml-subfeature-selection-genetic-algorithm/assets/64185555/3cb817b6-38cb-4cda-8e57-fbf9313b70bd)

## Result
![image](https://github.com/voidlotus/ml-subfeature-selection-genetic-algorithm/assets/64185555/8764ceda-005b-4536-9b30-2c645f9e6c91)

![image](https://github.com/voidlotus/ml-subfeature-selection-genetic-algorithm/assets/64185555/fdfe5b5e-9d26-4c55-bf38-0cd369f3671e)







