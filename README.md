# CS3244-PG-01: Predicting Resale Prices

## Overview
This project is part of the CS3244 Machine Learning module and focuses on building predictive models to estimate resale prices. It includes extensive data preprocessing, exploratory data analysis (EDA), feature engineering, and modelling using regression techniques.

The goal is to use a combination of categorical and numerical features to build robust predictive models and evaluate their performance. 

## Folder Structure
```
CS3244_Group01/
├── data/
│   ├── boost/                          
│   │   ├── X_test_boost.csv            # Boosting test data
│   │   ├── X_train_boost_part1.csv     # Boosting train data (part 1)
│   │   ├── X_train_boost_part2.csv     # Boosting train data (part 2)
│   │   ├── X_train_boost_part3.csv     # Boosting train data (part 3)
│   ├── raw/                            
│   │   ├── ResaleFlatPricesBasedonApprovalDate19901999.csv 
│   │   ├── ResaleFlatPricesBasedonApprovalDate2000Feb2012.csv 
│   │   ├── ResaleFlatPricesBasedonRegistrationDateFromMar2012toDec2014.csv 
│   │   ├── ResaleFlatPricesBasedonRegistrationDateFromJan2015toDec2016.csv 
│   │   ├── ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv 
│   ├── X_test.csv                      # Test dataset (processed)
│   ├── X_test_removed.csv              # Test dataset with removed features
│   ├── X_train_part1.csv               # Train dataset (part 1, processed)
│   ├── X_train_part2.csv               # Train dataset (part 2, processed)
│   ├── X_train_part3.csv               # Train dataset (part 3, processed)
│   ├── X_train_removed.csv             # Train dataset with removed features
│   ├── y_test.csv                      # Test target values
│   ├── y_train.csv                     # Train target values
├── src/                                
│   ├── catboost_info/                  # CatBoost training logs and metadata
│   ├── Boosting_model.ipynb            # Implementation of LightGBM and CatBoost models
│   ├── Data_Cleaning.ipynb             # Data preprocessing and cleaning script
│   ├── Random_Forest+Decision_Tree.ipynb # Random forest and decision tree models
│   ├── Regression_Models.ipynb         # Linear and KNN regression models
│   ├── neural_network_model.ipynb      # Neural network model with SHAP analysis
│   ├── lgbm_regression_model.txt       # Trained LightGBM model details
├── README.md                           # Project documentation
```
