# Credit Scoring Project

---

## Introduction

This project is part of the Data Scientist path on OpenClassrooms. The main goal is to develop a machine learning model that predicts a client's ability to repay a loan. The final application provides an interactive dashboard for bankers to make informed decisions based on the model's predictions.

## Context and Objectives

- **Context**: The project is based on a dataset from a Kaggle competition, which has been adapted to address our specific credit scoring problem.
- **Objectives**:
  - Build a predictive model to evaluate the probability of a client defaulting on a loan.
  - Develop a web application to present these predictions to the client.

## Data Used

- **Training Data**: `application_train.csv` with 307,511 entries and 122 features.
  - This dataset was used to train and test the models. It is well-labeled with the target variable (0 for a good client, 1 for a defaulting client).
- **Test Data**: `application_test.csv` with 48,744 entries and 121 features.
  - This dataset was used in the web application to predict the default probability for new clients.

## Data Preparation

- **Handling Categorical Variables**:
  - **Label Encoding**: Applied to categorical variables with fewer than three unique values (e.g., gender).
  - **One-Hot Encoding**: Applied to other categorical variables, resulting in the creation of 121 new features.

- **Feature Engineering**:
  - We expanded the original dataset from 122 features to 243 features by generating polynomial features and creating new calculated metrics, such as `CREDIT_INCOME_PERCENT`, `ANNUITY_INCOME_PERCENT`, `CREDIT_TERM`, and `DAYS_EMPLOYED_PERCENT`.

## Business Considerations

This is not a standard classification problem where the goal is simply to assign each client to the most probable class. The objective is to account for the cost implications of false positives (wrongly classifying a good client as a bad one) and false negatives (failing to identify a defaulting client).

- **Key Metric**: Recall score (true positives / (true positives + false negatives)) was selected as the primary metric, with the goal of identifying at least 50% of defaulting clients.

## Predictive Model

- **Data Transformation**:
  - The dataset was split into a training set (70%) and a test set (30%).
  - Missing values were imputed using the median strategy.
  - Features were scaled using MinMaxScaler to range between 0 and 1.

- **Logistic Regression**:
  - The best parameters were identified using Grid Search: `C=10`, `Penalty=l2`.
  - A threshold of 12% was chosen: any client with more than 12% probability of default is flagged as a potential defaulter.

### Model Performance:

- **Recall score**: 0.51
- **Precision score**: 0.20
- **F1 score**: 0.28
- **ROC AUC score**: 0.66

- **Random Forest**:
  - A Random Forest classifier was also tested but yielded similar results, so the logistic regression model was chosen for its interpretability.

## Results and Conclusion

- **Key Findings**:
  - The logistic regression model predicts that any potential client with more than a 12% probability of default should be refused.
  - The model successfully identifies 51% of actual defaulters (recall), but only 20% of the positive predictions (default) are accurate, meaning 16.5% of clients will be wrongly classified as defaulters.

- **Areas for Improvement**:
  - The model's performance is moderate, with room for improvement, especially in handling class imbalance (only 8% of the dataset represents defaulting clients).
  - Further feature engineering based on domain expertise could improve the model.
  - Testing other models, such as XGBoost, could also be beneficial.

## Deployment

The model has been saved as a pickle file and is used on the `application_test.csv` dataset, which represents future loan applicants. The results are presented in a web application designed for bankers.

**Links**:
- [Web Application](https://dbellaiche-oc.herokuapp.com/) (update: the link expired)
