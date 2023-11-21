# LENDING CLUB DATASET

## Introduction 

This data science project aims at automating lending decisions for a Lending Club. This dataset was provided direct by turing college on the following [link](https://storage.googleapis.com/335-lending-club/lending-club.zip)

Similar datasets for the lending club can be found directly on kaggle.

**What We Aim to Achieve**:

- **Exploration**: We will delve into the dataset to extract valuable insights and prepare the data for further analysis.

- **Understanding Data**: We will conduct an Exploratory Data Analysis (EDA) to uncover the underlying characteristics of the dataset and explore relationships between variables.

- **Statistical Insight**: Our journey will include statistical inference to test hypotheses related to the rejected loan applications.

- **Machine Learning Models**: We will develop and refine machine learning models designed to predict whether a loan application is accepted or rejected.

- **Real-world Deployment**: Our final goal is to deploy the best-performing machine learning model on Google Cloud Platform for practical, real-world usage.

- **Recommendations**: Throughout our exploration, we will provide valuable insights and recommendations to enhance our analysis and model performance.

**Specific Objective**:
1. Build a machine learning **model to classify loans as accepted or rejected**.

2. **Predict the loan grade**.

3. **Predict the loan subgrade and interest rate**.


## Results

The model for prediction of loans into Accepted and Rejected classified the requests correctly 93% of the time as rejected and 99% of the time as accepted (Decision Tree chosen).

Building the prediction of the Loan Grade proved to be a much harder task, our models did not provided good prediction levels. This showed specially the importance of understanding the factors that influence this business parameter.

## Deployment

Our model was successfully depolyed on Google Cloud and can be accessed through the link: https://loan-app-7ffjztyv7q-lm.a.run.app/
