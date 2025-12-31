# Iot_predictive_maintenance
This project focuses on predictive maintenance using machine learning.
It covers the full data science workflow—from loading and cleaning data to exploratory data analysis (EDA), model building, and deployment through an interactive dashboard.

The goal is to predict maintenance needs early and help reduce unexpected equipment failures.

Dataset

The dataset is loaded using Python

Contains historical machine or system data

Includes features used to analyze performance and predict maintenance requirements

Project Structure:

Dashboard

Built using Streamlit

Displays key insights from EDA

Allows users to view predictions in an interactive way

Helps visualize maintenance trends and risk

Workflow:

Cleaning and preparing the dataset.
Visualisiing the trends using Matplotlib and Seaborn.
Feature Engineering columns such as r (Appreciation Rate) and investment_type.
Evaluating which model performs best using cross_val_score.
Training the best models.
Saving the necessary files as parquet, encoders and trained models as joblib and feature list as json.
Logged the models, their metrics and parmaters into MLFlow.
Built the streamlit app where users can provide input for the desired specifications of their property and thereby know how much and what kind of investments they are going to make.
An analytics dashboard is also built-in the streamlit app.
