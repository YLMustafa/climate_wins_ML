# climate_wins_ML
This project explores how unsupervised and complex machine learning (ML) models can be applied to historical weather data across Europe to identify patterns, predict future weather trends, and assess the impact of climate change over the next 25–50 years.

## Climate Wins Advanced Data Analysis
This repository contains Python scripts and Jupyter notebooks developed to apply advanced machine learning techniques for ClimateWins. The goal is to use unsupervised learning, deep learning, and ensemble models to discover hidden patterns in European weather data, detect unusual weather events, and generate predictive insights for long-term climate planning.

## Project Overview
Developed for ClimateWins, a European nonprofit, this project builds on supervised learning techniques from my previous repo (climate_wins) by exploring practical applications of unsupervised learning and complex ML algorithms. As a data analyst, I applied clustering, dimensionality reduction, neural networks, and ensemble methods to weather data spanning the late 1800s to 2022, collected from 18 European weather stations.

Key objectives included:
- Identifying new patterns in European weather over the last 60 years.
- Detecting extreme weather patterns outside regional norms.
- Predicting the likelihood of unusual weather trends increasing.
- Simulating future weather conditions over the next 25–50 years.
- Assessing the safest regions for human habitation in Europe based on projected climate trends.

## Key Questions Addressed
- Can unsupervised ML uncover hidden patterns in historical European weather data?
- Are extreme weather events increasing, and can we predict their likelihood?
- How do deep learning models, random forests, and support vector machines perform on complex climate predictions?
- Which European regions are likely to remain safest under future climate projections?

## Data Set Information 
- Source: European Climate Assessment & Data Set project
- Period Covered: Late 1800s–2022
- Variables Included:
  - Temperature
  - Humidity
  - Precipitation
  - Snow
  - Global radiation
- Stations: 18 locations across mainland Europe
- Data Quality: Mostly complete, with preprocessing applied
- File Type: CSV (~16MB)
- [Download the temperature data set (.csv, 16.6MB)](https://s3.amazonaws.com/coach-courses-us/public/courses/da-spec-ml/Scripts/A1/Dataset-weather-prediction-dataset-processed.csv)
  
## Tools Used
- Python – data preprocessing, ML, and model evaluation
- Jupyter Notebook – interactive development and visualisation
- Pandas, NumPy – data manipulation and cleaning
- Scikit-learn – clustering, PCA, random forests, and SVM
- TensorFlow, Keras – building CNN, RNN, and GAN models
- Matplotlib, Seaborn, Plotly – visualising trends, clusters, and predictions

## Methods & Models
- Unsupervised Learning Algorithms
  - K-Means Clustering
  - Hierarchical Clustering
  - Principal Component Analysis (PCA)
- Complex ML Models
  - Random Forest Classifier
  - Support Vector Machine (SVM)
  - Convolutional Neural Networks (CNN)
  - Recurrent Neural Networks (RNN)
  - Generative Adversarial Networks (GANs)
- Hyperparameter Tuning
  - Grid search and manual optimisation for model performance
  - Analysis of hyperparameter impact on predictive accuracy

## Evaluation Techniques:
- Silhouette score and cluster validation
- Confusion matrices for supervised comparisons
- Model accuracy and prediction trends over time
- Visualisation of extreme weather pattern detection
- Scenario-based simulations for future climate predictions

## Project Structure
- Data – raw, processed, supervised, and unsupervised datasets
- Notebooks – experiments for clustering, deep learning, and predictive modeling
- Visuals – charts, graphs, and images summarising findings
- Presentation – final report and proposal for ClimateWins

## Final Presentation
Summary of insights and recommendations presented in:

[Download PowerPoint Presentation] (Final_Results.pdf)

