# pands-project 2025

## Programming & Scripting Project- Analysis of the Iris Flower Data Set

#### Author : Finian Doonan

## Introduction

This repository contains the pands project for the Programming and Scripting module, This forms part of the ATU HDIP in Data Analytics 2025.
This repository explores and analyzes the Iris Data Set. It includes data introduction, manipulation, visualizations like histograms and scatter plots, and a summary of the findings.


Write a program called analysis.py that:
1. Outputs a summary of each variable to a single text file,
2. Saves a histogram of each variable to png files, and
3. Outputs a scatter plot of each pair of variables.
4. Performs any other analysis you think is appropriate.

### breakdown analysis

1. Import Libraries

- Load essential Python libraries such as pandas, numpy, matplotlib, and seaborn.​

2. Inspect Variable Types

- Examine the data types of each column to understand the structure of the dataset.​

3. Data Exploration

- Display the first few rows to get an initial glimpse of the data.​

- Check the dataset's shape to determine the number of rows and columns.​

- Examine specific rows to understand individual data entries.​

- Analyze the distribution of the target variable, 'species', to see the balance among classes.​

- Compute summary statistics for numerical variables to assess central tendencies and dispersions.​

4. Generate and Save Variable Summaries

- Create descriptive statistics for each variable and save them for reference.​

5. Create Histograms for Each Variable

- Visualize the distribution of each numerical feature to identify patterns and potential anomalies.​

6. Scatter Plots of Variable Pairs by Species

- Plot pairwise relationships between numerical variables, color-coded by species, to observe potential correlations and class separations.​

7. Correlation Analysis & Heatmap

- Compute the correlation matrix to quantify relationships between numerical variables.​

- Visualize the correlation matrix using a heatmap to easily identify strong correlations.​

8. Box Plots for Outlier Detection

- Generate box plots for each numerical variable to detect and visualize outliers.​

9. Pair Plot Visualization

- Create pair plots to provide a comprehensive view of relationships between all pairs of variables, differentiated by species.