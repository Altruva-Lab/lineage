"""Chapter 4.3.1: Linear Regression as a Learning Setup.

This file demonstrates linear regression in a practical learning setup:
- How to prepare data
- How to train and evaluate the model
- How to make predictions on new data

We'll use the **California housing dataset** (a classic for regression) loaded via `sklearn.datasets`.
"""

# IMPORT MODULES

# data manipulation
import pandas as pd
import numpy as np
# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# preprocessing and modeling
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# dataset
from sklearn.datasets import fetch_california_housing


# load dataset
df = fetch_california_housing()

# view dataset
# df
# print(df.data)
# print(df.describe)

# check missing data
df.isnull().sum()
df.isnull().sum().sum()

# check duplicates
df.duplicate().sum()

# check dataframe columns
