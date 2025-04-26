# %% [markdown]
# # Project Configuration 
# This section will setup the project this includes:
# - Installing the required libraries
# - Loading the dataset
# - EDA
# - Data Preprocessing
# 
# After this section, the dataset will be used to train two different models: KNN and Logistic Regression.
# 
# Both models will be configured with different hyperparameters, to see which hyperparameters are the best for each model.
# 
# A comparison will be made between the two models, to see which model is the best for this dataset. The metrics used for this comparison are: accuracy, precision, recall and f1-score.

# %% [markdown]
# ## 1. Importing Libraries

# %%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

# Setup visualization config
sns.set()

# %% [markdown]
# ## 2. Loading the dataset and EDA (Exploratory Data Analysis)
# Each piece of code is in a separate cell because otherwise it will only show the last one.

# %%
# Load the weather dataset
weather_data = pd.read_csv('../data/weather-aus.csv')
# Show first 5 rows of the dataset and useful information
weather_data.head() 

# %%
weather_data.info(verbose=True)

# %%
weather_data.describe().T

