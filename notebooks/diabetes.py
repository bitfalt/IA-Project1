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
# Load the diabetes dataset
diabetes_data = pd.read_csv('../data/diabetes.csv')
# Show first 5 rows of the dataset and useful information
diabetes_data.head()

# %%
diabetes_data.info(verbose=True)

# %%
diabetes_data.describe().T

# %% [markdown]
# From initial exploration we can see that some data has a 0 where it makes no sense. For example, Glucose, Blood Pressure, SkinThickness, and BMI cannot be 0.

# %%
diabetes_data = diabetes_data[(diabetes_data['Glucose'] != 0) & (diabetes_data['BloodPressure']!= 0) & (diabetes_data['SkinThickness']!= 0) & (diabetes_data['BMI']!= 0)]
diabetes_data.describe().T

# %% [markdown]
# ## Visualiation of Outliers
# Since there are a lot of columns, we're going to split the data into two groups to visualize the outliers, if any.

# %%
plt.figure(figsize=(20, 5))
first_group_check = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness']
second_group_check = ['Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
for i, col in enumerate(first_group_check, 1):
    plt.subplot(1, len(first_group_check), i)
    sns.boxplot(x=diabetes_data[col])
    plt.title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()

# %%
plt.figure(figsize=(20, 5))
for i, col in enumerate(second_group_check, 1):
    plt.subplot(1, len(second_group_check), i)
    sns.boxplot(x=diabetes_data[col])
    plt.title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()

# %% [markdown]
# Find any outliers using IQR and visualize them with a scatter plot.

# %%
def find_outliers_iqr(data, column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# %%
outliers_first_group = []
outliers_second_group = []

for column in first_group_check:
    outliers, lower_bound, upper_bound = find_outliers_iqr(diabetes_data, column)
    outliers_first_group.append((outliers, lower_bound, upper_bound))

# TODO: figure out how to plot the outliers properly, if it is needed to do one by one or all together
plt.figure(figsize=(20, 12))
# for i, column in enumerate(first_group_check):
#     plt.scatter(range(len(diabetes_data)), diabetes_data[column], label=f'{column} values')
#     plt.scatter(outliers_first_group[i][0].index, outliers_first_group[i][0][column], color='red', label=f'{column} Outliers')
#     plt.title(f'Scatter plot for {column} with outliers highlighted')
#     plt.xlabel('Index')
#     plt.ylabel(column)
#     plt.legend()

# plt.show()


plt.scatter(range(len(diabetes_data)), diabetes_data['Pregnancies'], label='Pregnancies values')
plt.scatter(outliers_first_group[0][0].index, outliers_first_group[0][0]['Pregnancies'], color='red', label='Pregnancies Outliers')
plt.title('Scatter plot for Pregnancies with outliers highlighted')
plt.xlabel('Index')
plt.ylabel('Pregnancies')
plt.legend()
plt.show()



# %%
for column in second_group_check:
    outliers, lower_bound, upper_bound = find_outliers_iqr(diabetes_data, column)
    outliers_second_group.append((outliers, lower_bound, upper_bound))
    


