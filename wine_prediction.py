import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Read the dataset
df = pd.read_csv('/content/winequality-red.csv', on_bad_lines='skip')
df.head()

# Check the information about the dataset
df.info()

# Descriptive statistics of the dataset
df.describe().T

# Shape of the dataset (number of rows and columns)
df.shape

# Check for missing values
df.isnull().sum()

# Get the column names
df.columns

# Fill missing values with the mean of each column
for col in df.columns:
  if df[col].isnull().sum() > 0:
    df[col] = df[col].fillna(df[col].mean())

# Check if there are any missing values remaining
df.isnull().sum().sum()

# Plot histograms for each feature
df.hist(bins=10, figsize=(10, 15))
plt.show()

# Plot a bar chart of quality vs alcohol
plt.bar(df['quality'], df['alcohol'])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()

# Check the correlation between features using a heatmap
plt.figure(figsize=(12, 12))
sb.heatmap(df.corr() > 0.7, annot=True, cbar=False)
plt.show()

# Create a new column 'best quality' based on a threshold
df['best quality'] = [1 if x > 5 else 0 for x in df.quality]

# Separate features and target variables
features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(
	features, target, test_size=0.2, random_state=40)

# Check the shapes of the training and testing sets
xtrain.shape, xtest.shape

# Normalize the features using MinMaxScaler
norm = MinMaxScaler()
xtrain = norm.fit_transform(xtrain)
xtest = norm.transform(xtest)

# Define a list of models to evaluate
models = [LogisticRegression(), SVC(kernel='rbf')]

# Train and evaluate each model
for i in range(2):
	models[i].fit(xtrain, ytrain)

	print(f'{models[i]}:')
	print('Training Accuracy:', metrics.roc_auc_score(ytrain, models[i].predict(xtrain)))
	print('Validation Accuracy:', metrics.roc_auc_score(ytest, models[i].predict(xtest)))
	print()
