import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data_path = 'employee_churn_dataset.csv'
dict_path = 'employee_churn_data_dictionary.csv'

df = pd.read_csv(data_path)
data_dict = pd.read_csv(dict_path)

# Display basic info about the dataset
print('--- Dataset Info ---')
print(df.info())
print('\n--- First 5 Rows ---')
print(df.head())
print('\n--- Data Dictionary ---')
print(data_dict)

# Data Preprocessing
# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# Check for duplicates
print(f'Duplicates: {df.duplicated().sum()}')

# Encode categorical variables (example: one-hot encoding)
df_encoded = pd.get_dummies(df, drop_first=True)

# 5. Normalize numerical features (example: MinMax scaling)
num_cols = df.select_dtypes(include=[np.number]).columns
scaler = MinMaxScaler()
df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

print(df_encoded.head())

# # Save preprocessed data for next steps
# df_encoded.to_csv('employee_churn_preprocessed.csv', index=False)
# print("Preprocessing complete. Saved to employee_churn_preprocessed.csv")