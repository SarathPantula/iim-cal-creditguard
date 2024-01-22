import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Function to load data
def load_data(filepath):
    return pd.read_csv(filepath)

# Function to preprocess data
def preprocess_data(data):
    # Imputation for missing values
    imputer = SimpleImputer(strategy='mean')
    
    numerical_features = ['loan_amount', 'income', 'age', 'credit_history_length']
    categorical_features = ['employment_type', 'education_level']
    target = 'creditworthy'

    data[numerical_features] = imputer.fit_transform(data[numerical_features])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    X = data.drop(target, axis=1)
    y = data[target]

    return preprocessor, X, y

# Split data into training and testing sets
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)
