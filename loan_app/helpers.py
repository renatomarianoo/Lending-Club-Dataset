import pandas as pd
import pickle
from rapidfuzz import process
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder
from feature_engine.encoding import RareLabelEncoder
from sklearn.compose import ColumnTransformer

# Define classes for pipeline
class CreateHasIncomeColumn():
    '''Custom transformer to create 'has_Income' column'''
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['has_Income'] = X[self.column].apply(lambda x: 'yes' if x != -1 else 'no')
        return X


class CreateProvidedRiskScoreColumn():
    '''Custom transformer for creating 'provided_Risk_Score' column and attributing -1 for NA values and for 0 (out of range)'''
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.column][(X[self.column].isna()) | (X[self.column] == 0)] = -1
        X['provided_Risk_Score'] = X[self.column].apply(lambda x: 'no' if x == -1 else 'yes')
        return X


class ApplyMapToLoanTitles():
    '''
    Groups together elements with similar names based on the provided mapping.
    Orginal column is dropped.
    '''
    def __init__(self, column, mapping, similarity_threshold):
        self.column = column
        self.mapping = mapping
        self.similarity_threshold = similarity_threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cleaned_column = X[self.column].str.lower().str.strip().str.replace("-", " ").str.replace("_", " ")
        cleaned_column = cleaned_column.apply(lambda x: process.extractOne(x, self.mapping.keys()))
        cleaned_column = cleaned_column.apply(lambda x: self.mapping[x[0]] if x and x[1] >= self.similarity_threshold else "Other")
        
        X = X.drop(self.column, axis=1)
        X['Loan_Title_Grouped'] = cleaned_column
        return X


class ApplyLabelToEmploymentLength():
    '''Change the values in employment length to highlight the natural order of this feature'''
    def __init__(self, column, mapping):
        self.column = column
        self.mapping = mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.column] = X[self.column].map(self.mapping)
        return X    

# Define pipeline mappings
title_mapping = {
    "Debt consolidation": "Debt consolidation",
    "Credit card refinancing": "Credit card",
    "Car": "Car",
    "Major purchase": "Major purchase",
    "Home": "Home",
    "Medical expenses": "Medical expenses",
    "moving": "Moving",
    "business": "Business",
}

empl_length_mapping = {
    "< 1 year": 0,
    "1 year": 1,
    "2 years": 2,
    "3 years": 3,
    "4 years": 4,
    "5 years": 5,
    "6 years": 6,
    "7 years": 7,
    "8 years": 8,
    "9 years": 9,
    "10+ years": 10,
}
