import pandas as pd
import numpy as np
import os
import explore as ex

# acquire
from env import host, user, password
from pydataset import data

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression, RFE, SelectKBest
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler




####################
####################
# Get Data

def wrangle_stroke():
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    
    
    
    
    
    
    
    
def impute_knn(df, list_of_features, knn):
    '''
    This function performs a kNN impute on a dataframe and returns an imputed df.
    Parameters: df: dataframee
    list_of_features: a List of features, place the feature intended for impute first, then supporting features after.
    knn: an integer, indicates number of neighbors to find prior to selecting imputed value.
    '''
    knn_cols_df = df[list_of_features]
    imputer = KNNImputer(n_neighbors=knn)
    imputed = imputer.fit_transform(knn_cols_df)
    imputed = pd.DataFrame(imputed, index = df.index)
    df[list_of_features[0]] = imputed[[0]]
    return df



def train_validate_test_split(df, target, seed=42):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target]
                                           )
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target]
                                      )
    return train, validate, test



def scale_my_data(train, validate, test, quant_vars):
    scaler = MinMaxScaler()
    scaler.fit(train[quant_vars])
    
    X_train_scaled = scaler.transform(train[quant_vars])
    X_validate_scaled = scaler.transform(validate[quant_vars])
    X_test_scaled = scaler.transform(test[quant_vars])

    train[quant_vars] = X_train_scaled
    validate[quant_vars] = X_validate_scaled
    test[quant_vars] = X_test_scaled
    
    return train, validate, test