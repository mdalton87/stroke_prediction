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




##########################################################################################

# Stroke Data From Kaggle

##########################################################################################


# Data Obtained from Kaggle 



def wrangle_stroke():
    '''
    This function is used to create a usable dataframe for exploration and creating a stroke prediciting model.
    '''
    # 
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    
    
    df.set_index('id', inplace=True)
    
    # converted ever_married into binary feature
    df.ever_married = df.ever_married.replace('Yes', 1).replace('No', 0)
    
    # imputed mode - replacing "Unknown" with "never smoked"
    df.smoking_status = df.smoking_status.replace('Unknown', 'never smoked')
    
    # creating dummy variables for Residence_type
    dummy_df = pd.get_dummies(df['Residence_type'], drop_first=False)
    
    #rename dummy cols
    dummy_df.columns = ['rural_residence', 'urban_residence']
    
    # creating dummy variables for Residence_type
    dummy_df = pd.get_dummies(df['gender'], drop_first=False)
    
    # #rename dummy cols
    dummy_df.columns = ['is_female', 'is_male', 'other']
    
    # # merge data frames togeter
    df = pd.concat([df, dummy_df], axis= 1)
    
    # # drop "Residence_type" column
    df.drop(columns=['gender','other'], inplace=True)
    
    # merge data frames togeter
    df = pd.concat([df, dummy_df], axis= 1)
    
    # drop "Residence_type" column
    df.drop(columns=['Residence_type'], inplace=True)
    
    # establish features to impute using knn 
    features = ['bmi','age','avg_glucose_level','heart_disease','hypertension']
    
    # impute knn for bmi
    w.impute_knn(df, features, 4)
    
    # Create column of current smokers, converted boolean to binary
    df = df.assign(current_smoker = df.smoking_status == 'smokes')
    df.current_smoker = df.current_smoker.replace(True, 1).replace(False, 0).astype('int')
    


    
    return df
    
    
    
    
    
    


##########################################################################################

# Zero's and NULLs

##########################################################################################



#----------------------------------------------------------------------------------------#
###### Identifying Zeros and Nulls in columns and rows

def missing_zero_values_table(df):
    '''
    This function tales in a dataframe and counts number of Zero values and NULL values. Returns a Table with counts and percentages of each value type.
    '''
    zero_val = (df == 0.00).astype(int).sum(axis=0)
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)
    mz_table = mz_table.rename(
    columns = {0 : 'Zero Values', 1 : 'NULL Values', 2 : '% of Total NULL Values'})
    mz_table['Total Zero\'s plus NULL Values'] = mz_table['Zero Values'] + mz_table['NULL Values']
    mz_table['% Total Zero\'s plus NULL Values'] = 100 * mz_table['Total Zero\'s plus NULL Values'] / len(df)
    mz_table['Data Type'] = df.dtypes
    mz_table = mz_table[
        mz_table.iloc[:,1] >= 0].sort_values(
    '% of Total NULL Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"      
        "There are " + str((mz_table['NULL Values'] != 0).sum()) +
          " columns that have NULL values.")
    #       mz_table.to_excel('D:/sampledata/missing_and_zero_values.xlsx', freeze_panes=(1,0), index = False)
    return mz_table



def missing_columns(df):
    '''
    This function takes a dataframe, counts the number of null values in each row, and converts the information into another dataframe. Adds percent of total columns.
    '''
    missing_cols_df = pd.Series(data=df.isnull().sum(axis = 1).value_counts().sort_index(ascending=False))
    missing_cols_df = pd.DataFrame(missing_cols_df)
    missing_cols_df = missing_cols_df.reset_index()
    missing_cols_df.columns = ['total_missing_cols','num_rows']
    missing_cols_df['percent_cols_missing'] = round(100 * missing_cols_df.total_missing_cols / df.shape[1], 2)
    missing_cols_df['percent_rows_affected'] = round(100 * missing_cols_df.num_rows / df.shape[0], 2)
    
    return missing_cols_df


#----------------------------------------------------------------------------------------#
###### Do things to the above zeros and nulls ^^

def handle_missing_values(df, prop_to_drop_col, prop_to_drop_row):
    '''
    This function takes in a dataframe, 
    a number between 0 and 1 that represents the proportion, for each column, of rows with non-missing values required to keep the column, 
    a another number between 0 and 1 that represents the proportion, for each row, of columns/variables with non-missing values required to keep the row, and returns the dataframe with the columns and rows dropped as indicated.
    '''
    # drop cols > thresh, axis = 1 == cols
    df = df.dropna(axis=1, thresh = prop_to_drop_col * df.shape[0])
    # drop rows > thresh, axis = 0 == rows
    df = df.dropna(axis=0, thresh = prop_to_drop_row * df.shape[1])
    return df



# def impute_mode(df, col, strategy):
#     '''
#     impute mode for column as str
#     '''
#     train, validate, test = train_validate_test_split(df, seed=42)
#     imputer = SimpleImputer(strategy=strategy)
#     train[[col]] = imputer.fit_transform(train[[col]])
#     validate[[col]] = imputer.transform(validate[[col]])
#     test[[col]] = imputer.transform(test[[col]])
#     return train, validate, test



def impute_mode(df, cols):
    ''' 
    Imputes column mode for all missing data
    '''
    for col in cols:
        df = df.fillna(df[col].value_counts().index[0])
        return df


def impute_knn(df, list_of_features, knn):
    '''
    This function performs a kNN impute on a single column and returns an imputed df.
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


#----------------------------------------------------------------------------------------#
###### Removing outliers


def remove_outliers(df, col, multiplier):
    '''
    The function takes in a dataframe, column as str, and an iqr multiplier as a float. Returns dataframe with outliers removed.
    '''
    q1 = df[col].quantile(.25)
    q3 = df[col].quantile(.75)
    iqr = q3 - q1
    upper_bound = q3 + (multiplier * iqr)
    lower_bound = q1 - (multiplier * iqr)
    df = df[df[col] > lower_bound]
    df = df[df[col] < upper_bound]
    return df

##########################################################################################

# Feature Engineering

##########################################################################################


#----------------------------------------------------------------------------------------#
###### Adding Features

def create_features(df):
    '''
    Creates the folowing features: age, age_bin, taxrate, acres, acres_bin, sqft_bin, structure_dollar_per_sqft, structure_dollar_sqft_bin, land_dollar_per_sqft, lot_dollar_sqft_bin, bath_bed_ratio, and cola. 
    '''
    

    return df









##########################################################################################

# Split Data

##########################################################################################



####### PICK ONE OF THE METHODS OF SPLITTING DATA BELOW, NOT BOTH



# 1.)
#----------------------------------------------------------------------------------------#
def train_validate_test_split(df, target, seed):
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
                                            # stratify=df[target]
                                           )
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       # stratify=train_validate[target]
                                      )
    return train, validate, test



def train_validate_test_scale(train, validate, test, quant_vars):
    ''' 
    This function takes in the split data: train, validate, and test along with a list of quantitative variables, and returns scaled data for exploration and modeling
    '''
    scaler = MinMaxScaler()
    scaler.fit(train[quant_vars])
    
    X_train_scaled = scaler.transform(train[quant_vars])
    X_validate_scaled = scaler.transform(validate[quant_vars])
    X_test_scaled = scaler.transform(test[quant_vars])

    train[quant_vars] = X_train_scaled
    validate[quant_vars] = X_validate_scaled
    test[quant_vars] = X_test_scaled
    return train, validate, test
#----------------------------------------------------------------------------------------#




# 2.)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
def train_validate_test(df, target):
    '''
    this function takes in a dataframe and splits it into 3 samples, 
    a test, which is 20% of the entire dataframe, 
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe. 
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable. 
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test. 
    '''
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test



def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # create a mask of columns whether they are object type or not
    mask = np.array(df.dtypes == "object")
        
    # get a list of the column names that are objects (from the mask)
    object_cols = df.iloc[:, mask].columns.tolist()
    
    return object_cols



def get_numeric_X_cols(X_train, object_cols):
    '''
    takes in a dataframe and list of object column names
    and returns a list of all other columns names, the non-objects. 
    '''
    numeric_cols = [col for col in X_train.columns.values if col not in object_cols]
    
    return numeric_cols



def min_max_scale(X_train, X_validate, X_test, numeric_cols):
    '''
    this function takes in 3 dataframes with the same columns, 
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler. 
    it returns 3 dataframes with the same column names and scaled values. 
    '''
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).


    scaler = MinMaxScaler(copy=True).fit(X_train[numeric_cols])

    #scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train. 
    # 
    X_train_scaled_array = scaler.transform(X_train[numeric_cols])
    X_validate_scaled_array = scaler.transform(X_validate[numeric_cols])
    X_test_scaled_array = scaler.transform(X_test[numeric_cols])

    # convert arrays to dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled_array, 
                                  columns=numeric_cols).\
                                  set_index([X_train.index.values])
    X_validate_scaled = pd.DataFrame(X_validate_scaled_array, 
                                     columns=numeric_cols).\
                                     set_index([X_validate.index.values])
    X_test_scaled = pd.DataFrame(X_test_scaled_array, 
                                 columns=numeric_cols).\
                                 set_index([X_test.index.values])

    
    return X_train_scaled, X_validate_scaled, X_test_scaled
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#