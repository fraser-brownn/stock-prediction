import pandas as pd
import numpy as np
from features import *

def convert_dtypes(df):
    """
    Convert the data type of one or more columns in a pandas DataFrame.
    
    Parameters:
        df (pandas.DataFrame): The DataFrame to modify.
        col_dict (dict): A dictionary of column names and their new data types.
        
    Returns:
        pandas.DataFrame: The modified DataFrame.
    """
    for col_name in df.columns:
        if col_name == 'date':
            df[col_name] = pd.to_datetime(df[col_name])
        elif col_name == 'ticker':
            df[col_name] = df[col_name].astype(object)
            df[col_name] = df[col_name].apply(lambda x: x.strip())
        else:
            df[col_name] = df[col_name].astype(float)
    return df

def percentage_change(df, target_column):
    dfs = []
    for i in df['ticker'].unique():
        df_temp = df[df['ticker'] == i].sort_values(by = ['date'], ascending = True)
        df_temp['pct_change'] = df[target_column].pct_change()
        dfs.append(df_temp)
    df_pct_change = pd.concat(dfs)
    return df_pct_change

def test_output(df):
    return df.to_csv('/Users/fraserbrown/raw_data/test.csv')


def create_dict_of_dfs(df):
    dict_of_dfs = {i: df[df['ticker']==i] for i in df['ticker'].unique()}
    return dict_of_dfs

def create_y_column(df, prediction_interval: int):
    df['y'] = df['adj_close'].shift(-prediction_interval)
    df = convert_dtypes(df).dropna()
    return df

def apply_feature_eng(df):
    df = calculate_rsi(df)
    df = moving_averages(df, [3, 10])
    return df



