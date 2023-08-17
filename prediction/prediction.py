import requests
import json
import numpy as np
from joblib import load
import pathlib
from ingestion import execute_query
from utils import create_y_column, apply_feature_eng
import re
import pandas as pd
import datetime

def extract_numbers_from_string(s):
    # This regular expression pattern matches integers and floating-point numbers.
    pattern = r"[-+]?[.]?[d]+(?:,ddd)[.]?d(?:[eE][-+]?d+)?"
    numbers = re.findall(pattern, s)
    
    # Convert the extracted strings to either int or float.
    for i in range(len(numbers)):
        if "." in numbers[i] or "e" in numbers[i] or "E" in numbers[i]:
            numbers[i] = float(numbers[i])
        else:
            numbers[i] = int(numbers[i])
    
    return numbers

query = '''

     SELECT 
                *
                FROM ticker_trading_data

                where ticker in ('AAPL', 'GOOG', 'TSLA')
                and date >= current_date - interval '2' month
                    
'''

def create_dict_of_dfs(df):
    dict_of_dfs = {i: df[df['ticker']==i] for i in df['ticker'].unique()}
    return dict_of_dfs

def value_func(x):
    return df.shift()
    

df = execute_query(query)

df = df.dropna().set_index(['date'])

df['ticker'] = df['ticker'].str.strip()


df_dictionary = create_dict_of_dfs(df)
# when paramter is equal to 1 then the model is prediciting the close price tomorrow
df_dictionary_target = {k: create_y_column(apply_feature_eng(v).dropna(), 1) for k, v in df_dictionary.items() }

df_input = {k: v.drop(['ticker', 'adj_close','y'], axis =1) for k, v in df_dictionary_target.items()}

print(df_input)

results ={}
for i in df_input.keys():
    request_string ={i: j for i, j in zip(df_input[i].columns, df_input[i].iloc[-1].tolist())}
    print(request_string)
    response = requests.post(url = f'http://127.0.0.1:8000/score/{i.strip()}', json = request_string)
    print(type(response.content))
    results[i] = {'Date': df_input[i].reset_index().iloc[-1]['date'] + datetime.timedelta(days=1), 'Prediction':round(float(re.findall(r"[-+]?\d*\.\d+|\d+", str(response.content))[0]), 2)}



results = pd.DataFrame(results).transpose().reset_index()

results.columns = ['Ticker', 'Date', 'Prediction']

print(results)
