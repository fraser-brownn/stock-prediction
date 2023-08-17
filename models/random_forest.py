import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
from numpy import arange
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
import copy 
from joblib import dump
from os import PathLike
import pathlib

from ingestion import execute_query
from features import calculate_rsi, moving_averages

from utils import *

import argparse

    
class RandomForestStockPrediction():
    def __init__(self, X_train, X_test, y_train, y_test, ticker, save_model: bool, metrics_: bool, plot: bool):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train  = y_train
        self.y_test = y_test
        self.prediction_output = pd.DataFrame()
        self.ticker = ticker
        self.save_model = save_model
        self.metrics = metrics_
        self.plot = plot
        self.model_object = self.create_model_object()


    def create_model_object(self):
        model = RandomForestRegressor(random_state=42)
        X_training = self.X_train
        y_training = self.y_train
        model.fit(self.X_train, self.y_train)
        if self.save_model == True:
            dump(model, pathlib.Path(f'./models/saved/{self.ticker.strip()}_model.joblib'))
        return model
    
    def evaluation_metrics(self):
        if not self.model_object:
            self.create_model_object()
        else:
            pass

        predict = self.model_object.predict(self.X_test)

        flat_predictions = predict.flatten()
        df_accuracy = copy.deepcopy(self.y_test)
        df_accuracy.columns = ['actuals']
        df_accuracy['predictions'] = flat_predictions
        df_accuracy.columns = ['actuals', 'predictions']
        self.prediction_output = df_accuracy


        if  self.metrics == True:
            print("Mean Absolute Error:", round(metrics.mean_absolute_error(self.y_test, predict), 4))
            print("Mean Squared Error:", round(metrics.mean_squared_error(self.y_test, predict), 4))
            print("Root Mean Squared Error:", round(np.sqrt(metrics.mean_squared_error(self.y_test, predict)), 4))
            print("(R^2) Score:", round(metrics.r2_score(self.y_test, predict), 4))
            print(f'Train Score : {self.model_object.score(self.X_train, self.y_train) * 100:.2f}% \
                  and Test Score : {self.model_object.score(self.X_test, self.y_test) * 100:.2f}% using Random Tree Regressor.')
        if self.plot == True:

            plt.plot(range(0, len(predict)), predict, color = 'g')
            plt.plot(range(0, len(self.y_test)), self.y_test)
            plt.show()

        else:
            pass

if __name__ == "__main__":
    query = '''
            SELECT 
                *
                FROM ticker_trading_data

                where ticker in ('GOOG', 'TSLA', 'AAPL', 'AMZN')
                and date >= current_date - interval '2' year

                order by ticker, date
                    
            '''

    df = execute_query(query)

    df = df.dropna().set_index(['date'])

    df['ticker'] = df['ticker'].str.strip()

    scaler = StandardScaler()

    df_dictionary = create_dict_of_dfs(df)
    # when paramter is equal to 1 then the model is prediciting the close price tomorrow
    df_dictionary_target = {k: create_y_column(apply_feature_eng(v).dropna(), 1) for k, v in df_dictionary.items() }


    print(df_dictionary_target)


    def train_models(dictionary, save):
        model_objects = {}
        for i in dictionary.keys():
            x = dictionary[i].drop(['ticker', 'adj_close','y'], axis =1)
            y = dictionary[i][['y']]     
            split = int(0.95*len(x))
            X_train, X_test, y_train, y_test = x[:split], x[split:], y[:split], y[split:]
            rf = RandomForestStockPrediction(X_train, X_test, y_train, y_test, i, save_model=True, metrics_ = True, plot = False)
            rf.create_model_object()
            model_objects[i] = rf

        return model_objects


    models = train_models(df_dictionary_target, save = False)

    for i in df['ticker'].unique():
        models[i].evaluation_metrics()
        print(f'{i}:', models[i].prediction_output)


    
