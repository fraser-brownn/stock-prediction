from ingestion import execute_query
from utils import create_dict_of_dfs, create_y_column, convert_dtypes, apply_feature_eng
from models.random_forest import RandomForestStockPrediction
import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler
import argparse 
import warnings




def print_model_evaluation_metrics(model_object_dictionary, original_df):
    for i in original_df['ticker'].unique():
        model_object_dictionary[i].evaluation_metrics()
        print(f'{i}:', model_object_dictionary[i].prediction_output)


def train_models(dictionary, save: bool, metrics_: bool, plot: bool):
    model_objects = {}
    for i in dictionary.keys():
        x = dictionary[i].drop(['ticker', 'adj_close','y'], axis =1)
        y = dictionary[i][['y']]     
        split = int(0.95*len(x))
        X_train, X_test, y_train, y_test = x[:split], x[split:], y[:split], y[split:]
        rf = RandomForestStockPrediction(X_train, X_test, y_train, y_test, i, save_model=save, metrics_ = metrics_, plot = plot)
        rf.create_model_object()
        model_objects[i] = rf

    return model_objects

def run(save, metrics_, plot):
    query = '''
        SELECT 
            *
            FROM ticker_trading_data

            where ticker in ('GOOG', 'TSLA', 'AAPL', 'AMZN', 'AXP')
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


    models = train_models(df_dictionary_target, save = True, metrics_= True, plot = False)
    print_model_evaluation_metrics(models, df)







if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='StockPrediction',
                    description='Build, Evaluate and Save Stock Prediction Models to be Deployed via FastAPI',
                    )
    parser.add_argument('--save')
    parser.add_argument('--plot')
    parser.add_argument('--metrics')

    args = parser.parse_args()

    run(bool(args.save), bool(args.metrics), bool(args.plot))


    