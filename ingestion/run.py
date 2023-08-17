import os
import psycopg2
import pandas as pd
from ingestion import get_sp500, insert_dataframes
import yfinance as yf


PASSWORD = os.environ.get('PGPASSWORD')  
conn = psycopg2.connect(f"dbname=stocks user=postgres password={PASSWORD}")

string = " ".join(get_sp500()).lower()

df = pd.concat([yf.download(ticker, group_by="Ticker", period='max').assign(ticker=ticker).reset_index() for ticker in get_sp500()])

df = df.rename(columns={"Adj Close": "adj_close"})

insert_dataframes(conn, df, 'ticker_trading_data')