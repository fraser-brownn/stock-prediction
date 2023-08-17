import psycopg2
import os
import pandas as pd
import psycopg2.extras as extras
import yfinance as yf

def get_sp500():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp = pd.read_html(url, header=0)[0]
    print(list(sp['Symbol']))
    return list(sp['Symbol'])

def execute_query(query):
    PASSWORD = os.environ.get('PGPASSWORD') 
    conn = psycopg2.connect(f"dbname=stocks user=postgres password={PASSWORD}")
    cur = conn.cursor()
    cur.execute(query)
    column_names = [col_name[0] for col_name in cur.description]
    data = cur.fetchall()
    cur.close()
    conn.close()
    df = pd.DataFrame(data).drop_duplicates()
    df.columns = column_names
    return df

def insert_dataframes(conn, df, table):
    '''
    INSERTS DATAFRAMES INTO PRE BUILT SQL TABLES
    
    '''
    
    tuples = [tuple(x) for x in df.to_numpy()]
  
    cols = ', '.join(list(df.columns))
    # SQL query to execute
    query = "INSERT INTO %s (%s) VALUES %%s" % (table, cols)
    cursor = conn.cursor()
    try:
        extras.execute_values(cursor, query, tuples)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    print("the dataframe is inserted")
    cursor.close()

if __name__ == '__main__':  
    PASSWORD = os.environ.get('PGPASSWORD')  
    conn = psycopg2.connect(f"dbname=stocks user=postgres password={PASSWORD}")

    string = " ".join(get_sp500()).lower()

    df = pd.concat([yf.download(ticker, group_by="Ticker", period='max').assign(ticker=ticker).reset_index() for ticker in get_sp500()])

    df = df.rename(columns={"Adj Close": "adj_close"})

    insert_dataframes(conn, df, 'ticker_trading_data')