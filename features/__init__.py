import pandas as pd

def calculate_rsi(dataframe, window_length=14):
    """
    Calculates the Relative Strength Index (RSI) for a given pandas DataFrame.

    Args:
        dataframe: A pandas DataFrame containing the price data.
        window_length: An integer representing the window length for the RSI calculation (default is 14).

    Returns:
        A pandas DataFrame containing the RSI values.
    """
    dfs = []
    for i in dataframe['ticker'].unique():
        df_temp = dataframe[dataframe['ticker']==i]
        # Calculate the price difference between each day
        delta = df_temp['adj_close'].diff()

        # Get rid of the first row, which is NaN since it has no previous day to compare to
        delta = delta[1:]

        # Make a copy of delta to work with
        up = delta.copy()
        down = delta.copy()

        # Set all negative values in up to 0
        up[up < 0] = 0

        # Set all positive values in down to 0
        down[down > 0] = 0

        # Calculate the rolling mean of up and down using the window length
        roll_up = up.rolling(window_length).mean()
        roll_down = abs(down.rolling(window_length).mean())

        # Calculate the relative strength (RS)
        rs = roll_up / roll_down

        # Calculate the RSI
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Combine the RSI values with the original DataFrame
        df_temp['RSI'] = rsi

        dfs.append(df_temp)
    
    return pd.concat(dfs)


def moving_averages(df, moving_averages:list):
    for j in moving_averages:
        df[f'ma_{j}'] = df['adj_close'].rolling(window = j).mean()
    return df 