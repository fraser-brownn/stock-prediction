DROP TABLE ticker_trading_data;

CREATE TABLE IF NOT EXISTS ticker_trading_data (
    date                                                date
    ,open                                           decimal(15,4)
    ,high                                           decimal(15,4)
    ,low                                            decimal(15,4)
    ,close                                          decimal(15,4)
    ,adj_close                                      decimal(15,4)
    ,volume                                         bigint
    ,ticker                                         CHAR(5)
  );

