
from datetime import datetime
from datetime import date

# https://pypi.org/project/alpaca-trade-api/
# alpaca api key, alpaca secret key

import math
from unicodedata import numeric
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

class get_data:
    def __init__(self,
                key1: str,
                key2: str,
                past_days: int = 50,
                trading_pair: str = 'BTCUSD',
                exchange: str = 'FTXU',
                feature: str = 'close',
                ):
        self.key1 = key1
        self.key2 = key2
        import datetime 
        self.history = datetime.timedelta(days = past_days)
        self.trading_pair = trading_pair
        self.exchange = exchange
        self.feature = feature

    def get_data(key1, key2, start):
        from alpaca_trade_api.rest import REST, TimeFrame
        client = REST(key1, key2)
        trading_pair= 'BTCUSD'
        df = client.get_crypto_bars(trading_pair, TimeFrame.Hour, start = '2022-06-01', end = date.today()).df
        df = df[df.exchange == 'FTXU']
        df = df.drop(['exchange', 'trade_count', 'vwap'], axis = 1)
        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].rolling(40).mean() 
        df = df.reset_index(level=0)
        return df

