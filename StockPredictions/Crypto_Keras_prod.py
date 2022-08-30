import math
from unicodedata import numeric
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

### Get Data ###

import datetime 
history = datetime.timedelta(days = 50)
trading_pair= 'BTCUSD'

class stockPred:
    def __init__(self, 
                key1: str,
                key2: str,
                past_days: int = 50,
                trading_pair: str = 'BTCUSD',
                exchange: str = 'FTXU',

                metric: str = 'close',
                look_back: int = 72,
                
                neurons: int = 50,
                activ_func: str = 'linear',
                dropout: float = 0.2, 
                loss: str = 'mse', 
                optimizer: str = 'adam',
                epochs: int = 20,
                batch_size: int = 32,
                output_size: int = 1
                ):
        self.key1 = key1
        self.key1 = key2
        import datetime 
        self.history = datetime.timedelta(days = past_days)
        self.trading_pair = trading_pair
        self.exchange = exchange

        self.look_back = look_back
        self.metric = metric

        self.neurons = neurons
        self.activ_func = activ_func
        self.dropout = dropout
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_size = output_size

    def getAllData(self):
        from datetime import datetime
        from datetime import date
        # https://pypi.org/project/alpaca-trade-api/
        # alpaca api key, alpaca secret key
        from alpaca_trade_api.rest import REST, TimeFrame
        client = REST(self.key1, self.key2)
        df = client.get_crypto_bars(self.trading_pair, TimeFrame.Hour, 
                        start = date.today() - self.history, end = date.today()).df
        return df

    def getMetric(self): 
        df = self.getAllData()
        df = df[df.exchange == self.exchange]
        data = df.filter([self.metric])
        data = data.values
        return data

    def scaleData(data):
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaled_data = scaler.fit_transform(data)   
        return scaled_data 

    # train on all data for which labels are available (train + test from dev)
    def getTrainData(self):
        scaled_data = self.scaleData()
        x = []
        y = []
        for price in range(self.look_back, len(self.scaled_data)):
            x.append(scaled_data[price - self.look_back:price, :])
            y.append(scaled_data[price, :])
        return np.array(x), np.array(y)

    def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
        fig, ax = plt.subplots(1, figsize=(13, 7))
        ax.plot(line1, label=label1, linewidth=lw)
        ax.plot(line2, label=label2, linewidth=lw)
        ax.set_ylabel('XRP/USDT', fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.legend(loc='best', fontsize=16)
    # line_plot(train_data[metric], test_data[metric], 'train', 'test', title='')

    def LSTM_model(self, input_data, output_size):

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Activation, Dense, Dropout, LSTM
        from tensorflow.keras import layers
        import matplotlib.pyplot as plt

        model = Sequential()
        model.add(LSTM(self.neurons, input_shape=(input_data.shape[1], input_data.shape[2]), return_sequences = True))
        model.add(Dropout(self.dropout))
        model.add(LSTM(self.neurons, return_sequences = True))
        model.add(Dropout(self.dropout))
        model.add(LSTM(self.neurons))
        model.add(Dropout(self.dropout))
        model.add(Dense(units=output_size))
        model.add(Activation(self.activ_func))

        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def trainModel(self):
        x, y = self.getTrainData()
        x_train = x[: len(x) - 1]
        y_train = y[: len(x) - 1]

        model = self.LSTM_model(x_train, output_size=1, neurons= self.neurons, dropout= self.dropout, loss= self.loss,
        optimizer= self.optimizer)
        modelfit = model.fit(x_train, y_train, epochs= self.epochs, batch_size= self.batch_size, verbose=1, shuffle=True)
        return model, modelfit

    def predictModel(self):
        x = self.getTrainData()[0]
        x_pred = x[-1]

        model = self.trainModel()[0]
        pred = model.predict(x_pred).squeeze()

        return pred
    
    