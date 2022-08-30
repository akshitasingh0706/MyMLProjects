import math
from unicodedata import numeric
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

class stockPred:
    def __init__(self, 
                key1: str,
                key2: str,
                past_days: int = 50,
                trading_pair: str = 'BTCUSD',
                exchange: str = 'FTXU',
                feature: str = 'close',
                
                look_back: int = 72,
                
                neurons: int = 50,
                activ_func: str = 'linear',
                dropout: float = 0.2, 
                loss: str = 'mse', 
                optimizer: str = 'adam',
                epochs: int = 20,
                batch_size: int = 32,
                output_size: int = 1,

                retrain_freq: int = 24 # once a day
                ):
        self.key1 = key1
        self.key2 = key2
        import datetime 
        self.history = datetime.timedelta(days = past_days)
        self.trading_pair = trading_pair
        self.exchange = exchange
        self.feature = feature

        self.look_back = look_back

        self.neurons = neurons
        self.activ_func = activ_func
        self.dropout = dropout
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_size = output_size

        self.retrain_freq = retrain_freq

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

    def getFeature(self): 
        df = self.getAllData()
        df = df[df.exchange == self.exchange]
        data = df.filter([self.feature])
        data = data.values
        return data

    def scaleData(self):
        data = self.getFeature()
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaled_data = scaler.fit_transform(data)   
        return scaled_data, scaler

    # train on all data for which labels are available (train + test from dev)
    def getTrainData(self):
        scaled_data = self.scaleData()[0]
        x, y = [], []
        for price in range(self.look_back, len(scaled_data)):
            x.append(scaled_data[price - self.look_back:price, :])
            y.append(scaled_data[price, :])
        return np.array(x), np.array(y)

    def LSTM_model(self, input_data):
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
        model.add(Dense(units=self.output_size))
        model.add(Activation(self.activ_func))

        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def trainModel(self):
        x, y = self.getTrainData()
        x_train = x[: len(x) - 1]
        y_train = y[: len(x) - 1]

        model = self.LSTM_model(x_train)
        modelfit = model.fit(x_train, y_train, epochs= self.epochs, batch_size= self.batch_size, verbose=1, shuffle=True)
        return model, modelfit

    def predictModel(self):
        scaled_data = self.scaleData()[0]
        x_pred = scaled_data[-self.look_back:]
        x_pred = np.reshape(x_pred, (1, x_pred.shape[0]))

        model = self.trainModel()[0]
        pred = model.predict(x_pred).squeeze()
        pred = np.array([float(pred)])
        pred = np.reshape(pred, (pred.shape[0], 1))
        
        scaler = self.scaleData()[1]
        pred_true = scaler.inverse_transform(pred)
        return pred_true
    
    
    