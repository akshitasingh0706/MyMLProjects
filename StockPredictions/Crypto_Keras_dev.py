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
                split_perc: int = .8,
                
                neurons: int = 50,
                activ_func: str = 'linear',
                dropout: float = 0.2, 
                loss: str = 'mse', 
                optimizer: str = 'adam',
                epochs: int = 20,
                batch_size: int = 32,
                output_size: int = 1,

                plot: bool = True,

                mse_thresh: float = .002,
                r2_thresh: float = .8
                ):
        # parameters to retrieve data and filter relevent features
        self.key1 = key1
        self.key2 = key2
        import datetime 
        self.history = datetime.timedelta(days = past_days)
        self.trading_pair = trading_pair
        self.exchange = exchange
        self.feature = feature

        # parameters to set up the time series
        self.look_back = look_back
        self.split_perc = split_perc

        # paramters to define LSTM
        self.neurons = neurons
        self.activ_func = activ_func
        self.dropout = dropout
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_size = output_size

        # plot
        self.plot = plot
        
        # thresholds for model retraining
        self.mse_thresh = mse_thresh 
        self.r2_thresh = r2_thresh

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

    def splitDataforViz(self):
        pass

    def scaleData(self):
        data = self.getFeature()
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(-1,1))
        scaled_data = scaler.fit_transform(data)   
        return scaled_data, scaler

    def getTrainData(self):
        scaled_data = self.scaleData()[0]
        training_data_len = int(np.ceil(len(scaled_data) * self.split_perc))
        look_back = 72
        x_train = []
        y_train = []
        train_data = scaled_data[0:int(training_data_len), :]
        for price in range(look_back, len(train_data)):
            x_train.append(train_data[price - self.look_back:price, :])
            y_train.append(train_data[price, :])
        return np.array(x_train), np.array(y_train)

    def getTestData(self):
        scaled_data = self.scaleData()[0]
        training_data_len = int(np.ceil(len(scaled_data) * self.split_perc))
        test_data = scaled_data[training_data_len - self.look_back: , :]
        x_test = []
        y_test = np.array(scaled_data[training_data_len:, :])
        for price in range(self.look_back, len(test_data)):
            x_test.append(test_data[price - self.look_back:price, 0])
        # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))   
        return np.array(x_test), np.array(y_test)

    def line_plot(self, line1, line2, label1=None, label2=None, title='', lw=2):
        fig, ax = plt.subplots(1, figsize=(13, 7))
        ax.plot(line1, label=label1, linewidth=lw)
        ax.plot(line2, label=label2, linewidth=lw)
        ax.set_ylabel('XRP/USDT', fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.legend(loc='best', fontsize=16)

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
        x_train, y_train = self.getTrainData()
        model = self.LSTM_model(x_train)
        # modelfit = model.fit(
        #     x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
        modelfit = model.fit(
            x_train, y_train, epochs= self.epochs, batch_size= self.batch_size, 
            verbose=1, shuffle=True)
        return model, modelfit

    def predictModel(self):
        from sklearn.metrics import mean_absolute_error
        model = self.trainModel()[0]
        x_test, y_test = self.getTestData()

        preds = model.predict(x_test).squeeze()
        mean_absolute_error(preds, y_test)

        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(preds, y_test)

        from sklearn.metrics import r2_score
        r2 = r2_score(y_test, preds)

        scaler = self.scaleData()[1]
        y_test_true = scaler.inverse_transform(y_test)
        preds_true = scaler.inverse_transform(np.reshape(preds, (preds.shape[0], 1)))
        self.line_plot(y_test_true, preds_true, 'actual', 'prediction')
        return mse, r2