from get_data import *


class format_data:
    def __init__(self,
                batch_size = 32,
                seq_len = 128,
                d_k = 256,
                d_v = 256,
                n_heads = 12,
                ff_dim = 256
                ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.ff_dim = ff_dim


    def preproces_data(self):
        '''Calculate percentage change'''
        df = get_data(**kwargs)

        df['open'] = df['open'].pct_change() # Create arithmetic returns column
        df['high'] = df['high'].pct_change() # Create arithmetic returns column
        df['low'] = df['low'].pct_change() # Create arithmetic returns column
        df['close'] = df['close'].pct_change() # Create arithmetic returns column
        df['volume'] = df['volume'].pct_change()

        ###############################################################################
        '''Create indexes to split dataset'''

        times = sorted(df.index.values)
        last_10pct = sorted(df.index.values)[-int(0.1*len(times))] # Last 10% of series
        last_20pct = sorted(df.index.values)[-int(0.2*len(times))] # Last 20% of series

        ###############################################################################
        '''Normalize price columns'''
        #
        min_return = min(df[(df.index < last_20pct)][['open', 'high', 'low', 'close']].min(axis=0))
        max_return = max(df[(df.index < last_20pct)][['open', 'high', 'low', 'close']].max(axis=0))

        # Min-max normalize price columns (0-1 range)
        df['open'] = (df['open'] - min_return) / (max_return - min_return)
        df['high'] = (df['high'] - min_return) / (max_return - min_return)
        df['low'] = (df['low'] - min_return) / (max_return - min_return)
        df['close'] = (df['close'] - min_return) / (max_return - min_return)

        ###############################################################################
        '''Normalize volume column'''

        min_volume = df[(df.index < last_20pct)]['volume'].min(axis=0)
        max_volume = df[(df.index < last_20pct)]['volume'].max(axis=0)

        # Min-max normalize volume columns (0-1 range)
        df['volume'] = (df['volume'] - min_volume) / (max_volume - min_volume)

        ###############################################################################
        '''Create training, validation and test split'''

        df_train = df[(df.index < last_20pct)]  # Training data are 80% of total data
        df_val = df[(df.index >= last_20pct) & (df.index < last_10pct)]
        df_test = df[(df.index >= last_10pct)]

        # Remove date column
        df_train.drop(columns=['timestamp'], inplace=True)
        df_val.drop(columns=['timestamp'], inplace=True)
        df_test.drop(columns=['timestamp'], inplace=True)

        # Convert pandas columns into arrays
        train_data = df_train.values
        val_data = df_val.values
        test_data = df_test.values
        return train_data, val_data, test_data

    # Training data
    def train_data(self):
        train_data = self.preproces_data()[0]
        X_train, y_train = [], []
        for i in range(self.seq_len, len(train_data)):
            X_train.append(train_data[i-self.seq_len:i]) # Chunks of training data with a length of 128 df-rows
            y_train.append(train_data[:, 3][i]) #Value of 4th column (Close Price) of df-row 128+1
            X_train, y_train = np.array(X_train), np.array(y_train)
        return X_train, y_train

###############################################################################

    # Validation data
    def val_data(self):
        val_data = self.preproces_data()[1]
        X_val, y_val = [], []
        for i in range(self.seq_len, len(val_data)):
            X_val.append(val_data[i-self.seq_len:i])
            y_val.append(val_data[:, 3][i])
        X_val, y_val = np.array(X_val), np.array(y_val)
        return X_val, y_val

###############################################################################

# Test data
    def test_data(self):
        test_data = self.preproces_data()[2]
        X_test, y_test = [], []
        for i in range(self.seq_len, len(test_data)):
            X_test.append(test_data[i-self.seq_len:i])
            y_test.append(test_data[:, 3][i])    
        X_test, y_test = np.array(X_test), np.array(y_test)
        return X_test, y_test
