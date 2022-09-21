from model import *
from format_data import *
import tensorflow as tf

class run():
    def __init__(self,
                batch_size: int = 32):
        self.batch_size = batch_size
        
    def data(self):
        data = format_data()

        X_train, y_train = data.train_data()
        X_val, y_val = data.val_data()
        X_test, y_test = data.test_data()
        return X_train, y_train, X_val, y_val, X_test, y_test

    def develop_model(self):

        X_train, y_train, X_val, y_val = self.data()[:4]

        model = create_model()
        model.summary()

        callback = tf.keras.callbacks.ModelCheckpoint('Transformer+TimeEmbedding.hdf5', 
                                                    monitor='val_loss', 
                                                    save_best_only=True, 
                                                    verbose=1)

        history = model.fit(X_train, y_train, 
                            batch_size=self.batch_size, 
                            epochs= 35, 
                            callbacks=[callback],
                            validation_data=(X_val, y_val))  

        model = tf.keras.models.load_model('/content/Transformer+TimeEmbedding.hdf5',
                                        custom_objects={'Time2Vector': Time2Vector, 
                                                        'SingleAttention': SingleAttention,
                                                        'MultiAttention': MultiAttention,
                                                        'TransformerEncoder': TransformerEncoder})
        return history, model


    def predict(self):

        history, model = self.develop_model()
        X_train, y_train, X_val, y_val, X_test, y_test = self.data()
        ###############################################################################
        '''Calculate predictions and metrics'''

        #Calculate predication for training, validation and test data
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)

        #Print evaluation metrics for all datasets
        train_eval = model.evaluate(X_train, y_train, verbose=0)
        val_eval = model.evaluate(X_val, y_val, verbose=0)
        test_eval = model.evaluate(X_test, y_test, verbose=0)
        print(' ')
        print('Evaluation metrics')
        print('Training Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(train_eval[0], train_eval[1], train_eval[2]))
        print('Validation Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(val_eval[0], val_eval[1], val_eval[2]))
        print('Test Data - Loss: {:.4f}, MAE: {:.4f}, MAPE: {:.4f}'.format(test_eval[0], test_eval[1], test_eval[2]))

        return train_pred, val_pred, test_pred, train_eval, val_eval, test_eval

    def plot_output(self):
        

        ###############################################################################
        '''Display results'''

        fig = plt.figure(figsize=(15,20))
        st = fig.suptitle("Moving Average - Transformer + TimeEmbedding Model", fontsize=22)
        st.set_y(0.92)

        #Plot training data results
        ax11 = fig.add_subplot(311)
        ax11.plot(train_data[:, 3], label='IBM Closing Returns')
        ax11.plot(np.arange(seq_len, train_pred.shape[0]+seq_len), train_pred, linewidth=3, label='Predicted IBM Closing Returns')
        ax11.set_title("Training Data", fontsize=18)
        ax11.set_xlabel('Date')
        ax11.set_ylabel('IBM Closing Returns')
        ax11.legend(loc="best", fontsize=12)

        #Plot validation data results
        ax21 = fig.add_subplot(312)
        ax21.plot(val_data[:, 3], label='IBM Closing Returns')
        ax21.plot(np.arange(seq_len, val_pred.shape[0]+seq_len), val_pred, linewidth=3, label='Predicted IBM Closing Returns')
        ax21.set_title("Validation Data", fontsize=18)
        ax21.set_xlabel('Date')
        ax21.set_ylabel('IBM Closing Returns')
        ax21.legend(loc="best", fontsize=12)

        #Plot test data results
        ax31 = fig.add_subplot(313)
        ax31.plot(test_data[:, 3], label='IBM Closing Returns')
        ax31.plot(np.arange(seq_len, test_pred.shape[0]+seq_len), test_pred, linewidth=3, label='Predicted IBM Closing Returns')
        ax31.set_title("Test Data", fontsize=18)
        ax31.set_xlabel('Date')
        ax31.set_ylabel('IBM Closing Returns')
        ax31.legend(loc="best", fontsize=12)