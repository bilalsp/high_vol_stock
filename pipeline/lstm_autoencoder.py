import numpy as np

# tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping

# scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor



class SequenceSegmenter(BaseEstimator, TransformerMixin):

  def __init__(self, window_size:int=3):
    """ Transform time-series sequence into subsequences of given window_size 
    
    Args:
        window_size (int): size of window to create subsequences

    """
    self.window_size = window_size

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    """ Function to transform time-series sequence into subsequences
    
    Args:
        X (numpy.ndarray) of shape = [n_instances, n_features]

    Returns:
        X_trans (numpy.ndarray) of shape = [n_instances, window_size, n_features]

    """
    n_instances = X.shape[0]
    n_features = X.shape[1]

    pad_amnt = np.int(self.window_size)
    X = np.pad(X, [(pad_amnt,0),(0,0)], mode='edge')

    X_trans = np.array([X[i:i+self.window_size] for i in range(n_instances)])

    return X_trans



class StockHyperModel:

  def __init__(self, input_shape):
    """ Class to build the LSTM model

    Args:
        input_shape: for first LSTM layer of model
    """
    self.input_shape = input_shape


  def get_lstm_autoencoder_model(self):
    """ """
    model = Sequential()
    model.add(
        layers.LSTM(
            units=128,
            activation='relu',
            input_shape=self.input_shape
        )
    )
    model.add(layers.Dropout(0.2))
    model.add(layers.RepeatVector(self.input_shape[0]))
    model.add(
        layers.LSTM(
            units=128,
            activation='relu',
            return_sequences=True
        )
    )
    model.add(layers.Dropout(0.2))
    model.add(
        layers.TimeDistributed(
            layers.Dense(self.input_shape[1])
        )
    )  

    return model


  def build(self):
    """Compile Sequential Model"""
    model = self.get_lstm_autoencoder_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,),
        loss="mse"
    )
    
    return model



class LSTMAutoEncoderPipeline:

  def __init__(self, look_back:int, n_features:int, **kwargs):
    self.look_back = look_back
    self.n_features = n_features
    config = kwargs.get('params', {})
    self.params = config.get('params', {})


  def build(self, cachedir=None):
    """ """
    look_back = self.look_back  
    n_features = self.n_features
    params = self.params
   
    estimator = KerasRegressor(
        build_fn=StockHyperModel((look_back,n_features)).build,
        epochs=params.get('epochs',10),
        batch_size=params.get('batch_size',1),
        verbose=params.get('verbose',1),
        validation_split = params.get('validation_split',None),
        shuffle=False
    )

    pipe = Pipeline(
        [
         ('scaling', StandardScaler()),
         ('sequence.segmenter', SequenceSegmenter(look_back)),
         ('regressor', estimator)
        ],
        memory=cachedir
    )

    # Apply transformation on target
    pipe_t = TransformedTargetRegressor(
        regressor=pipe,
        transformer=StandardScaler(),
        check_inverse=False
    )

    return pipe_t




