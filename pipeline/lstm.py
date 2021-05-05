import numpy as np

# tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping

# scikit-learn
from sklearn.preprocessing import MinMaxScaler
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


  def get_lstm_model(self):
    """ """
    model = Sequential()
    model.add(
        layers.LSTM(
            units=128,
            return_sequences=True,
            input_shape=self.input_shape
        )
    )
    model.add(
        layers.LSTM(
            units=64,
            return_sequences=False
        )
    )
    model.add(layers.Dense(units=25))
    model.add(layers.Dense(units=1))

    return model
   

  def build(self):
    """Compile Sequential Model"""
    model = self.get_lstm_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,),
        loss="mse"
    )
    
    return model



class LSTMPipeline:

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
    )

    pipe = Pipeline(
        [
         ('scaling', MinMaxScaler(feature_range=(0, 1))),
         ('sequence.segmenter', SequenceSegmenter(look_back)),
         ('regressor', estimator)
        ],
        memory=cachedir
    )

    # Apply transformation on target
    pipe_t = TransformedTargetRegressor(
        regressor=pipe,
        transformer=MinMaxScaler(feature_range=(0, 1)),
        check_inverse=False
    )

    return pipe_t




