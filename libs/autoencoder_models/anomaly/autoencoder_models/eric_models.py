
from keras.layers import Input, Dense, LSTM, TimeDistributed, RepeatVector, Conv1D, BatchNormalization, \
    MaxPooling1D, UpSampling1D, Flatten, Reshape, GRU
from keras.models import Model, Sequential
from keras import regularizers
import numpy as np
'''
COPIED/EDITED FROM ERIC MORENO,
https://github.com/eric-moreno/Anomaly-Detection-Autoencoder/blob/master/ANN-Autoencoder/model.py
'''

def autoencoder_LSTM(input_shape:tuple, bottleneck:int):
    print("got input shape", input_shape)
    encoder = Sequential()
    encoder.add(Input(shape=(input_shape[0], input_shape[1])))
    encoder.add(LSTM(32, activation='tanh', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00)))
    encoder.add(LSTM(bottleneck, activation='tanh', return_sequences=False))

    print(encoder.summary())

    decoder = Sequential()
    decoder.add(Input(shape=(encoder.output.shape[1:])))
    decoder.add(RepeatVector(input_shape[0]))
    decoder.add(LSTM(bottleneck, activation='tanh', return_sequences=True))
    decoder.add(LSTM(32, activation='tanh', return_sequences=True))
    decoder.add(TimeDistributed(Dense(input_shape[1])))

    autoencoder = Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs))
    print(decoder.summary()) 
    print(autoencoder.summary())
    return autoencoder, encoder, decoder

def autoencoder_LSTM_deep(X, bandwidth):
    X = np.zeros(shape=(1, X[0], X[1]))
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(64, activation='tanh', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(32, activation='tanh', return_sequences=True)(L1)
    L3 = LSTM(8, activation='tanh', return_sequences=False)(L2)
    L4 = RepeatVector(X.shape[1])(L3)
    L5 = LSTM(8, activation='tanh', return_sequences=True)(L4)
    L6 = LSTM(32, activation='tanh', return_sequences=True)(L5)
    L7 = LSTM(64, activation='tanh', return_sequences=True)(L6)
    output = TimeDistributed(Dense(X.shape[2]))(L7)    
    model = Model(inputs=inputs, outputs=output)
    return model, None, None

def autoencoder_LSTM_big(X, bandwidth=16):
    X = np.zeros(shape=(1, X[0], X[1]))
    inputs = Input(shape=(X.shape[1], X.shape[2]))

    L1 = LSTM(64, activation='tanh', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(bandwidth, activation='tanh', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(bandwidth, activation='tanh', return_sequences=True)(L3)
    L5 = LSTM(64, activation='tanh', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model, None, None

def autoencoder_LSTM_big_CNN(X, bandwidth=16):
    X = np.zeros(shape=(1, X[0], X[1]))
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    
    L0 = Conv1D(filters=5, kernel_size=4, strides=2, activation="relu", padding="same")(inputs)
    
    L1 = LSTM(64, activation='tanh', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(L0)
    L2 = LSTM(bandwidth, activation='tanh', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(bandwidth, activation='tanh', return_sequences=True)(L3)
    L5 = LSTM(64, activation='tanh', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    
    print(model.summary())
    return model, None, None

def autoencoder_LSTM_big_2channel(X, bandwidth=16):
    print("X SHAPE", X[0], X[1])
    X = np.zeros(shape=(2, X[0], X[1]))
    inputs = Input(shape=(X.shape[1], X.shape[2]))

    
    L1 = LSTM(64, activation='tanh', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(bandwidth, activation='tanh', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(bandwidth, activation='tanh', return_sequences=True)(L3)
    L5 = LSTM(64, activation='tanh', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model, None, None

def autoencoder_LSTM_big_big_2channel(X, bandwidth=16):
    factor=2
    print("X SHAPE", X[0], X[1])
    X = np.zeros(shape=(2, X[0], X[1]))
    inputs = Input(shape=(X.shape[1], X.shape[2]))

    
    L1 = LSTM(64*factor, activation='tanh', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(bandwidth*factor, activation='tanh', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(bandwidth*factor, activation='tanh', return_sequences=True)(L3)
    L5 = LSTM(64*factor, activation='tanh', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model, None, None

def autoencoder_LSTM_big_2channel_CNN(X, bandwidth=16):
    print("X SHAPE", X[0], X[1])
    X = np.zeros(shape=(2, X[0], X[1]))
    inputs = Input(shape=(X.shape[1], X.shape[2]))

    L0 = Conv1D(filters=5, kernel_size=4, strides=2, activation="relu", padding="same")(inputs)
    L1 = LSTM(64, activation='tanh', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(L0)
    L2 = LSTM(bandwidth, activation='tanh', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(bandwidth, activation='tanh', return_sequences=True)(L3)
    L5 = LSTM(64, activation='tanh', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model, None, None

def autoencoder_LSTM_smaller_2channel(X, bandwidth=8):
    print("X SHAPE", X[0], X[1])
    X = np.zeros(shape=(2, X[0], X[1]))
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(16, activation='tanh', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(bandwidth, activation='tanh', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(bandwidth, activation='tanh', return_sequences=True)(L3)
    L5 = LSTM(16, activation='tanh', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model, None, None

def autoencoder_LSTM_big_2channel_batchnorm(X, bandwidth=16):
    print("X SHAPE", X[0], X[1])
    X = np.zeros(shape=(2, X[0], X[1]))
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L0 = BatchNormalization()(inputs)
    L1 = LSTM(64, activation='tanh', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(L0)
    L2 = LSTM(bandwidth, activation='tanh', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(bandwidth, activation='tanh', return_sequences=True)(L3)
    L5 = LSTM(64, activation='tanh', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model, None, None


def autoencoder_GRU(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = GRU(64, activation='tanh', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = GRU(16, activation='tanh', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = GRU(16, activation='tanh', return_sequences=True)(L3)
    L5 = GRU(64, activation='tanh', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model

def autoencoder_Conv(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = Conv1D(16, 3, activation="relu", padding="same")(inputs) # 10 dims
    #x = BatchNormalization()(x)
    L2 = MaxPooling1D(4, padding="same")(L1) # 5 dims
    L3 = Conv1D(10, 3, activation="relu", padding="same")(L2) # 5 dims
    #x = BatchNormalization()(x)
    encoded = MaxPooling1D(4, padding="same")(L3) # 3 dims
    # 3 dimensions in the encoded layer
    L4 = Conv1D(10, 3, activation="relu", padding="same")(encoded) # 3 dims
    #x = BatchNormalization()(x)
    L5 = UpSampling1D(4)(L4) # 6 dims
    L6 = Conv1D(16, 2, activation='relu')(L5) # 5 dims
    #x = BatchNormalization()(x)
    L7 = UpSampling1D(4)(L6) # 10 dims
    output = Conv1D(1, 3, activation='sigmoid', padding='same')(L7) # 10 dims
    model = Model(inputs=inputs, outputs = output)
    return model 

#edited for QUAK
def autoencoder_Conv_paper(X, bottleneck): #bottlneck feature not yet incorporated
    X = np.zeros(shape=(1, X[0]))
    inputs = Input(shape=(X.shape[1],1))
    L1 = Conv1D(256, 3, activation="relu", padding="same")(inputs) # 10 dims
    #x = BatchNormalization()(x)
    L2 = MaxPooling1D(2, padding="same")(L1) # 5 dims
    encoded = Conv1D(128, 3, activation="relu", padding="same")(L2) # 5 dims
    # 3 dimensions in the encoded layer
    L3 = UpSampling1D(2)(encoded) # 6 dims
    L4 = Conv1D(256, 3, activation='relu', padding="same")(L3)
    output = Conv1D(1, 3, activation='sigmoid', padding="same")(L4)
    model = Model(inputs=inputs, outputs = output)
    return model, None, None

def autoencoder_Conv2(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = Conv1D(16, 4, activation="relu", dilation_rate=1, padding="same")(inputs)
    L2 = MaxPooling1D(2)(L1)
    L3 = Conv1D(32, 4, activation="relu", dilation_rate=2, padding="same")(L2)
    L4 = MaxPooling1D(2)(L3) 
    L5 = Conv1D(64, 4, activation="relu", dilation_rate=2, padding="same")(L4)
    L6 = MaxPooling1D(4)(L5)
    L7 = Conv1D(128, 8, activation="relu", dilation_rate=2, padding="same")(L6)
    encoded = MaxPooling1D(4)(L7)
    L7 = Conv1D(128, 8, activation="relu", dilation_rate=2, padding="same")(encoded)
    L8 = UpSampling1D(4)(L7)
    L9 = Conv1D(64, 4, activation="relu", dilation_rate=2, padding="same")(L8)
    L10 = UpSampling1D(4)(L9)
    L11 = Conv1D(32, 4, activation="relu", dilation_rate=2, padding="same")(L10)
    L12 = UpSampling1D(4)(L11)
    L13 = Conv1D(16, 3, activation="relu", dilation_rate=1, padding="same")(L12)
    L14 = UpSampling1D(2)(L13)
    output = Conv1D(1, 4, activation="relu", dilation_rate=1, padding="same")(L12)
    model = Model(inputs=inputs, outputs = output)
    return model 
    
def autoencoder_ConvDNN(X, bottleneck):
    X = np.zeros((1, X[0], X[1]))
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = Conv1D(16, 3, activation="relu", padding="same")(inputs) # 10 dims
    #x = BatchNormalization()(x)
    L2 = MaxPooling1D(4, padding="same")(L1) # 5 dims
    L3 = Conv1D(10, 3, activation="relu", padding="same")(L2) # 5 dims
    #x = BatchNormalization()(x)
    encoded = MaxPooling1D(4, padding="same")(L3) # 3 dims
    x = Flatten()(encoded)
    x = Dense(30, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    x = Dense(30, activation='relu')(x)
    x = Dense(70, activation='relu')(x)
    x = Reshape((7, 10))(x)
    # 3 dimensions in the encoded layer
    L4 = Conv1D(10, 3, activation="relu", padding="same")(x) # 3 dims
    #x = BatchNormalization()(x)
    L5 = UpSampling1D(4)(L4) # 6 dims
    L6 = Conv1D(16, 2, activation='relu')(L5) # 5 dims
    #x = BatchNormalization()(x)
    L7 = UpSampling1D(4)(L6) # 10 dims
    output = Conv1D(1, 3, activation='sigmoid', padding='same')(L7) # 10 dims
    model = Model(inputs=inputs, outputs = output)
    return model , None, None

def autoencoder_ConvLSTM(X, bottleneck):
    X = np.zeros(shape=(1, X[0], X[1]))
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = Conv1D(16, 3, activation="relu", padding="same")(inputs) # 10 dims
    #x = BatchNormalization()(x)
    L2 = MaxPooling1D(4, padding="same")(L1) # 5 dims
    L3 = Conv1D(10, 3, activation="relu", padding="same")(L2) # 5 dims
    #x = BatchNormalization()(x)
    encoded = MaxPooling1D(4, padding="same")(L3) # 3 dims
    x = Reshape((70, 1))(encoded)
    
    x = LSTM(32, activation='relu', return_sequences=False, 
              kernel_regularizer=regularizers.l2(0.00))(x)
    x = RepeatVector(70)(x)
    x = LSTM(32, activation='relu', return_sequences=True)(x)
    out = TimeDistributed(Dense(1))(x)  
    
    x = Reshape((7, 10))(out)
    # 3 dimensions in the encoded layer
    L4 = Conv1D(10, 3, activation="relu", padding="same")(x) # 3 dims
    #x = BatchNormalization()(x)
    L5 = UpSampling1D(4)(L4) # 6 dims
    L6 = Conv1D(32, 2, activation='relu')(L5) # 5 dims
    #x = BatchNormalization()(x)
    L7 = UpSampling1D(4)(L6) # 10 dims
    output = Conv1D(1, 3, activation='sigmoid', padding='same')(L7) # 10 dims
    model = Model(inputs=inputs, outputs = output)
    return model, None, None

def autoencoder_DeepConv(X):
    #X = np.zeros(shape=X)
    ### Use autoencoder_ConvDNN instead ###
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    x = Conv1D(16, 16, activation="relu", padding="same")(inputs)
    x = MaxPooling1D(4, padding="same")(x)
    x = Conv1D(32, 8, activation="relu", padding="same",dilation_rate=4)(x)
    x = MaxPooling1D(4, padding="same")(x)
    x = Conv1D(64, 8, activation="relu", padding="same",dilation_rate=4)(x)
    x = MaxPooling1D(4, padding="same")(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = UpSampling1D(4)(x)
    x = Conv1D(64, 8, activation="relu", padding="same",dilation_rate=4)(x)
    x = UpSampling1D(4)(x)
    x = Conv1D(32, 8, activation="relu", padding="same",dilation_rate=4)(x)
    x = UpSampling1D(4)(x)
    x = Conv1D(16, 16, activation="relu", padding="same")(inputs)
    x = Dense(X.shape[1], activation='relu')(x)
    output = Reshape((X.shape[1], 1))(x)
    model = Model(inputs=inputs, outputs=output)
    return model

def autoencoder_DNN(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    x = Flatten()(inputs)
    x = Dense(int(X.shape[1]/2), activation='relu')(x)
    x = Dense(int(X.shape[1]/10), activation='relu')(x)
    x = Dense(int(X.shape[1]/2), activation='relu')(x)
    x = Dense(X.shape[1], activation='relu')(x)
    output = Reshape((X.shape[1], 1))(x)
    model = Model(inputs=inputs, outputs=output)
    return model