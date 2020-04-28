#資料預處理
import numpy as np
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model

means = pd.Series([883, 265, 194, 462], dtype=float)
max_min = pd.Series([1802, 456, 675, 942], dtype=float)
X_train, Y_train = [], []


for data in os.listdir('s_train'):
    Data = []
    Data = pd.read_csv('s_train/'+data)
    Data = Data.values.tolist()

    for i in range(len(Data)-1):
        X_train.append((Data[i]))
        Y_train.append(np.array(Data[i+1]))

X_train = pd.DataFrame(X_train)
Y_train = pd.DataFrame(Y_train)

X_train = pd.DataFrame(X_train).apply(lambda x: (x - means) / max_min, axis=1)
Y_train = pd.DataFrame(Y_train).apply(lambda x: (x - means) / max_min, axis=1)
X_train = np.array(X_train).reshape(len(X_train),1,4)
Y_train = np.array(Y_train).reshape(len(Y_train),1,4)

X_train = X_train[int(X_train.shape[0]*0.2):]
Y_train = Y_train[int(Y_train.shape[0]*0.2):]
X_val = X_train[:int(X_train.shape[0]*0.2)]
Y_val = Y_train[:int(Y_train.shape[0]*0.2)]

model = Sequential()
model.add(LSTM(50, input_shape=(1, 4), return_sequences=True))
model.add(LSTM(100, return_sequences=True))
model.add(TimeDistributed(Dense(4)))
model.compile(loss="mse", optimizer="adam")
model.summary()
callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
history = model.fit(X_train, Y_train, epochs=15, batch_size=16, validation_data=(X_val, Y_val)
                    , callbacks=[callback])
model.save('tracking_model.h5')

plot_model(model, show_shapes='true', to_file='model.png')

loss_path = "./loss.jpg"
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["loss", "val_loss"], loc="upper left")
plt.savefig(loss_path)
plt.close(2)