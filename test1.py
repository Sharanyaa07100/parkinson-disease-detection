import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
import os

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, InputLayer, BatchNormalization, Dropout
from keras.models import model_from_json

dataset = pd.read_csv("ProcessedData/processed_results.csv")
dataset.fillna(0, inplace = True)
dataset = dataset.values 
X = dataset[:,0:dataset.shape[1]-1]
Y = dataset[:,dataset.shape[1]-1]

scaler = MinMaxScaler() 
scaler.fit(X)
X = scaler.transform(X)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
print(X.shape)
print(Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

svm_cls = svm.SVC() 
svm_cls.fit(X_train, y_train) 
predict = svm_cls.predict(X_test)
acc = accuracy_score(predict, y_test)
print(acc)

xgb_cls = xgb.XGBClassifier() 
xgb_cls.fit(X_train, y_train) 
predict = xgb_cls.predict(X_test)
acc = accuracy_score(predict, y_test)
print(acc)


mlp_cls = MLPClassifier(max_iter=5000) 
mlp_cls.fit(X_train, y_train) 
predict = mlp_cls.predict(X_test)
acc = accuracy_score(predict, y_test)
print(acc)

X1 = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
Y1 = to_categorical(Y)
X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.2)


if os.path.exists('model/model.json'):
    with open('model/model.json', "r") as json_file:
        loaded_cnn_json = json_file.read()
        cnn = model_from_json(loaded_cnn_json)
    json_file.close()
    cnn.load_weights("model/model_weights.h5")
    cnn._make_predict_function()
else:
    cnn = Sequential()
    cnn.add(Convolution2D(128, 1, 1, input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    cnn.add(MaxPooling2D(pool_size = (1, 1)))
    cnn.add(Convolution2D(256, 1, 1, activation = 'relu'))
    cnn.add(MaxPooling2D(pool_size = (1, 1)))
    cnn.add(Flatten())
    cnn.add(Dense(output_dim = 256, activation = 'relu'))
    cnn.add(Dense(output_dim = y_train.shape[1], activation = 'softmax'))
    cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    hist = cnn.fit(X_train, y_train, batch_size=4, epochs=30, shuffle=True, verbose=2, validation_data=(X_test, y_test))
    cnn.save_weights('model/model_weights.h5')            
    model_json = cnn.to_json()
    with open("model/model.json", "w") as json_file:
        json_file.write(model_json)
    json_file.close()    

predict = cnn.predict(X_test)
y_test = np.argmax(y_test, axis=1)
predict = np.argmax(predict, axis=1)  
acc = accuracy_score(predict, y_test)
print(acc)













