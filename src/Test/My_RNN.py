# https://tykimos.github.io/2017/09/09/Time-series_Numerical_Input_Numerical_Prediction_Model_Recipe/
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 예측 모델이기 때문에 X(데이터)는 이전수치가 되고 Y(라벨)은 다음 데이터가 된다!!!
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        dataX.append(dataset[i:(i+look_back), 0])
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


loaded_data = pd.read_csv("test.csv")
scaler = MinMaxScaler(feature_range=(0, 1))
preprocessed_data = scaler.fit_transform(loaded_data)


# 데이터세트를 학습, 테스트 데이터로 나눔, 아래 코드는 학습데이터 67%, 나머지는 테스트 데이터로 나눔
train_size = int(len(preprocessed_data) * 0.7)  # 데이터셋에서 자를 부분을 계산함, 70%는 트레이닝 데이터
test_size = len(preprocessed_data) - train_size  # 나머지 30%는 테스트 데이터로 나눔
train, test = preprocessed_data[0:train_size], preprocessed_data[train_size:len(preprocessed_data)]  # 각 배열을 만들어서 나눠서 넣음


trainX, trainY = create_dataset(train, look_back=1)
testX, testY = create_dataset(test, look_back=1)

'''
model = Sequential()
model.add(LSTM(units=32, activation='relu', input_dim=1))
model.add(Dropout(0.3))
for i in range(2):
    model.add(LSTM(32, activation="relu"))
    model.add(Dropout(0.3))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=1000, batch_size=32)
'''
'''
model = Sequential()
model.add(Embedding(1, output_dim=1))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(trainX, trainY, epochs=1000, batch_size=32)
'''
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(1, 1)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(1, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
model.fit(trainX, trainY, epochs=1000, batch_size=32)