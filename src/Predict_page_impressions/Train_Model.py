# https://github.com/Kulbear/stock-prediction 참조함
# https://github.com/kimanalytics/Recurrent-Neural-Network-to-Predict-Stock-Prices   >> 여기도 참조해서 만들어볼것
import time
import math
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras import optimizers
import numpy as np
import pandas as pd
import sklearn.preprocessing as prep


epochs = 5000
learning_rate = 0.01
batch_size = 32
sequence = 5
lstm = 100
dataset_rate = 0.85


loaded_dataset = pd.read_csv('C:/Users/User/Desktop/Mamamia_Internship/RNN_Visits_Prediction/src/Data_Preprocessing/train_dataset(binary).csv')


def standard_scaler(x_train, x_test):  # train, result 가 넘어옴, X_train = train, X_test = result

    train_samples, train_nx, train_ny = x_train.shape  # 데이터의 길이, 시퀀스 길이(window + 1), Feature 수로 나눠짐
    test_samples, test_nx, test_ny = x_test.shape

    '''
    train_samples : 세로로 쌓인 데이터의 수(윈도우로 묶인 데이터의 수)
    train_nx : 윈도우의 크기 + 1
    train_ny : Feature 의 수
    '''

    x_train = x_train.reshape((train_samples, train_nx * train_ny))  # X_train.shape = (데이터 길이, 시퀀스 길이 * Feature 수)
    x_test = x_test.reshape((test_samples, test_nx * test_ny))

    print(x_train)

    preprocessor = prep.StandardScaler().fit(x_train)  # 0을 기준으로 정규분포를 만드는 것 같다.
    x_train = preprocessor.transform(x_train)       # 어쨌든 데이터 전처리 과정임
    x_test = preprocessor.transform(x_test)

    x_train = x_train.reshape((train_samples, train_nx, train_ny))  # 다시 Feature x 시퀀수 수 모양으로 나눔
    x_test = x_test.reshape((test_samples, test_nx, test_ny))

    return x_train, x_test  # 전처리된 데이터를 반환함


def preprocess_data(dataset, seq_len):
    amount_of_features = len(dataset.columns)  # 피쳐 수
    data = dataset.as_matrix()
    sequence_length = seq_len + 1  # 넘어온 시퀀스 길이에 + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])  # dataset을 시퀀스 길이 +1 만큼 묶음  >> 이거 왜 하는거지?

    result = np.array(result)  # result 는 시퀀스 길이로 묶인 dataset
    row = round(dataset_rate * result.shape[0])  # 여기서 곱해지는 숫자 (0 ~ 1.0 가 데이터셋을 학습, 테스트로 나누는 비율이다
    train = result[: int(row), :]
    train, result = standard_scaler(train, result)  # 정규분포를 이용하여 데이터 전처리 과정을 통과시키고 반환받음

    x_train = train[:, : -1]  # 각 집합의 마지막 행을 빼고 전부 X_train 으로 넣음 >> 마지막 행의 마지막 요소를 레이블로 나누기 위함. 존나 대단한 코드인줄 알았네
    y_train = train[:, -1][:, -1]  # 마지막 행의 마지막 열 (레이블) 을 y_train 으로 나눈다. 쉽게말해서 오른쪽 맨 아래것만 전부 모아서 레이블로 만듦
    X_test = result[int(row):, : -1]  # 마찬가지로 test 데이터를 나눔
    y_test = result[int(row):, -1][:, -1]  # test 데이터의 레이블 화

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))  # reshape 하기 전과 같은데 왜 했는지 모르겠다

    return [x_train, y_train, X_test, y_test]


def build_model(layers):  # layer[Feature 의 수(input_dim), 윈도우의 크기, ??,마지막 출력 output_dim]
    model = Sequential()

    # By setting return_sequences to True we are able to stack another LSTM layer
    model.add(LSTM(input_dim=layers[0], output_dim=layers[1], return_sequences=True))
    model.add(Dropout(0.4))

    model.add(LSTM(layers[2], return_sequences=True))
    model.add(Dropout(0.3))

    model.add(LSTM(layers[2], return_sequences=False))
    model.add(Dropout(0.3))

    model.add(Dense(units=layers[3]))  # units = output_dimensions
    model.add(Activation("linear"))

    start = time.time()
    rmsprop = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss="mse", optimizer=rmsprop, metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model


loaded_dataset.head()

window = sequence  # 시퀀스 길이로 넘겨진다, 이전의 값을 보고 다음 값 1개를 예측하는 방식
X_train, y_train, X_test, y_test = preprocess_data(loaded_dataset, window)  # [::-1] 을 하면 리스트가 역순으로 반환된다. 12 > 21

model = build_model([X_train.shape[2], window, lstm, 1])  # Feature 의 수(input_dim), 윈도우의 크기, ??,마지막 출력 output_dim 을 넘긴다

print(model.summary())  # 모델의 그래프 구조를 정리해서 보여줌

# 텐서보드와 연동
# tensorboard --logdir= 그래프 파일 경로
tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=2, callbacks=[tb_hist])

# model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=2)  # tensorboard

trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))
testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

diff = []
ratio = []
pred = model.predict(X_test)  # 들어간 값(테스트 데이터) 에 따라 예측값을 낸다.
for u in range(len(y_test)):  # 테스트 값의 라벨 만큼 반복문을 돌림
    pr = pred[u][0]
    ratio.append((y_test[u] / pr) - 1)
    diff.append(abs(y_test[u] - pr))


import matplotlib.pyplot as plt2

plt2.plot(y_test, color='blue', label='real_page_impressions')
plt2.plot(pred, color='red', label='Predicted_page_impressions')
plt2.legend(loc='lower left')
plt2.show()

model.save('my_model.h5')
del model
