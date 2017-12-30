# https://github.com/Kulbear/stock-prediction 참조함

# https://github.com/clintonreece/keras-cloud-ml-engine  >> 구글 클라우드 튜토리얼

from __future__ import print_function
import argparse
import h5py  # for saving the model
from datetime import datetime  # for filename conventions
from tensorflow.python.lib.io import file_io
import sys

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


epochs = 10000
learning_rate = 0.005
batch_size = 768
sequence = 20
lstm = 100
dataset_rate = 0.85


def standard_scaler(X_train, X_test):  # train, result 가 넘어옴, X_train = train, X_test = result
    train_samples, train_nx, train_ny = X_train.shape  # 데이터의 길이, 시퀀스 길이(window + 1), Feature 수로 나눠짐
    test_samples, test_nx, test_ny = X_test.shape

    '''
    train_samples : 세로로 쌓인 데이터의 수(윈도우로 묶인 데이터의 수)
    train_nx : 윈도우의 크기 + 1
    train_ny : Feature 의 수
    '''

    X_train = X_train.reshape((train_samples, train_nx * train_ny))  # X_train.shape = (데이터 길이, 시퀀스 길이 * Feature 수)
    X_test = X_test.reshape((test_samples, test_nx * test_ny))

    preprocessor = prep.StandardScaler().fit(X_train)  # 0을 기준으로 정규분포를 만드는 것 같다.
    X_train = preprocessor.transform(X_train)       # 어쨌든 데이터 전처리 과정임
    X_test = preprocessor.transform(X_test)

    X_train = X_train.reshape((train_samples, train_nx, train_ny))  # 다시 Feature x 시퀀수 수 모양으로 나눔
    X_test = X_test.reshape((test_samples, test_nx, test_ny))

    return X_train, X_test  # 전처리된 데이터를 반환함



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


    print('트레인 : ', train)
    X_train = train[:, : -1]  # 각 집합의 마지막 행을 빼고 전부 X_train 으로 넣음 >> 마지막 행의 마지막 요소를 레이블로 나누기 위함. 존나 대단한 코드인줄 알았네

    y_train = train[:, -1][:, -1]  # 마지막 행의 마지막 열 (레이블) 을 y_train 으로 나눈다. 쉽게말해서 오른쪽 맨 아래것만 전부 모아서 레이블로 만듦

    X_test = result[int(row):, : -1]  # 마찬가지로 test 데이터를 나눔

    y_test = result[int(row):, -1][:, -1]  # test 데이터의 레이블 화

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))  # reshape 하기 전과 같은데 왜 했는지 모르겠다

    return [X_train, y_train, X_test, y_test]


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


# create function to allow for different training data and other options
def train_model(train_file='data/data-02-stock_daily.csv', job_dir='./tmp/rnn_model', **args):
    # set the logging path for ML Engine logging to Storage bucket
    logs_path = job_dir + '/logs/' + datetime.now().isoformat()
    print('Using logs_path located at {}'.format(logs_path))

    loaded_dataset = pd.read_csv('C:/Users/User/Desktop/Mamamia_Internship/RNN_Visits_Prediction/src/gcptest/data/data-02-stock_daily.csv')

    window = sequence  # 시퀀스 길이로 넘겨진다, 타임 스탬프 비슷한건가? 이전의 5개의 값을 보고 다음 값 1개를 예측하는 방식
    X_train, y_train, X_test, y_test = preprocess_data(loaded_dataset[:: -1], window)  # [::-1] 을 하면 리스트가 역순으로 반환된다. 12 > 21


    model = build_model([X_train.shape[2], window, lstm, 1])  # Feature 의 수(input_dim), 윈도우의 크기, ??,마지막 출력 output_dim 을 넘긴다

    print(model.summary())  # 모델의 그래프 구조를 정리해서 보여줌

    # 텐서보드와 연동
    # tensorboard --logdir= 그래프 파일 경로
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=2)


    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

    testScore = model.evaluate(X_test, y_test, verbose=0)
    print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))



    model.save('my_model.h5')

    with file_io.FileIO('my_model.h5', mode='r') as input_f:
        with file_io.FileIO(job_dir + '/model.h5', mode='w+') as output_f:
            output_f.write(input_f.read())


if __name__ == '__main__':
    # Parse the input arguments for common Cloud ML Engine options
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train-file',
        help = 'Cloud Storage bucket or local path to training data')
    parser.add_argument(
        '--job-dir',
        help = 'Cloud storage bucket to export the model and store temp files')
    args = parser.parse_args()
    arguments = args.__dict__
    train_model(**arguments)
