import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


def create_dataset(dataset, look_back):  # dataset 은 우리가 데이터셋으로 만들고 싶은 Numpy 배열, look_back 은 다음기간을 예측하기위한 입력 변수
    dataX, dataY = [], []  # X는 주어진 시간(t)의 승객 수이고, Y는 다음시간(t + 1)에 해당하는  이거 뭔소린지 모르겠네요오오오옹
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


dataframe = read_csv('test.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')  # 정수값을 신경망을 사용하여 모델링하는데 적합한 부동소수점으로 변환


# scikit-learn 라이브러리의 MinMaxScaler 전처리 클래스를 이용하여 데이터셋을 정규화, 0과 1사이로 설정
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# 데이터세트를 학습, 테스트 데이터로 나눔, 아래 코드는 학습데이터 67%, 나머지는 테스트 데이터로 나눔
train_size = int(len(dataset) * 0.67)  # 데이터셋에서 자를 부분을 계산함
test_size = len(dataset) - train_size
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]


# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
model = Sequential()  
model.add(LSTM(300, input_shape=(1,), return_sequences=True))
model.add(LSTM(300, return_sequences=True))
model.add(Dropout(0.2))  
model.add(LSTM(300, return_sequences=False))
model.add(Dropout(0.2))  
model.add(Dense(200, 1))
model.add(Activation("linear"))  
model.compile(loss="mean_squared_error", optimizer="rmsprop")  