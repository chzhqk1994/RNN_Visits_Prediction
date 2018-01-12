# https://github.com/keras-team/keras/issues/2921  >>  나랑 같은 고민, train 때와 test 때의 데이터 feature 수가 다름
import numpy as np
import pandas as pd
import sklearn.preprocessing as prep

from keras.models import load_model
from keras.engine import InputLayer


loaded_dataset = pd.read_csv('C:/Users/User/Desktop/Mamamia_Internship/RNN_Visits_Prediction/src/Data_Preprocessing/train_dataset(binary).csv')
# loaded_dataset = pd.read_csv('C:/Users/User/Desktop/Mamamia_Internship/RNN_Visits_Prediction/src/Predict_page_impressions/predict.csv')
window = 2  # 시퀀스 길이로 넘겨진다, 타임 스탬프 비슷한건가? 이전의 5개의 값을 보고 다음 값 1개를 예측하는 방식


def standard_scaler(dataset):  # train, result 가 넘어옴, X_train = train, X_test = result

    dataset_samples, dataset_nx, dataset_ny = dataset.shape  # 데이터의 길이, 시퀀스 길이(window + 1), Feature 수로 나눠짐

    '''
    dataset_samples : 세로로 쌓인 데이터의 수(윈도우로 묶인 데이터의 수)
    dataset_nx : 윈도우의 크기 + 1
    dataset_ny : Feature 의 수
    '''

    dataset = dataset.reshape((dataset_samples, dataset_nx * dataset_ny))  # X_train.shape = (데이터 길이, 시퀀스 길이 * Feature 수)

    print('Reshaped X_train.shapeㅌ : ', dataset.shape)
    print(dataset)

    preprocessor = prep.StandardScaler().fit(dataset)  # 0을 기준으로 정규분포를 만드는 것 같다.
    dataset = preprocessor.transform(dataset)       # 어쨌든 데이터 전처리 과정임

    print('StandardScaler().fit() X_train.shape : ', dataset.shape)
    print(dataset)

    dataset = dataset.reshape((dataset_samples, dataset_nx, dataset_ny))  # 다시 Feature x 시퀀수 수 모양으로 나눔

    print('Reshaped processed X_train.shape : ', dataset.shape)
    print(dataset)

    return dataset  # 전처리된 데이터를 반환함



def preprocess_data(dataset, seq_len):
    amount_of_features = len(dataset.columns)  # 피쳐 수
    data = dataset.as_matrix()
    print("len",amount_of_features)
    print("data",data)
    sequence_length = seq_len + 1  # 넘어온 시퀀스 길이에 + 1
    result = []

    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])  # dataset을 시퀀스 길이 +1 만큼 묶음  >> 이거 왜 하는거지?

    result = np.array(result)  # result 는 시퀀스 길이로 묶인 dataset
    print('hgihi', result.shape)
    result = standard_scaler(result)  # 정규분포를 이용하여 데이터 전처리 과정을 통과시키고 반환받음

    new_result = np.reshape(result, (result.shape[0], result.shape[1], amount_of_features))

    return new_result


real_data = preprocess_data(loaded_dataset, window)  # [::-1] 을 하면 리스트가 역순으로 반환된다. 12 > 21

print(real_data)

input_layer = InputLayer(input_shape=(None, None), name="input_1")

model = load_model('epochs_10000.learning_rate_0.01.sequence_5.RNN.4.h5')
model.layers[0] = input_layer
model.summary()

yFit = model.predict(real_data, batch_size=2, verbose=1)
#
# print()
print(yFit)

import matplotlib.pyplot as plt2

plt2.plot(yFit, color='red', label='Predicted_page_impressions')
plt2.legend(loc='lower left')
plt2.show()