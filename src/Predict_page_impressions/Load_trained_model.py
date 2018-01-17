# https://github.com/keras-team/keras/issues/2921  >>  나랑 같은 고민, train 때와 test 때의 데이터 feature 수가 다름
import numpy as np
import pandas as pd
import sklearn.preprocessing as prep

from keras.models import load_model
from keras.engine import InputLayer

loaded_dataset = pd.read_csv('C:/Users/User/Desktop/Mamamia_Internship/RNN_Visits_Prediction/src/Data_Preprocessing/Prediction_test.csv')
window = 7  # 시퀀스 길이로 넘겨진다, 타임 스탬프 비슷한건가? 이전의 7개의 값을 보고 다음 값 1개를 예측하는 방식
predict_period = 31  # 앞으로 예측할 기간을 설정

def preprocess_data(dataset, seq_len):  # loaded_data, window
    amount_of_features = len(dataset.columns)  # 피쳐 수
    data = dataset.as_matrix()
    print("len(피쳐 수 : ",amount_of_features)
    print("리스트화 된 data : \n",data)
    result = []
    
    for index in range(len(data) - seq_len):  # 데이터의 길이(행 수) 에서 시퀀스의 길이를 뺌
        result.append(data[index: index + seq_len])  # dataset을 시퀀스 길이 +1 만큼 묶음 >> 예측할때 이전의 데이터를 함께 사용

    result = np.array(result)  # result 는 시퀀스 길이로 묶인 dataset
    print('result.shape : ', result.shape)
    print('result : ', result)
    result = standard_scaler(result)  # 정규분포를 이용하여 데이터 전처리 과정을 통과시키고 반환받음
    print("result : ", result)

    new_result = np.reshape(result, (result.shape[0], result.shape[1], amount_of_features))

    return new_result


def standard_scaler(dataset):  # train, result 가 넘어옴, X_train = train, X_test = result

    dataset_samples, dataset_nx, dataset_ny = dataset.shape  # 데이터의 길이, 시퀀스 길이, Feature 수로 나눠짐

    '''
    dataset_samples : 세로로 쌓인 데이터의 수(윈도우로 묶인 데이터의 수)
    dataset_nx : 윈도우(시퀀스)의 크기
    dataset_ny : Feature 의 수
    '''

    dataset = dataset.reshape((dataset_samples, dataset_nx * dataset_ny))  # X_train.shape =(데이터 길이, 시퀀스 길이 * Feature 수)

    preprocessor = prep.StandardScaler().fit(dataset)  # 0을 기준으로 정규분포를 만드는 것 같다.
    dataset = preprocessor.transform(dataset)       # 어쨌든 데이터 전처리 과정임

    dataset = dataset.reshape((dataset_samples, dataset_nx, dataset_ny))  # 다시 Feature x 시퀀수 수 모양으로 나눔

    return dataset  # 전처리된 데이터를 반환함



print(loaded_dataset)
real_data = preprocess_data(loaded_dataset, window)  # [::-1] 을 하면 리스트가 역순으로 반환된다. 12 > 21

input_layer = InputLayer(input_shape=(None, None), name="input_1")

model = load_model('epochs_5000.learning_rate_0.01.sequence_5.RNN.4.h5')
model.layers[0] = input_layer
model.summary()



# 예측된 Y를 반복시키는 부분

Predicted_visitors=[]

for pred in range(predict_period):  # 나중에 이 숫자 7은 입력으로 받는걸로 ㄱㄱ
    print(real_data[[pred]])
    Predicted_visitors.append(model.predict(real_data[[pred]], batch_size=2, verbose=1))
    tmp = Predicted_visitors[pred]
    print(Predicted_visitors[pred])
    # print("tmp[0][pred] : ", tmp[pred][0])
    if(pred < predict_period-1):
        print("real_data[pred + 1, -1, -1] : ",real_data[pred + 1, -1, -1])
        real_data[pred + 1, -1, -1] = tmp[0][0]  # 예측된 Y를 사용하기 위해 다음날짜의 page_impressions 에 채워넣음
        print("real_data[pred + 1, -1, -1] : ",real_data[pred + 1, -1, -1])



Predicted_visitors = np.array(Predicted_visitors)
Predicted_visitors = Predicted_visitors.reshape(predict_period, 1)


import matplotlib.pyplot as plt2
plt2.plot(Predicted_visitors, color='red', label='Predicted_page_impressions', marker='o')
plt2.legend(loc='lower left')
plt2.show()

