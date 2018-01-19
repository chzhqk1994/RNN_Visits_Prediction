import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Reading CSV file into training set
training_set = pd.read_csv('C:/Users/User/Desktop/Mamamia_Internship/RNN_Visits_Prediction/src/Data_Preprocessing/train_dataset(binary).csv')
training_set = training_set.iloc[:, 0:9]  # iloc 은 pandas 라이브러리의 함수이며, 지정한 범위까지의 데이터를 불러온다. iloc[:,] 여기서 :, 는 왜 있는지 모르겠다
print(training_set.head())

training_set = training_set.values  # csv 파일의 값만 불러와서 튜플로 저장함
print(training_set)

import sklearn.preprocessing as prep

sc = prep.StandardScaler().fit(training_set)  # 0을 기준으로 정규분포를 만드는 것 같다.
training_set = sc.transform(training_set)       # 어쨌든 데이터 전처리 과정임



print(training_set)

X_train = training_set[0:len(training_set)-1]
Y_train = training_set[1:, -1]  # 각 행의 마지막 요소(page_impressions) 를 Y_train 으로 넣음 맨앞의 1은 두번째 행부터 넣음을 의미 (다음 값을 라벨로 하기 위함)

print(Y_train)

today = pd.DataFrame(X_train[0:])   # 여기 코드는 걍 이해를 위해 보여주는 용. X 와 Y 를 비교해보자 특히 각각의 page_impressions 부분을 보면 한칸씩 밀어서 라벨링함
tomorrow = pd.DataFrame(Y_train[0:])
ex = pd.concat([today, tomorrow], axis=1)
ex.colums = (['today', 'tomorrow'])

print(ex)

X_train = np.reshape(X_train, (len(training_set)-1, 9, 1))

print(X_train)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# initializing the Recurrent Neural Network
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units=40, activation='sigmoid', input_shape=(None, 1)))

# Adding the output layer
regressor.add(Dense(units=1))

# Compiling teh Recurrent Neural Network
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the Recurrent Neural Network to the Training set
regressor.fit(X_train, Y_train, batch_size=32, epochs=2, verbose=2)


# Getting the real page_impressions
test_set = pd.read_csv('C:/Users/User/Desktop/Mamamia_Internship/RNN_Visits_Prediction/src/Data_Preprocessing/train_dataset(binary).csv')
real_page_impressions = test_set.iloc[:, 0:9]  # iloc 은 pandas 라이브러리의 함수이며, 지정한 범위까지의 데이터를 불러온다. iloc[:,] 여기서 :, 는 왜 있는지 모르겠다

# Converting to 2D array
real_page_impressions = real_page_impressions.values

# Getting the predicted page_impressions
inputs = real_page_impressions
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (len(inputs), 9, 1))

predicted_page_impressions = regressor.predict(inputs)
# predicted_page_impressions = sc.inverse_transform(predicted_page_impressions)

print(predicted_page_impressions)
print(predicted_page_impressions.shape)
print(type(predicted_page_impressions))
plt.plot(real_page_impressions, color='red', label="Real Page Impressions")
plt.plot(predicted_page_impressions, color='blue', label="Predicted Page Impressions")
plt.show()


