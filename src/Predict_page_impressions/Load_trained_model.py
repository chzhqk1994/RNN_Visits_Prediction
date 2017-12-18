import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras import optimizers
import numpy as np
import pandas as pd
import sklearn.preprocessing as prep

from keras.models import load_model


loaded_dataset = pd.read_csv('C:/Users/User/Desktop/Mamamia_Internship/RNN_Visits_Prediction/src/Data_Preprocessing/real_test_set.csv')

preprocessor = prep.StandardScaler().fit(loaded_dataset)  # 0을 기준으로 정규분포를 만드는 것 같다.
X_test = preprocessor.transform(loaded_dataset)  # 어쨌든 데이터 전처리 과정임

model = load_model('my_model.h5')


yFit = model.predict(X_test, batch_size=10, verbose=1)
print()
print(yFit)

restore = preprocessor.inverse_transform(X_test)
print(restore)
