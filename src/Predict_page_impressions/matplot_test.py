
import numpy as np
import pandas as pd
import sklearn.preprocessing as prep


loaded_dataset = pd.read_csv('C:/Users/User/Desktop/Mamamia_Internship/RNN_Visits_Prediction/src/Data_Preprocessing/real_dataset(less).csv')

preprocessor = prep.StandardScaler().fit(loaded_dataset)  # 0을 기준으로 정규분포를 만드는 것 같다.

print(loaded_dataset)

preprocessor = prep.StandardScaler().fit(loaded_dataset)  # 0을 기준으로 정규분포를 만드는 것 같다.
transfomred_X = preprocessor.transform(loaded_dataset)  # 어쨌든 데이터 전처리 과정임

print(transfomred_X)

x_new_inverse = preprocessor.inverse_transform(transfomred_X)

print(x_new_inverse)

print(preprocessor)