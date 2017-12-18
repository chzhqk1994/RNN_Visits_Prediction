# http://devanix.tistory.com/306   >>  파이썬 시간 라이브러리 설명

from RNN_Visits_Prediction.src.Modularization import Connect_to_Firebase
import pandas as pd
import collections

KEY_DIRECTORY = 'C:/Users/User/Desktop/Mamamia_Internship/serviceKey.json'
DB_URL = 'https://mamamia-analytics.firebaseio.com'
DATA_page_impressions = 'facebook/Mamamia/page_impressions'

connect = Connect_to_Firebase(KEY_DIRECTORY, DB_URL)
data_page_impressions = connect.load_data_from_firebase(DATA_page_impressions)

value = []
key = []

for v, k in data_page_impressions.items():
    v2 = pd.to_datetime(v, unit='s').isoformat()
    value.append(v2)
    key.append(k)

dic = dict(zip(value, key))  # 두 리스트를 하나의 Dictionary 형으로 합침

sorted_data = collections.OrderedDict(sorted(dic.items()))  # 시간순으로 정렬
print(sorted_data)

pd.DataFrame.from_dict(data=sorted_data, orient='index').to_csv('page_impressions.csv', header=False)  # csv 파일로 변환
