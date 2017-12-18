from RNN_Visits_Prediction.src.Modularization import Connect_to_Firebase
import pandas as pd
import numpy as np

KEY_DIRECTORY = 'C:/Users/User/Desktop/Mamamia_Internship/serviceKey.json'
DB_URL = 'https://mamamia-analytics.firebaseio.com'
DATA_page_impressions = 'facebook/Mamamia/page_impressions'

connect = Connect_to_Firebase(KEY_DIRECTORY, DB_URL)
data_page_impressions = connect.load_data_from_firebase(DATA_page_impressions)

date = []
impressions = []
week = []
for v, k in data_page_impressions.items():
    v2 = pd.to_datetime(v, unit='s').isoformat()  # 타임스탬프를 날짜로 변환
    w = pd.to_datetime(v, unit='s').isoweekday()  # 타임스탬프로부터 요일을 계산하여 코드로 변환(1~7)
    date.append(v2)
    impressions.append(k)
    week.append(w)

new = []
for i in range(len(date)):  # 날짜, 요일, DB_value 3개의 칼럼이 번갈아가며 나오게 함    날짜, 요일, value, 날짜, 요일, value...
    print(i)
    new.append(date[i])
    new.append(week[i])
    new.append(impressions[i])

new = np.reshape(new, (int(len(new)/3), 3))  # 날짜 요일 value 3개가 하나로 묶인 리스트를 만듦
sorted_list = sorted(new, key=lambda x: x[0])  # list of list 의 첫번째 요소인 날짜를 기준으로 정렬
print(sorted_list)

pd.DataFrame(sorted_list, columns=['#date', '#week', '#page_impressions']).to_csv('list.csv', header=False, index=None)
