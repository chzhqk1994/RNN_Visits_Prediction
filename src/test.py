from RNN_Visits_Prediction.src.Modularization import Connect_to_Firebase # 모듈화 해둔 Firebase 연결 함수를 가져옴
import collections  # dic 형을 정렬하는데에 사용
import pandas as pd  # dic 형을 csv 파일로 저장하는데에 사용

KEY_DIRECTORY = 'C:/Users/User/Desktop/Mamamia Internship/serviceKey.json'
DB_URL = 'https://mamamia-analytics.firebaseio.com'

DATA_DIR = 'facebook/Mamamia/posts'


# 꺄륵 모듈화 시킨 함수를 불러옴! 매우 간단간단해짐
# 왠지는 몰라도 DB를 읽어오면 순서가 엉망징쨩이됨 뿌우 오또카징
connect = Connect_to_Firebase(KEY_DIRECTORY, DB_URL)
data = connect.load_data_from_firebase(DATA_DIR)

# 저장할 데이터인 data 와 파일명 'save.csv' 를 넘김
connect.sort_and_save_to_csv(data, 'save.csv')
