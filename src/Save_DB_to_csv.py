from RNN_Visits_Prediction.src.Firebase_to_python import Connect_to_Firebase # 모듈화 해둔 Firebase 연결 함수를 가져옴
import collections  # dic 형을 정렬하는데에 사용
import pandas as pd  # dic 형을 csv 파일로 저장하는데에 사용

KEY_DIRECTORY = 'C:/Users/User/Desktop/Mamamia Internship/serviceKey.json'
DB_URL = 'https://mamamia-analytics.firebaseio.com'
DATA_DIR = 'facebook/Mamamia/page_impressions'


# 꺄륵 모듈화 시킨 함수를 불러옴! 매우 간단간단해짐
# 왠지는 몰라도 DB를 읽어오면 순서가 엉망징쨩이됨 뿌우 오또카징
data = Connect_to_Firebase(KEY_DIRECTORY, DB_URL, DATA_DIR)

# 그냥 sorted 함수를 사용하니 List 형으로 바뀌면서 csv 로 저장이 안됨, collection 라이브러리를 이용하여 Dictionary 를 List 형으로 바꾸지 않고 정렬
sorted_x = collections.OrderedDict(sorted(data.items()))

print(type(sorted_x), sorted_x)  # dictionary 형인것을 확인하기 위한 코드

# dictionary 형 배열을 csv 파일로 변환
pd.DataFrame.from_dict(data=sorted_x, orient='index').to_csv('dict_file.csv', header=False)
