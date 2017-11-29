import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import collections
import pandas as pd


class Connect_to_Firebase:
    def __init__(self, service_key, db_address):

        # Fetch the service account key JSON file contents
        # 인증키 경로, git 폴더 밖에 저장
        self.cred = credentials.Certificate(service_key)

        # Initialize the app with a service account, granting admin privileges
        # 데이터베이스 주소 지정
        firebase_admin.initialize_app(self.cred, {
            'databaseURL': db_address
        })

    def load_data_from_firebase(self, data_dir):

        # As an admin, the app has access to read and write all data, regradless of Security Rules
        # 불러올 데이터의 경로 지정
        ref = db.reference(data_dir)
        data = ref.get()

        return data

    # 정렬 기능은 빼던가 key, value 두 기준으로 정렬할 수 있도록 업데이트 해야한다.
    def sort_and_save_to_csv(self, data, filename):

        # 그냥 sorted 함수를 사용하니 List 형으로 바뀌면서 csv 로 저장이 안됨, collection 라이브러리를 이용하여 Dictionary 를 List 형으로 바꾸지 않고 정렬
        sorted_x = collections.OrderedDict(sorted(data.items()))

        print(type(sorted_x), sorted_x)  # dictionary 형인것을 확인하기 위한 코드

        # dictionary 형 배열을 csv 파일로 변환
        pd.DataFrame.from_dict(data=sorted_x, orient='index').to_csv(filename, header=False)

#########################################################################################

# load_data_from_firebase 사용법

# from RNN_Visits_Prediction.src.Firebase_to_python import Connect_to_Firebase
#
# KEY_DIRECTORY = 'C:/Users/User/Desktop/Mamamia Internship/serviceKey.json'
# DB_URL = 'https://mamamia-analytics.firebaseio.com'
#
# DATA_page_impressions = 'facebook/Mamamia/page_impressions'
# DATA_posts = 'facebook/Mamamia/posts'
#
#
# connect = Connect_to_Firebase(KEY_DIRECTORY, DB_URL)
#
# data_page_impressions = connect.load_data_from_firebase(DATA_page_impressions)
# data_posts = connect.load_data_from_firebase(DATA_posts)
#
# print(data_page_impressions)
# print(data_posts)

#######################################################################################

# sort_and_save_to_csv 사용법

# 꺄륵 모듈화 시킨 함수를 불러옴! 매우 간단간단해짐
# 왠지는 몰라도 DB를 읽어오면 순서가 엉망징쨩이됨 뿌우 오또카징
# connect = Connect_to_Firebase(KEY_DIRECTORY, DB_URL)
# data = connect.load_data_from_firebase(DATA_DIR)

# 저장할 데이터인 data 와 파일명 'save.csv' 를 넘김
# connect.sort_and_save_to_csv(data, 'save.csv')
