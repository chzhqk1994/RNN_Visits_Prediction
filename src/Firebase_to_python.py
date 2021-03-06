import firebase_admin
from firebase_admin import credentials
from firebase_admin import db


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


####################################################################################

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

#################################################