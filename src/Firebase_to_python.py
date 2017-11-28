import firebase_admin
from firebase_admin import credentials
from firebase_admin import db


def Connect_to_Firebase(SERVICE_KEY, DB_ADDRESS, DATA_DIR):
    # Fetch the service account key JSON file contents
    # 인증키 경로, git 폴더 밖에 저장
    cred = credentials.Certificate(SERVICE_KEY)

    # Initialize the app with a service account, granting admin privileges
    # 데이터베이스 주소 지정
    firebase_admin.initialize_app(cred, {
        'databaseURL': DB_ADDRESS
    })

    # As an admin, the app has access to read and write all data, regradless of Security Rules
    # 불러올 데이터의 경로 지정
    ref = db.reference(DATA_DIR)
    data = ref.get()

    return data


key_dir = 'C:/Users/User/Desktop/Mamamia Internship/serviceKey.json'
db_url = 'https://mamamia-analytics.firebaseio.com'
data_dir = 'facebook/Mamamia/page_impressions'


collected_data = Connect_to_Firebase(key_dir, db_url, data_dir)

print(collected_data)
