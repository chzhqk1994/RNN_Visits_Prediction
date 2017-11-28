import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Fetch the service account key JSON file contents
# 인증키 경로, git 폴더 밖에 저장
cred = credentials.Certificate('C:/Users/User/Desktop/Mamamia Internship/serviceKey.json')

# Initialize the app with a service account, granting admin privileges
# 데이터베이스 주소 지정
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://mamamia-analytics.firebaseio.com'
})

# As an admin, the app has access to read and write all data, regradless of Security Rules
# 불러올 데이터의 경로 지정
ref = db.reference('facebook/Mamamia/page_impressions')
print(ref.get())