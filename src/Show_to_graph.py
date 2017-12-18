import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

import matplotlib.pyplot as plt

# Fetch the service account key JSON file contents
# 인증키 경로, git 폴더 밖에 저장
cred = credentials.Certificate('C:/Users/User/Desktop/Mamamia_Internship/serviceKey.json')

# Initialize the app with a service account, granting admin privileges
# 데이터베이스 주소 지정
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://mamamia-analytics.firebaseio.com'
})

# As an admin, the app has access to read and write all data, regradless of Security Rules
# 불러올 데이터의 경로 지정
page_imp = db.reference('facebook/Mamamia/page_impressions')  # page_impressions
posts = db.reference('facebook/Mamamia/posts') # posts

data_page_imp = page_imp.get()
data_posts = posts.get()

imp_value = []
posts_value = []

for attr, value in data_page_imp.items():
    imp_value.append(value)

for attr, value in data_posts.items():
    posts_value.append(value)

print(imp_value)
print(posts_value)

plt.xlabel('data_posts')
plt.ylabel('data_page_imp')
plt.plot(posts_value[:80], imp_value[:80], "ro")
plt.show()
