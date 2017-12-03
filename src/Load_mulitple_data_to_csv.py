from RNN_Visits_Prediction.src.Firebase_to_python import Connect_to_Firebase

KEY_DIRECTORY = 'C:/Users/User/Desktop/Mamamia Internship/serviceKey.json'
DB_URL = 'https://mamamia-analytics.firebaseio.com'

DATA_page_impressions = 'facebook/Mamamia/page_impressions'
DATA_posts = 'facebook/Mamamia/posts'


connect = Connect_to_Firebase(KEY_DIRECTORY, DB_URL)

data_page_impressions = connect.load_data_from_firebase(DATA_page_impressions)
data_posts = connect.load_data_from_firebase(DATA_posts)

print(data_page_impressions)
print(data_posts)
