# RNN_Visits_Prediction
---
## Facebook의 posts 수에 따른 page_impressions 를 예측
---


* 1일차 (2017. 11. 27)
    * 프로젝트 시작
    * 개발방향 토론
    * 데이터베이스와 파이썬 연동

* 2일차
    * DB 내용을 Dictionary 형태로 불러와서 csv 파일로 저장
    * 아직 Training data를 어떻게 만들어야 할지 모르겠어서 정렬기준은 정하지 못함, 일단 key 값으로 정렬하여 저장

* 3일차
    * Facebook 에서 제공하는 DB에 문제가 있어 page_impression을 예측하기 위해서 어떤 Neural Network를 이용해야 하는지 조사
    * 교수님 말씀으론 기훈이형이 사용했던 RNN 모델을 사용해보는것도 좋을거라고 하심
    * 근데 게시물 수에 따른 방문자수 예측인데 timestamp 가 굳이 필요할까

* 4일차
    * 어제 RNN에 대해서 검색을 해보며 공부를 해봤다.
    * 오늘은 주식 성장률을 예측하는 RNN 모델을 가져와서 page_impressisons을 예측하는 모델로 수정해 볼 것이다.

* 5일차
    * 주식 성장률을 예측하는 RNN 모델을 가져와서 코드를 분석하며 내 프로젝트에 맞춰서 수정하고 있다.
    * 아직 page_impressions과 posts사이의 관계를 알수없어 임의로 dataset을 만들어서 테스트를 하고 있다.
    * predict_page_impressions_test 폴더의 rnn.py 파일은 예측된 그래프가 변형된 상태로 나와버린다. 이거 어떻게 원래대로 늘려놓지
    * RNN_test 폴더의 LSTM_RNN.py 파일은 예측된 그래프가 변형되지 않은 모습으로 나타난다. 예측은 잘 되는것 같은데 다른 Dataset 으로 테스트 해볼 필요가 있다.
    * Paul이 얼른 제대로된 데이터를 내놔야 한다. 지금 주어진 Dataset은 넘나 이상하다. 빨리 내놔요 Paul님 현기증난다구




* 8일차
    * 현재 수정하고 있는 모델이 Tensorflow 말고 Keras를 이용하여 설계된 모델이라 Keras를 공부하고 있다. 확실히 Tensorflow 보단 간단한듯
    * 모델이 돌아가긴 해서 레이어를 중첩해보려고 하는데 LSTM input dimensions 문제가 계속 발생한다.
    * expected lstm_1_input to have 3 dimensions, but got array with shape (215, 1)  이거 정체가 뭐냐
    * 나이거 ㄹㅇ 모르겠어 따흐흙