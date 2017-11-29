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

