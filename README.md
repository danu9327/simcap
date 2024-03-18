# 2024.03.18 심화캡스톤 다중 객체 추적 실시간입니다!😿

yolov9와 bytetrack을 이용하여 실시간 다중 객체 추적을 했습니다.

밑에 링크는 열심히 탐구탐구한 욜로9 깃허브입니다.
https://github.com/WongKinYiu/yolov9

pt파일을 직접 만들었고 실시간 추적 코드를 짰습니다.
밑에는 실행사진 및 영상입니다

# 결과 사진
![ 다중객체 추적 실시간 실행사진 ](assets/1.png)
기존 블로그 예제 코드에서 document[0]으로 PDF파일의 1페이지만 불러오던 것을 
수정해 전체 PDF파일을 불러오게 바꿨습니다.

![ 실행영상 ](assets/2.gif)
bert-base-multilingual-uncased모델을 HuggingFaceEmbeddings을 이용해 임베딩에 사용했다.

## 실행 방법
1. 아나콘다 가상환경 생성(파이썬 버전 >=3.8)
2. git clone repo
3. 리콰이어먼트.txt 설치
4. cpu,gpu 사용할 device를 333.py에서 설정
5. 터미널에서 333.py 실행
6. 짜잔
