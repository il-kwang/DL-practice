import numpy as np

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

np.random.seed(64)

'''
1. iris 데이터를 불러오고, 
   불러온 데이터를 학습용, 테스트용 데이터로 
   분리하여 반환하는 함수를 구현합니다.
   
   Step01. 불러온 데이터를 학습용 데이터 80%, 
           테스트용 데이터 20%로 분리합니다.
           
           일관된 결과 확인을 위해 random_state를 
           0으로 설정합니다.        
'''

def load_data():
    iris = load_iris()
    
    # 꽃잎의 길이와 너비만 사용
    X = iris.data[:, 2:4]
    Y = iris.target
    
    # 데이터를 학습용 80%, 테스트용 20%로 분리
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
    
    return X_train, X_test, Y_train, Y_test

'''
2. 사이킷런의 Perceptron 클래스를 사용하여 
   Perceptron 모델을 정의하고,
   학습용 데이터에 대해 학습시킵니다.
   
   Step01. 앞서 완성한 함수를 통해 데이터를 불러옵니다.
   
   Step02. Perceptron 모델을 정의합니다.
           max_iter와 eta0를 자유롭게 설정해보세요.
   
   Step03. 학습용 데이터에 대해 모델을 학습시킵니다.
   
   Step04. 테스트 데이터에 대한 모델 예측을 수행합니다. 
'''

def main():   
    # 데이터 로드
    X_train, X_test, Y_train, Y_test = load_data()
    
    # Perceptron 모델 정의
    perceptron = Perceptron(max_iter=2000, eta0=0.1, random_state=64)
    
    # 모델 학습
    perceptron.fit(X_train, Y_train)
    
    # 테스트 데이터에 대한 예측
    pred = perceptron.predict(X_test)
    
    # 정확도 계산
    accuracy = accuracy_score(pred, Y_test)
    
    print("Test 데이터에 대한 정확도 : %0.5f" % accuracy)
    
    return X_train, X_test, Y_train, Y_test, pred

if __name__ == "__main__":
    main()