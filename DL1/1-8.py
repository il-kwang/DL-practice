import sys
import warnings

import numpy as np
from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings(action="ignore")
np.random.seed(100)


def load_data(X, y):
    """1. 손글씨 데이터를 X, y로 읽어온 후
    학습 데이터, 테스트 데이터로 나눕니다.
    """
    # 학습 데이터: 앞의 1600개
    X_train = X[:1600]
    Y_train = y[:1600]

    # 테스트 데이터: 나머지
    X_test = X[1600:]
    Y_test = y[1600:]

    return X_train, Y_train, X_test, Y_test


def train_MLP_classifier(X, y):
    """2. MLPClassifier를 정의하고 hidden_layer_sizes를
    조정해 hidden layer의 크기 및 레이어의 개수를
    바꿔본 후, 학습을 시킵니다.
    """
    # MLPClassifier 정의 및 학습
    clf = MLPClassifier(hidden_layer_sizes=(70, 50),solver='adam', max_iter=1000, random_state=100)
    clf.fit(X, y)
    return clf


def report_clf_stats(clf, X, y):
    """3. 정확도를 출력하는 함수를 완성합니다."""
    hit = 0
    miss = 0

    for x, y_ in zip(X, y):
        if clf.predict([x])[0] == y_:
            hit += 1
        else:
            miss += 1

    # 정확도 계산
    score = (hit / len(y)) * 100
    print(f"Accuracy: {score:.1f}% ({hit} hit / {miss} miss)")

    return score


def main():
    """4. main 함수를 완성합니다."""
    # 손글씨 데이터 로드
    digits = load_digits()

    X = digits.data
    y = digits.target

    # 데이터 분리
    X_train, Y_train, X_test, Y_test = load_data(X, y)

    # MLPClassifier 학습
    clf = train_MLP_classifier(X_train, Y_train)

    # 테스트 데이터에 대한 정확도 출력
    report_clf_stats(clf, X_test, Y_test)

    return 0


if __name__ == "__main__":
    sys.exit(main())