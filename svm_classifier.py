import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  # 모델 저장 및 로드

def train_svm_from_files(file_paths, test_size=0.99, kernel='rbf', C=1, gamma='scale'):
    """
    여러 파일에서 데이터를 병합한 후 SVM 모델을 학습하고 평가합니다.

    Args:
    - file_paths (list of str): 데이터 파일 경로 리스트
    - test_size (float): 테스트 데이터 비율
    - kernel (str): SVM 커널 유형 (기본값: 'rbf')
    - C (float): SVM 정규화 파라미터
    - gamma (str or float): RBF 커널의 감마 값 ('scale' 또는 'auto' 가능)

    Returns:
    - accuracy (float): 테스트 데이터에 대한 정확도
    """
    # 데이터 병합
    dataframes = [pd.read_csv(file, header=None) for file in file_paths]
    merged_data = pd.concat(dataframes, axis=0, ignore_index=True)

    # 데이터를 무작위로 섞기
    merged_data = merged_data.sample(frac=1).reset_index(drop=True)

    # 로그: 섞인 데이터 확인
    # print("\n[INFO] 섞인 데이터 샘플:")
    # print(merged_data.head(10))  # 섞인 데이터 상위 10개 출력

    # 특징과 라벨 분리
    X = merged_data.iloc[:, :-1]  # 특징
    y = merged_data.iloc[:, -1]  # 라벨

    # 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train distribution:\n{y_train.value_counts()}")

    # 로그: 학습/테스트 데이터 분포 확인
    # print("\n[INFO] 학습 데이터 클래스 분포:")
    # print(y_train.value_counts())
    # print("\n[INFO] 테스트 데이터 클래스 분포:")
    # print(y_test.value_counts())

    # SVM 모델 학습
    model = SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, y_train)

    # 예측
    y_pred = model.predict(X_test)

    # 정확도 평가
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    # Confusion Matrix 출력 및 확률 변환
    cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4, 5, 6])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Confusion Matrix 시각화 (확률)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=[1, 2, 3, 4, 5, 6], yticklabels=[1, 2, 3, 4, 5, 6])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Probabilities)')
    plt.show()

    return accuracy


def train_and_save_svm(file_paths, model_save_path, test_size=0.001, kernel='rbf', C=1, gamma='scale'):
    """
    여러 파일에서 데이터를 병합한 후 SVM 모델을 학습하고, 모델을 저장합니다.

    Args:
    - file_paths (list of str): 데이터 파일 경로 리스트
    - model_save_path (str): 저장할 모델 파일 경로
    - test_size (float): 테스트 데이터 비율
    - kernel (str): SVM 커널 유형
    - C (float): SVM 정규화 파라미터
    - gamma (str or float): RBF 커널의 감마 값

    Returns:
    - accuracy (float): 테스트 데이터에 대한 정확도
    """
    # 데이터 병합
    dataframes = [pd.read_csv(file, header=None) for file in file_paths]
    merged_data = pd.concat(dataframes, axis=0, ignore_index=True)

    # 데이터를 무작위로 섞기
    merged_data = merged_data.sample(frac=1).reset_index(drop=True)

    # 특징과 라벨 분리
    X = merged_data.iloc[:, :-1]
    y = merged_data.iloc[:, -1]

    # 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)

    # SVM 모델 학습
    model = SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, y_train)

    # 모델 저장
    joblib.dump(model, model_save_path)
    print(f"[INFO] 훈련된 모델이 '{model_save_path}'에 저장되었습니다.")

    # 예측
    y_pred = model.predict(X_test)

    # 정확도 평가
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    # Confusion Matrix 출력 및 시각화
    cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4, 5, 6])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1, 2, 3, 4, 5, 6], yticklabels=[1, 2, 3, 4, 5, 6])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    return accuracy

# 데이터 파일 경로 리스트
file_paths = [
    './testing/1/result/normalized_data.csv',
    './testing/2/result/normalized_data.csv',
    './testing/3/result/normalized_data.csv',
    './testing/4/result/normalized_data.csv',
    './testing/5/result/normalized_data.csv',
    './testing/6/result/normalized_data.csv'
]

# 함수 실행
# train_svm_from_files(file_paths)

model_save_path = './svm_trained_model.joblib'

# 함수 실행
train_and_save_svm(file_paths, model_save_path)