import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# use test data dividing from training data
def train_svm_from_files(file_path, test_size=0.8, kernel='rbf', C=1, gamma='scale'):
    merged_data = pd.read_csv(file_path)

    # 특징과 라벨 분리
    X = merged_data.iloc[:, :-1]  # 특징
    y = merged_data.iloc[:, -1]  # 라벨 (activity class)

    # 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train distribution:\n{y_train.value_counts()}")

    # SVM 모델 학습
    model = SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, y_train)

    # 예측
    y_pred = model.predict(X_test)

    # 정확도 평가
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    matrix_label = [1, 2, 3, 4, 5, 6]

    # Confusion Matrix 출력 및 확률 변환
    cm = confusion_matrix(y_test, y_pred, labels=matrix_label)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Confusion Matrix 시각화 (확률)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=matrix_label, yticklabels=matrix_label)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Probabilities)')
    plt.show()

    return accuracy

# making svm_trained_model
def train_and_save_svm(file_path, model_save_path, kernel='rbf', C=1, gamma='scale'):
    merged_data = pd.read_csv(file_path)

    # 특징과 라벨 분리
    X = merged_data.iloc[:, :-1]
    y = merged_data.iloc[:, -1]

    # SVM 모델 학습
    model = SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(X, y)

    # 모델 저장
    joblib.dump(model, model_save_path)
    print(f"[INFO] 훈련된 모델이 '{model_save_path}'에 저장되었습니다.")