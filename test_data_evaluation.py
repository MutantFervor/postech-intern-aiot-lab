import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# 경고 무시 설정
warnings.filterwarnings("ignore")

def evaluate_saved_model(model_path, test_file_path, labels):
    """
    저장된 SVM 모델로 새로운 데이터를 검증합니다.

    Args:
    - model_path (str): 저장된 모델 파일 경로
    - test_file_path (str): 검증 데이터 파일 경로
    - labels (list): Confusion Matrix에서 사용할 클래스 라벨 리스트

    Returns:
    - None
    """
    # 모델 로드
    model = joblib.load(model_path)
    print(f"[INFO] 모델 '{model_path}' 로드 완료.")

    # 테스트 데이터 로드
    test_data = pd.read_csv(test_file_path, header=None)
    X_test = test_data.iloc[:, :-1]  # 특징 데이터
    y_test = test_data.iloc[:, -1]  # 라벨 데이터

    # 예측
    y_pred = model.predict(X_test)

    # 정확도 평가
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("\nConfusion Matrix:")
    print(cm)

    # Confusion Matrix 시각화
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification Report
    report = classification_report(y_test, y_pred, labels=labels)
    print("\nClassification Report:")
    print(report)

# 경로 설정
model_path = "./svm_trained_model.joblib"  # 저장된 모델 파일 경로
test_file_path = "./test_data/result/normalized_data.csv"   # 검증 데이터 파일 경로
labels = [1, 2, 3, 4, 5, 6]  # 클래스 라벨

# 함수 실행
evaluate_saved_model(model_path, test_file_path, labels)
