import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, \
    mean_squared_error, mean_absolute_error
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# 경고 무시 설정
warnings.filterwarnings("ignore")

# model의 성능을 평가하는 함수
def evaluate_model_performance(y_true, y_pred):
    """
    :param y_true: 실제 라벨
    :param y_pred: 예측 라벨
    """
    print("=== Model Performance Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f} (값이 클수록 좋음)")
    print(f"Precision: {precision_score(y_true, y_pred, average='weighted'):.4f} (값이 클수록 False Positive가 적음)")
    print(f"Recall: {recall_score(y_true, y_pred, average='weighted'):.4f} (값이 클수록 False Negative가 적음)")
    print(f"F1-Score: {f1_score(y_true, y_pred, average='weighted'):.4f} (정밀도와 재현율의 조화평균)")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.4f} (작을수록 좋음)")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f} (작을수록 좋음)")


# 저장된 모델로 결과를 평가하는 함수
def evaluate_saved_model(model_path, test_file_path, labels):
    model = joblib.load(model_path)
    print(f"[INFO] 모델 '{model_path}' 로드 완료.")

    # 테스트 데이터 로드
    test_data = pd.read_csv(test_file_path, header=None)
    # 데이터를 섞음 (shuffle)
    test_data_shuffled = test_data.sample(frac=1).reset_index(drop=True)
    # X_test = test_data_shuffled.iloc[:, :-1]  # 특징 데이터
    # y_test = test_data_shuffled.iloc[:, -1]  # 라벨 데이터
    X_test = test_data.iloc[:, :-1]  # 특징 데이터
    y_test = test_data.iloc[:, -1]  # 라벨 데이터
    y_pred = model.predict(X_test)

    # 정확도 평가 및 Confusion Matrix로 표현
    evaluate_model_performance(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    # cm_normalized = np.nan_to_num(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
    # sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()