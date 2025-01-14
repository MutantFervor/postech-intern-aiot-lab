from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def train_svm_from_files(file_paths, test_size=0.3, kernel='rbf', C=1, gamma='scale'):
    # 데이터 병합
    dataframes = [pd.read_csv(file, header=None) for file in file_paths]
    merged_data = pd.concat(dataframes, axis=0, ignore_index=True)

    # 데이터를 무작위로 섞기
    merged_data = merged_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # 특징과 레이블 분리
    X = merged_data.iloc[:, :-1]  # 특징
    y = merged_data.iloc[:, -1]  # 라벨

    # 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # SVM 모델 학습
    model = SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(X_train, y_train)

    # 예측
    y_pred = model.predict(X_test)

    # 정확도 평가
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Confusion Matrix 시각화
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification Report
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)

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
train_svm_from_files(file_paths)

