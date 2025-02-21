import pandas as pd

def save_as_svm_format(data, save_path):
    # 데이터가 경로인 경우 파일 로드
    if isinstance(data, str):
        data = pd.read_csv(data, header=None)

    labels = data.iloc[:, -1]
    features = data.iloc[:, :-1]

    # SVM 포맷으로 변환
    svm_format_lines = []
    for i in range(len(data)):
        label = labels.iloc[i]
        feature_values = features.iloc[i]
        svm_line = f"{label} " + " ".join(
            [f"{idx + 1}:{value}" for idx, value in enumerate(feature_values) if value != 0]
        )
        svm_format_lines.append(svm_line)

    # SVM 형식으로 파일 저장
    with open(save_path, 'w') as f:
        for line in svm_format_lines:
            f.write(line + "\n")

    print(f"SVM 형식 데이터가 '{save_path}'에 저장되었습니다.")
    return