import sys
sys.path.append(r'C:\Users\USER\PycharmProjects\activity_sensor\libsvm\python')
from svmutil import *


def train_and_save_libsvm(file_path, model_save_path, kernel=2, C=10, gamma=1):
    # (1) SVM 형식 데이터 로드
    y, X = svm_read_problem(file_path)  # .svm 형식 파일 로드

    # (2) SVM 파라미터 설정
    param = f'-s 0 -t {kernel} -c {C} -g {gamma}'  # -s: SVC 유형, -t: 커널 유형, -c: 규제, -g: gamma

    # (3) SVM 모델 학습
    model = svm_train(y, X, param)

    # (4) 학습된 모델 저장
    svm_save_model(model_save_path, model)
    print(f"[INFO] LibSVM 모델이 '{model_save_path}'에 저장되었습니다.")