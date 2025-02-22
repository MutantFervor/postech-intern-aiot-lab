import random
from svmutil import *
from utils import timestamp_sync as ts
from utils import feature_value_conversion as fvc
import os

def random_search_window_overlap_with_svm(raw_data_folder, processed_data_folder, log_file_path,
                                          window_values, overlap_values, kernel_values, C_values, gamma_values, n_iter=20):
    """
    Window, Overlap, SVM 하이퍼파라미터(C, Gamma, Kernel)를 주어진 값들 중 랜덤 탐색.

    :param raw_data_folder: 원본 데이터 폴더 경로
    :param processed_data_folder: 전처리된 데이터 폴더 경로
    :param log_file_path: 결과를 저장할 로그 파일 경로
    :param window_values: Window 크기 리스트
    :param overlap_values: Overlap 비율 리스트
    :param kernel_values: SVM 커널 유형 리스트
    :param C_values: SVM C 값 리스트
    :param gamma_values: SVM Gamma 값 리스트
    :param n_iter: 랜덤 탐색 반복 횟수
    """
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, 'w') as log_file:
        log_file.write("Window, Overlap, Kernel, C, Gamma, Accuracy\n")

    for j in range(1, 7):
        ts.adjust_and_sync_files(f"{raw_data_folder}/{j}", f"{processed_data_folder}/{j}")
        fvc.merge_sensor_files(f"{processed_data_folder}/{j}", f"{processed_data_folder}/{j}")

    for i in range(n_iter):
        try:
            # 주어진 값들 중 랜덤 선택
            window = random.choice(window_values)
            overlap = random.choice(overlap_values)
            kernel = random.choice(kernel_values)
            C = random.choice(C_values)
            gamma = random.choice(gamma_values)

            # 데이터 전처리
            for j in range(1, 7):
                fvc.cut_to_window(f"{processed_data_folder}/{j}", f"{processed_data_folder}/{j}",
                                  window_size=window, overlap=overlap)
            fvc.normalize_data(processed_data_folder)

            # SVM 형식 데이터 로드
            file_path = f"{processed_data_folder}/normalize_data.svm"
            y, X = svm_read_problem(file_path)

            # SVM 파라미터 설정
            param = f"-s 0 -t {kernel} -c {C:.6f} -g {gamma:.6f}"
            accuracy = svm_train(y, X, f"{param} -v 5")  # 5-fold 교차 검증

            # 결과 기록
            log_entry = f"{window}, {overlap}, {kernel}, {C:.6f}, {gamma:.6f}, {accuracy:.4f}\n"
            with open(log_file_path, 'a') as log_file:
                log_file.write(log_entry)

            print(f"[INFO] Iteration {i+1}/{n_iter}: Window={window}, Overlap={overlap}, "
                  f"Kernel={kernel}, C={C:.4f}, Gamma={gamma:.4f}, Accuracy={accuracy:.4f}")

        except Exception as e:
            print(f"[ERROR] Iteration {i+1} failed: {e}")
            continue

# 설정
raw_data_folder = "../../data/raw/train"
processed_data_folder = "../../data/processed/validation"
log_file_path = "../../data/results/random_search_window_overlap_svm_fixed.txt"
window_values = [1.0, 1.5, 2.0]  # Window 크기 리스트
overlap_values = [0.5, 0.75, 0.9]  # Overlap 비율 리스트
kernel_values = [0, 2]  # Linear, RBF 커널 사용
C_values = [0.1, 1, 10, 100]  # SVM C 값 리스트
gamma_values = [0.01, 0.1, 1, 10]  # SVM Gamma 값 리스트

# 랜덤 탐색 실행
random_search_window_overlap_with_svm(raw_data_folder, processed_data_folder, log_file_path,
                                      window_values, overlap_values, kernel_values, C_values, gamma_values, n_iter=10)
