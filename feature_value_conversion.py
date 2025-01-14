import os

import pandas as pd
import numpy as np

# 기존 file을 timestamp 기준으로 하나로 합병
def load_file(input_folder, output_folder):
    linear_file = f"{input_folder}/result/linear.csv"
    gyro_file = f"{input_folder}/result/gyro.csv"
    gravity_file = f"{input_folder}/result/gravity.csv"

    # 1. 개별 파일 로드
    accel = pd.read_csv(linear_file, header=None)
    gyro = pd.read_csv(gyro_file, header=None)
    gravity = pd.read_csv(gravity_file, header=None)

    # col에 이름 부여
    accel.columns = ['activity_class', 'timestamp', 'x', 'y', 'z']
    gyro.columns = ['activity_class', 'timestamp', 'x', 'y', 'z']
    gravity.columns = ['activity_class', 'timestamp', 'x', 'y', 'z']

    # 비교용 타임스탬프 생성 (소수점 두 번째 자리까지 비교, 내림 사용)
    def create_comparable_timestamp(df):
        df['timestamp_compare'] = np.floor(df['timestamp'] / 1_000_000 * 100) / 100
        return df

    accel = create_comparable_timestamp(accel)
    gyro = create_comparable_timestamp(gyro)
    gravity = create_comparable_timestamp(gravity)

    # activity_class 삭제하여 하나로 묶기
    gyro = gyro.drop(columns=['activity_class', 'timestamp'])
    gravity = gravity.drop(columns=['activity_class', 'timestamp'])

    # 2. 타임스탬프를 기준으로 병합
    merged_data = pd.merge(accel, gyro, on='timestamp_compare', how='inner', suffixes=('_accel', '_gyro'))
    merged_data = pd.merge(merged_data, gravity, on='timestamp_compare', how='inner', suffixes=('', '_gravity'))
    merged_data = merged_data.drop(columns=['timestamp_compare'])

    # 3. 병합 결과 확인
    # print(f"병합된 데이터 크기: {merged_data.shape}")
    print(merged_data.head())

    # 4. 새로운 CSV 파일로 저장
    merged_data.to_csv(f"{output_folder}/result/merged_data.csv", header=False, index=False)
    print("병합된 데이터를 'merged_data.csv'로 저장했습니다.")
    return


# timestamp 기준으로 1초로 window 자르기 (0.5s씩 겹쳐서)
def cut_to_window(input_folder, window_size=1.0, overlap=0.5):
    # 파일 경로 설정
    data_file = f"{input_folder}/result/merged_data.csv"
    save_path = f"{input_folder}/result/feature.csv"

    # 데이터 로드
    data = pd.read_csv(data_file, header=None)

    # 열 이름 설정
    data.columns = ['activity_class', 'timestamp', 'accel_x', 'accel_y', 'accel_z',
                    'gyro_x', 'gyro_y', 'gyro_z', 'gravity_x', 'gravity_y', 'gravity_z']

    # 슬라이딩 윈도우 설정
    start_timestamp = data.iloc[0, 1]
    end_timestamp = data.iloc[-1, 1]
    window_step = overlap * 1_000_000  # 오버랩 간격 (마이크로초 단위)
    window_size_in_microseconds = window_size * 1_000_000  # 윈도우 크기 (마이크로초 단위)

    # 결과 저장용 데이터프레임
    feature_list = []

    # 슬라이딩 윈도우 반복
    while start_timestamp + window_size_in_microseconds <= end_timestamp:
        current_window_end = start_timestamp + window_size_in_microseconds

        # 특징 벡터 계산
        features = []
        for sensor in ['accel', 'gyro', 'gravity']:
            # x, y, z 축별로 데이터 슬라이싱
            window_x = cut_data(data, sensor, 'x', start_timestamp, current_window_end)
            window_y = cut_data(data, sensor, 'y', start_timestamp, current_window_end)
            window_z = cut_data(data, sensor, 'z', start_timestamp, current_window_end)

            # 독립적인 축 데이터를 특징 계산 함수에 전달
            features.extend(calculate_features(window_x, window_y, window_z))

        # 레이블 추가 (activity_class)
        label = data[(data['timestamp'] >= start_timestamp) &
                     (data['timestamp'] <= current_window_end)]['activity_class'].iloc[0]
        features.append(label)  # 레이블 추가

        # 결과 저장 (feature list에 추가)
        feature_list.append(features)

        # 타임스탬프 업데이트
        start_timestamp += window_step

    # 결과를 CSV로 저장
    feature_df = pd.DataFrame(feature_list)
    feature_df.to_csv(save_path, index=False, header=False)
    print(f"Feature extraction complete. Saved to {save_path}.")
    return


# timestamp에 따라 data cutting
def cut_data(data, sensor, axis, startTimestamp, stopTimestamp):
    filtered_data = data[(data['timestamp'] >= startTimestamp) & (data['timestamp'] <= stopTimestamp)]
    sensor_start_num = {'accel': 2, 'gyro': 5, 'gravity': 8}
    axis_start_num = {'x': 0, 'y': 1, 'z': 2}

    column_index = sensor_start_num[sensor] + axis_start_num[axis]
    window = filtered_data.iloc[:, column_index].values
    return window


# feature 계산 함수
def calculate_features(window_x, window_y, window_z):
    # 1. DC (평균값)
    dc_x, dc_y, dc_z = np.mean(window_x), np.mean(window_y), np.mean(window_z)

    # 2. Information Entropy
    def calculate_entropy(window):
        fft_result = np.fft.fft(window)
        magnitude = np.abs(fft_result)  # 복소수의 크기 |X[k]|
        # magnitude[0] = 0
        prob_distribution = magnitude ** 2 / np.sum(magnitude ** 2)  # 에너지 기반 확률 분포
        entropy_num = -np.sum(prob_distribution * np.log10(prob_distribution))
        return entropy_num

    h_x = calculate_entropy(window_x)
    h_y = calculate_entropy(window_y)
    h_z = calculate_entropy(window_z)

    # 3. Total Energy of Frequency Spectrum
    def calculate_energy(window):
        fft_result = np.fft.fft(window)
        magnitude = np.abs(fft_result)  # 복소수의 크기 |X[k]|
        # magnitude[0] = 0
        energy_num = np.sum(magnitude ** 2) / len(window)
        return energy_num

    e_x = calculate_energy(window_x)
    e_y = calculate_energy(window_y)
    e_z = calculate_energy(window_z)

    # 4. Correlation
    r_xy = np.corrcoef(window_x, window_y)[0, 1]
    r_yz = np.corrcoef(window_y, window_z)[0, 1]
    r_xz = np.corrcoef(window_x, window_z)[0, 1]

    # 12개
    return [dc_x, h_x, e_x, r_xy, r_xz,  # x축
            dc_y, h_y, e_y, r_yz,        # y축
            dc_z, h_z, e_z]              # z축


# feaute data를 nomalization : Min-Max Normalization
def normalization(input_folder):
    # 파일 경로 설정
    data_file = f"{input_folder}/result/feature.csv"
    save_path = f"{input_folder}/result/normalized_data.csv"
    params_path = f"{input_folder}/result/normalization_params.csv"

    # 데이터 로드
    data = pd.read_csv(data_file, header=None)  # 헤더 없이 데이터 로드
    columns_to_normalize = list(range(data.shape[1]))

    # 정규화 파라미터 저장용 리스트
    normalization_params = []

    # 정규화 수행
    for column in columns_to_normalize:
        min_val = data[column].min()
        max_val = data[column].max()

        # a_i와 b_i 계산
        a_i = 1 / (max_val - min_val) if max_val != min_val else 1
        b_i = -min_val / (max_val - min_val) if max_val != min_val else 0

        # 데이터 normalization
        data[column] = a_i * data[column] + b_i

        # 파라미터 저장
        normalization_params.append({
            "column_index": column,
            "a_i": a_i,
            "b_i": b_i
        })

    # 정규화된 데이터 저장
    data.to_csv(save_path, index=False, header=False)
    print(f"정규화된 데이터를 '{save_path}'로 저장했습니다.")

    # 파라미터 저장 (CSV 파일)
    pd.DataFrame(normalization_params).to_csv(params_path, index=False)
    print(f"정규화 파라미터를 '{params_path}'로 저장했습니다.")


# normailzation 값을 libsvm이 읽을 수 있는 형식으로 바꿈 (사용 안할 것 같은데)
def prepare_libsvm_file(file_num):
    # CSV 파일 로드
    input_csv = f"./testing/{file_num}/result/normalized_data.csv"
    output_svm = f"./testing/{file_num}/result/dataset.svm"
    data = pd.read_csv(input_csv, header=None)

    # 마지막 열을 라벨로 추출
    labels = data.iloc[:, -1]

    # 나머지 열을 특징으로 추출
    features = data.iloc[:, :-1]

    # libSVM 형식으로 변환
    with open(output_svm, 'w') as f:
        for i in range(len(labels)):
            line = f"{int(labels[i])} " + " ".join([f"{j + 1}:{features.iloc[i, j]}" for j in range(features.shape[1])])
            f.write(line + "\n")

    print("Saved to dataset.svm.")
    return

def execute():
    for i in range(1, 7):
        input_folder = f"./testing/{i}"
        output_folder = f"./testing/{i}"
        result_folder = f"./testing/{i}/result"
        os.makedirs(result_folder, exist_ok=True)
        load_file(input_folder, output_folder)
        cut_to_window(input_folder)
        normalization(input_folder)

# execute()