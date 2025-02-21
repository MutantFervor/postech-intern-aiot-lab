import os
import pandas as pd
import numpy as np
from scipy.fft import fft
from utils import make_svm_format as msf

# 1. 기존 linear, gyro, gravity file을 timestamp 기준으로 하나로 합병
def merge_sensor_files(input_folder, output_folder):
    files = ["linear", "gyro", "gravity"]
    columns = ["activity_class", "timestamp", "x", "y", "z"]

    # (1) file의 timestamp를 비교
    data_frames = {}
    for file_name in files:
        file_path = f"{input_folder}/{file_name}.csv"
        df = pd.read_csv(file_path, header=None, names=columns)
        df['timestamp_compare'] = np.floor(df['timestamp'] / 1_000_000 * 100) / 100  # Create comparable timestamp
        data_frames[file_name] = df

    # (2) gyro와 gravity의 activity_class과 timestamp를 없애기
    accel = data_frames['linear']
    gyro = data_frames['gyro'].drop(columns=['activity_class', 'timestamp'])
    gravity = data_frames['gravity'].drop(columns=['activity_class', 'timestamp'])

    # (3) 타임스탬프를 기준으로 병합 (timestamp_compare 이용)
    merged_data = pd.merge(accel, gyro, on='timestamp_compare', how='inner', suffixes=('_accel', '_gyro'))
    merged_data = pd.merge(merged_data, gravity, on='timestamp_compare', how='inner', suffixes=('', '_gravity'))
    merged_data = merged_data.drop(columns=['timestamp_compare'])

    # (4) save merged data
    merged_data.to_csv(f"{output_folder}/merged_sensor_data.csv", header=False, index=False)
    print(f"Merged data saved to '{output_folder}/merged_sensor_data.csv'.")
    return


# 2. timestamp 기준으로 window 자르기 (window_size와 over_lap 조정)
def cut_to_window(input_folder, output_folder, window_size=1.0, overlap=0.8):
    # (1) data road 및 열 이름 설정
    data_file = f"{input_folder}/merged_sensor_data.csv"
    save_path = f"{output_folder}/feature_data.csv"
    data = pd.read_csv(data_file, header=None)
    data.columns = ['activity_class', 'timestamp', 'accel_x', 'accel_y', 'accel_z',
                    'gyro_x', 'gyro_y', 'gyro_z', 'gravity_x', 'gravity_y', 'gravity_z']

    # (2) window 크기와 step 간극 설정
    start_timestamp = data.iloc[0, 1]
    end_timestamp = data.iloc[-1, 1]
    window_step = (1 - overlap) * 1_000_000  # 오버랩 간격 (마이크로초 단위)
    window_size_in_microseconds = window_size * 1_000_000  # 윈도우 크기 (마이크로초 단위)

    # 결과 저장용 데이터프레임
    feature_list = []
    
    # (3) timestamp 기준으로 window 자르기
    while start_timestamp + window_size_in_microseconds <= end_timestamp:
        current_window_end = start_timestamp + window_size_in_microseconds

        # 특징 벡터 계산
        features = []
        for sensor in ['accel', 'gyro', 'gravity']:
            # (3) - 1. x, y, z 축별로 데이터 슬라이싱
            window_x = cut_data(data, sensor, 'x', start_timestamp, current_window_end)
            window_y = cut_data(data, sensor, 'y', start_timestamp, current_window_end)
            window_z = cut_data(data, sensor, 'z', start_timestamp, current_window_end)

            # 독립적인 축 데이터를 특징 계산 함수에 전달
            features.extend(calculate_features(window_x, window_y, window_z))

        # 레이블 추가 (activity_class)
        label = data[(data['timestamp'] >= start_timestamp) &
                     (data['timestamp'] <= current_window_end)]['activity_class'].iloc[0]
        features.append(label)  # 레이블 추가
        feature_list.append(features)

        # 타임스탬프 업데이트
        start_timestamp += window_step

    # (4) 결과를 CSV로 저장

    """
    base_headers = ['dc_x', 'h_x', 'e_x', 'r_xy', 'r_xz', 'std_x', 'skew_x', 'mad_x', 'kurtosis_x', 'iqr_x', 'sma',
                     'dc_y', 'h_y', 'e_y', 'r_yz', 'std_y', 'skew_y', 'mad_y', 'kurtosis_y', 'iqr_y',
                     'dc_z', 'h_z', 'e_z', 'std_z', 'skew_z', 'mad_z', 'kurtosis_z', 'iqr_z']
    """

    base_headers = ['dc_x', 'h_x', 'e_x', 'r_xy', 'r_xz', 'std_x',
                    'dc_y', 'h_y', 'e_y', 'r_yz', 'std_y',
                    'dc_z', 'h_z', 'e_z', 'std_z']

    # 레이블 붙은 헤더 생성
    headers = [f"{sensor}_{feature}" for sensor in ['linear', 'gyro', 'gravity'] for feature in base_headers
              ] + ['activity_class']  # 레이블 추가

    feature_df = pd.DataFrame(feature_list, columns=headers)
    feature_df.to_csv(save_path, index=False)
    print(f"Feature extraction complete. Saved to {save_path}.\n")
    return


# 3. timestamp에 따라 data cutting
def cut_data(data, sensor, axis, startTimestamp, stopTimestamp):
    filtered_data = data[(data['timestamp'] >= startTimestamp) & (data['timestamp'] <= stopTimestamp)]
    sensor_start_num = {'accel': 2, 'gyro': 5, 'gravity': 8}
    axis_start_num = {'x': 0, 'y': 1, 'z': 2}

    column_index = sensor_start_num[sensor] + axis_start_num[axis]
    window = filtered_data.iloc[:, column_index].values
    return window


# 3-1. feature 계산 함수 (수정 중)
def calculate_features(window_x, window_y, window_z):
    # 1. DC
    dc_x, dc_y, dc_z = np.mean(window_x), np.mean(window_y), np.mean(window_z)

    # 2. Frequency-domain entropy (no Information entropy)
    def calculate_entropy(window):
        fft_result = np.fft.fft(window)
        magnitude = np.abs(fft_result)  # fft_result는 복소수 형태이므로 절댓값 취해줌
        magnitude[0] = 0 # exception of DC component
        prob_distribution = magnitude ** 2 / np.sum(magnitude ** 2)  # 에너지 기반 확률 분포
        prob_distribution = np.where(prob_distribution == 0, 1e-10, prob_distribution)
        entropy_num = -np.sum(prob_distribution * np.log10(prob_distribution))
        return entropy_num

    h_x = calculate_entropy(window_x)
    h_y = calculate_entropy(window_y)
    h_z = calculate_entropy(window_z)

    # 3. Total Energy of Frequency Spectrum
    def calculate_energy(window):
        fft_result = np.fft.fft(window)
        magnitude = np.abs(fft_result)
        magnitude[0] = 0 # exception of DC component
        energy_sum = np.sum(magnitude ** 2) / len(window)
        return energy_sum

    e_x = calculate_energy(window_x)
    e_y = calculate_energy(window_y)
    e_z = calculate_energy(window_z)

    # 4. Correlation
    r_xy = np.corrcoef(window_x, window_y)[0, 1]
    r_yz = np.corrcoef(window_y, window_z)[0, 1]
    r_xz = np.corrcoef(window_x, window_z)[0, 1]

    # 5. SMA (Signal Magnitude area)
    def calculate_sma(x, y, z):
        return (np.sum(np.abs(x)) + np.sum(np.abs(y)) + np.sum(np.abs(z))) / 3

    sma = calculate_sma(window_x, window_y, window_z)

    # 6. Standard Deviation (표준 편차)
    std_x = np.std(window_x)
    std_y = np.std(window_y)
    std_z = np.std(window_z)

    # 7. skewness
    def calculate_skewness(sensor_data):
        fft_values = np.abs(fft(sensor_data)) # frequency domain 영역에서 실행
        mean_fft = np.mean(fft_values)
        std_fft = np.std(fft_values)
        skewness = np.mean(((fft_values - mean_fft) / std_fft) ** 3)
        return skewness

    skew_x = calculate_skewness(window_x)
    skew_y = calculate_skewness(window_y)
    skew_z = calculate_skewness(window_z)

    # 8. Median Absolute Deviation
    def calculate_mad(window):
        return np.median(np.abs(window - np.median(window)))

    mad_x = calculate_mad(window_x)
    mad_y = calculate_mad(window_y)
    mad_z = calculate_mad(window_z)

    # 9. Kurtosis
    def calculate_kurtosis(sensor_data):
        fft_values = np.abs(fft(sensor_data))  # Magnitude of FFT
        mean_fft = np.mean(fft_values)
        centered_fft = fft_values - mean_fft
        moment_2 = np.mean(centered_fft ** 2)  # 2nd moment
        moment_4 = np.mean(centered_fft ** 4)  # 4th moment
        kurtosis = moment_4 / (moment_2 ** 2)
        return kurtosis

    kurtosis_x = calculate_kurtosis(window_x)
    kurtosis_y = calculate_kurtosis(window_y)
    kurtosis_z = calculate_kurtosis(window_z)

    # 10. Interquartile Range
    def calculate_iqr(window):
        return np.percentile(window, 75) - np.percentile(window, 25)

    iqr_x = calculate_iqr(window_x)
    iqr_y = calculate_iqr(window_y)
    iqr_z = calculate_iqr(window_z)

    # +7. mean_freq
    """
    def calculate_mean_freq(window):
        fft_result = np.fft.fft(window)
        magnitude = np.abs(fft_result)  # Magnitude of FFT
        magnitude[0] = 0  # Exclude DC component (optional)
        indices = np.arange(1, len(magnitude) + 1)

        # Calculate the weighted mean frequency using the provided formula
        numerator = np.sum(indices * magnitude)  # Σ(i * S_i)
        denominator = np.sum(magnitude)  # Σ(S_j)

        mean_freq = numerator / denominator
        return mean_freq

    mean_freq_x = calculate_mean_freq(window_x)
    mean_freq_y = calculate_mean_freq(window_y)
    mean_freq_z = calculate_mean_freq(window_z)

    def calculate_max_freq_ind(window, sample_rate=100, scaling_factor=1.0):
        # Perform FFT
        fft_result = np.fft.fft(window)
        magnitude = np.abs(fft_result)  # Magnitude of FFT (|S_i|)

        # Frequency axis
        freqs = np.fft.fftfreq(len(window), d=1 / sample_rate)

        # Find the index of the maximum magnitude
        max_index = np.argmax(magnitude)  # 0-based index of the maximum value
        max_freq = freqs[max_index]  # Corresponding frequency
        max_freq_ind = max_index * scaling_factor + max_freq

        return max_freq_ind

    max_freq_ind_x = calculate_max_freq_ind(window_x)
    max_freq_ind_y = calculate_max_freq_ind(window_x)
    max_freq_ind_z = calculate_max_freq_ind(window_x)
    """

    return [dc_x, h_x, e_x, r_xy, r_xz, std_x,
            dc_y, h_y, e_y, r_yz, std_y,
            dc_z, h_z, e_z, std_z]

    """
    return [dc_x, h_x, e_x, r_xy, r_xz, std_x, skew_x, mad_x, kurtosis_x, iqr_x, sma,
            dc_y, h_y, e_y, r_yz, std_y, skew_y, mad_y, kurtosis_y, iqr_y,
            dc_z, h_z, e_z, std_z, skew_z, mad_z, kurtosis_z, iqr_z]
    """



# 4. 모든 파일의 feature 파일을 합치는 함수
def combine_feature_files(input_folder, file_path="feature_data.csv"):
    files = [os.path.join(input_folder, str(i), file_path) for i in range(1, 7)]
    combined_data = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
    # combined_data.to_csv(f"{input_folder}/merged_sensor_data.csv", index=False, header=False)
    combined_data.to_csv(f"{input_folder}/combined_feature_data.csv", index=False, header=False)
    print("combined_file made.")
    return


# 5. feature data normalization : using Min-Max Normalization
def normalize_data(input_folder):
    # (1) 파일 합치기
    combine_feature_files(input_folder)
    data_file = f"{input_folder}/combined_feature_data.csv"
    save_path = f"{input_folder}/normalize_data.csv"
    params_path = f"{input_folder}/normalize_params.csv"

    # 데이터 로드
    data = pd.read_csv(data_file, header=None)
    columns_to_normalize = list(range(data.shape[1]))
    columns_to_normalize = columns_to_normalize[:-1]  # exception to activity_class column

    # 정규화 파라미터 저장용 리스트
    normalization_params = []

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

    # SVM 형식으로 저장
    msf.save_as_svm_format(f"{input_folder}/normalize_data.csv", f"{input_folder}/normalize_data.svm")
    print("normalize_data.svm가 저장되었습니다.")
    return