import pandas as pd
import numpy as np
from scipy.stats import entropy

# 기존 file을 timestamp 기준으로 하나로 합병
def loadFile(fileNum):
    linear_file = f"./testing/{fileNum}/result/linear.csv"
    gyro_file = f"./testing/{fileNum}/result/gyro.csv"
    gravity_file = f"./testing/{fileNum}/result/gravity.csv"

    # 1. 개별 파일 로드
    accel = pd.read_csv(linear_file)
    gyro = pd.read_csv(gyro_file)
    gravity = pd.read_csv(gravity_file)

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
    print(f"병합된 데이터 크기: {merged_data.shape}")
    print(merged_data.head())

    # 4. 새로운 CSV 파일로 저장
    merged_data.to_csv(f"./testing/{fileNum}/result/merged_data.csv", header=False, index=False)
    print("병합된 데이터를 'merged_data.csv'로 저장했습니다.")


# timestamp 기준으로 1초로 window 자르기 (0.5s씩 겹쳐서)
def cutToWindow(fileNum, window_size=1.0, overlap=0.5):
    data_file = f"./testing/{fileNum}/result/merged_data.csv"
    save_path = f"./testing/{fileNum}/result/feature.csv"
    data = pd.read_csv(data_file)

    # data_file에 index 부여
    data.columns = ['activity_class', 'timestamp', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'
                    , 'gravity_x', 'gravity_y', 'gravity_z']

    # 0.5초 기준 tuple을 잘라 window를 저장한다. (대략 50개 정도, EOF까지 반복)
    startTimestamp = data.iloc[0, 1]
    endTimestamp = data.iloc[-1, 1]
    window_step = overlap * 1_000_000  # 오버랩(초) 단위

    # 첫 번째 파일에 헤더 생성
    with open(save_path, 'w') as f:
        f.write(','.join([f'feature_{i}' for i in range(36)]) + ',label\n')  # 헤더 추가

    # 슬라이딩 윈도우 반복
    while startTimestamp + window_size * 1_000_000 <= endTimestamp:
        stopTimestamp = startTimestamp + window_size * 1_000_000

        # feature vector 계산
        features = []
        for sensor in ['accel', 'gyro', 'gravity']:
            # x, y, z 축별로 데이터 슬라이싱
            window_x = cutData(data, sensor, 'x', startTimestamp, stopTimestamp)
            window_y = cutData(data, sensor, 'y', startTimestamp, stopTimestamp)
            window_z = cutData(data, sensor, 'z', startTimestamp, stopTimestamp)

            # 독립적인 축 데이터를 calculateFeatures에 전달
            features.extend(calculateFeatures(window_x, window_y, window_z))

        # 레이블 추가 (activity_class)
        label = \
        data[(data['timestamp'] >= startTimestamp) & (data['timestamp'] <= stopTimestamp)]['activity_class'].iloc[0]
        features.append(label)  # 레이블 추가

        # 결과 저장 (CSV에 추가)
        with open(save_path, 'a') as f:
            f.write(','.join(map(str, features)) + '\n')

        # 타임스탬프 업데이트
        startTimestamp += window_step

    print(f"Feature extraction complete. Saved to {save_path}.")
    return


# feature 계산 함수
def calculateFeatures(window_x, window_y, window_z):
    # 1. DC (평균값)
    dc_x, dc_y, dc_z = np.mean(window_x), np.mean(window_y), np.mean(window_z)

    # 2. Information Entropy
    h_x = entropy(np.array(window_x) / sum(window_x), base=2)
    h_y = entropy(np.array(window_y) / sum(window_y), base=2)
    h_z = entropy(np.array(window_z) / sum(window_z), base=2)

    # 3. Total Energy of Frequency Spectrum
    e_x = np.sum(np.abs(np.fft.fft(window_x)) ** 2) / len(window_x)
    e_y = np.sum(np.abs(np.fft.fft(window_y)) ** 2) / len(window_y)
    e_z = np.sum(np.abs(np.fft.fft(window_z)) ** 2) / len(window_z)

    # 4. Correlation
    r_xy = np.corrcoef(window_x, window_y)[0, 1]
    r_yz = np.corrcoef(window_y, window_z)[0, 1]
    r_xz = np.corrcoef(window_x, window_z)[0, 1]

    # 12개
    return [dc_x, dc_y, dc_z, h_x, h_y, h_z, e_x, e_y, e_z, r_xy, r_yz, r_xz]


# timestamp에 따라 data cutting
def cutData(data, sensor, axis, startTimestamp, stopTimestamp):
    filtered_data = data[(data['timestamp'] >= startTimestamp) & (data['timestamp'] <= stopTimestamp)]
    sensor_start_num = {'accel': 2, 'gyro': 5, 'gravity': 8}
    axis_start_num = {'x': 0, 'y': 1, 'z': 2}

    column_index = sensor_start_num[sensor] + axis_start_num[axis]
    window = filtered_data.iloc[:, column_index].values
    return window


# feature vector를 정규화한다
# def nomarlization():


# 데이터를 libSVM에 넣고 작동시킨다.
# def svmClassifier():

# 실행 (temp)
for i in range(1, 2):
    loadFile(i)
    cutToWindow(i)
