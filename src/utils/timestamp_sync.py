import os
import csv
import pandas as pd
import make_file as mk
import numpy as np

# 1. 파일의 timestamp 조정, 보간
def adjust_and_sync_files(input_folder, output_folder, interval=10_000):
    files = ["linear", "gyro", "gravity"]
    columns = ['activity_class', 'timestamp', 'x', 'y', 'z']
    mk.ensure_directory(output_folder)

    # (1) 파일 읽기
    data_frames = {}
    for file in files:
        file_path = f"{input_folder}/{file}.csv"
        df = pd.read_csv(file_path, header=None).iloc[:, :5]
        df.columns = columns
        data_frames[file] = df

    # (2) 동기화된 파일 생성 및 저장
    synced_files = {}
    for file in files:
        output_path = f"{output_folder}/{file}_revised.csv"
        synced_files[file] = output_path
        data_frames[file].to_csv(output_path, index=False, header=False, float_format='%.9e')

    # (3) 동기화 로직 수행
    readers = {file: csv.reader(open(synced_files[file], 'r')) for file in files}
    synced_data = {key: [] for key in files}

    lines = {file: next(readers[file], None) for file in files}
    while all(lines.values()):
        for file in files:
            while lines[file] is not None:
                if is_valid_line(lines[file]):
                    break
                lines[file] = next(readers[file], None)

        if not all(lines.values()):
            break

        if is_same_timestamp(*[lines[file][1] for file in files]):
            for file in files:
                synced_data[file].append(lines[file])
            lines = {file: next(readers[file], None) for file in files}
        else:
            min_file = find_min_timestamp(*[lines[file][1] for file in files])
            lines[files[min_file]] = next(readers[files[min_file]], None)
    
    # (4) 타임스탬프 file끼리 조정
    for file in files:
        temp_df = pd.DataFrame(synced_data[file], columns=columns)
        adjusted_df = adjust_timestamps(temp_df, interval)
        
        # timestamp 보간
        data_frames[file] = interpolation(adjusted_df, ['x', 'y', 'z'])

        result_path = f"{output_folder}/{file}.csv"
        mk.ensure_directory(os.path.dirname(result_path))
        adjusted_df.to_csv(result_path, index=False, header=False, float_format='%.9e')
        print(f"Saved output file: {result_path}")
    return


# 2. 데이터 유효성 검사
def is_valid_line(line):
    if not line or len(line) != 5:  # 필드 개수가 5개가 아닌 경우
        return False
    try:
        # timestamp가 정수로 변환 가능하고 6자리 이상인지 확인
        timestamp = float(line[1])
        if not timestamp.is_integer() or timestamp < 100000:
            return False
    except (ValueError, AttributeError):  # 변환 불가능한 경우
        return False
    return True


# 2-1. 동일 타임스탬프 확인
def is_same_timestamp(linear_timestamp, gyro_timestamp, gravity_timestamp):
    timestamps = [linear_timestamp, gyro_timestamp, gravity_timestamp]
    comparison_keys = [str(int(ts))[:len(str(int(ts))) - 4] for ts in timestamps]
    return all(key == comparison_keys[0] for key in comparison_keys)


# 2-2. 최소 값을 가진 타임스탬프 찾기
def find_min_timestamp(linear_timestamp, gyro_timestamp, gravity_timestamp):
    timestamps = [linear_timestamp, gyro_timestamp, gravity_timestamp]
    comparison_values = [int(str(int(ts))[:len(str(int(ts))) - 4]) for ts in timestamps]
    return comparison_values.index(min(comparison_values))


# 2-3. Interpolation function
def interpolation(df, columns, z_threshold=3.0):
    for column in columns:
        # (1) Convert invalid values to NaN
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df[column].interpolate(method='linear', inplace=True)

        # (2) Remove outliers using Z-Score
        mean = df[column].mean()
        std = df[column].std()
        df[column] = df[column].apply(lambda x: x if abs((x - mean) / std) <= z_threshold else np.nan)

        # (3) Perform final interpolation
        df[column].interpolate(method='linear', inplace=True)
    return df




# 3. 타임스탬프 조정 함수
def adjust_timestamps(data_frame, interval):
    df = data_frame.copy()
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    timestamps = df['timestamp'].values

    adjustment = 0
    for i in range(1, len(timestamps)):
        expected = timestamps[i - 1] + interval
        current = timestamps[i] - adjustment  # 조정값 반영 후 현재 값 계산
        if current > expected: # 간극 조정을 위해
            adjustment += current - expected
            timestamps[i] = expected
        else:
            timestamps[i] = current

    # 조정된 타임스탬프를 DataFrame에 반영
    df['timestamp'] = timestamps
    return df