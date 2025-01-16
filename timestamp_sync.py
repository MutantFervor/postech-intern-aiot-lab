import csv
import os
import pandas as pd
import numpy as np


def adjust_and_sync_files(input_folder, output_folder, result_folder, interval=10_000):
    # 파일 경로 설정
    linear_file = f"{input_folder}/linear.csv"
    gyro_file = f"{input_folder}/gyro.csv"
    gravity_file = f"{input_folder}/gravity.csv"

    # 파일 읽기
    linear = pd.read_csv(linear_file, header=None)
    gyro = pd.read_csv(gyro_file, header=None)
    gravity = pd.read_csv(gravity_file, header=None)
    linear = linear.iloc[:, :5]
    gyro = gyro.iloc[:, :5]
    gravity = gravity.iloc[:, :5]

    # 열 이름 설정
    linear.columns = ['activity_class', 'timestamp', 'x', 'y', 'z']
    gyro.columns = ['activity_class', 'timestamp', 'x', 'y', 'z']
    gravity.columns = ['activity_class', 'timestamp', 'x', 'y', 'z']

    # 보간 과정 추가
    linear = interpolation(linear, linear.columns)
    gyro = interpolation(gyro, gyro.columns)
    gravity = interpolation(gravity, gravity.columns)

    # 동기화된 파일 생성
    linear_revised = f"{output_folder}/linear_revised.csv"
    gyro_revised = f"{output_folder}/gyro_revised.csv"
    gravity_revised = f"{output_folder}/gravity_revised.csv"

    # 파일 저장을 위한 writers 준비
    with open(linear_revised, 'w', newline='', encoding='utf-8') as linear_out, \
            open(gyro_revised, 'w', newline='', encoding='utf-8') as gyro_out, \
            open(gravity_revised, 'w', newline='', encoding='utf-8') as gravity_out:

        linear_writer = csv.writer(linear_out)
        gyro_writer = csv.writer(gyro_out)
        gravity_writer = csv.writer(gravity_out)

        # DataFrame을 리스트로 변환
        linear_list = linear.values.tolist()
        gyro_list = gyro.values.tolist()
        gravity_list = gravity.values.tolist()

        readers = [iter(linear_list), iter(gyro_list), iter(gravity_list)]
        writers = [linear_writer, gyro_writer, gravity_writer]
        lines = [next(readers[0], None), next(readers[1], None), next(readers[2], None)]

        # 동기화 로직 시작
        while all(lines):
            for i in range(3):
                while lines[i] is not None:
                    if isValidLine(lines[i]):
                        break
                    lines[i] = next(readers[i], None)

            if not all(lines):
                break

            if isSameTimestamp(lines[0][1], lines[1][1], lines[2][1]):
                for i in range(3):
                    writers[i].writerow(lines[i])
                lines = [next(reader, None) for reader in readers]
            else:
                tempNum = findMinTimestamp(lines[0][1], lines[1][1], lines[2][1])
                lines[tempNum] = next(readers[tempNum], None)

    # 타임스탬프 조정
    adjust_timestamps(linear_revised, f"{result_folder}/result/linear.csv", interval)
    adjust_timestamps(gyro_revised, f"{result_folder}/result/gyro.csv", interval)
    adjust_timestamps(gravity_revised, f"{result_folder}/result/gravity.csv", interval)


# 데이터 유효성 검사
def isValidLine(line):
    if not line or len(line) != 5:
        return False
    try:
        value = float(line[1])
        if value.is_integer() and len(str(int(value))) >= 6:  # 6자리 미만일 경우 비교가 불가
            return True
    except ValueError:
        return False
    return False


# 동일 타임스탬프 확인
def isSameTimestamp(linear_timestamp, gyro_timestamp, gravity_timestamp):
    timestamps = [linear_timestamp, gyro_timestamp, gravity_timestamp]
    comparison_keys = [
        str(int(ts))[:len(str(int(ts))) - 4] for ts in timestamps
    ]
    return all(key == comparison_keys[0] for key in comparison_keys)


# 최소 타임스탬프 찾기
def findMinTimestamp(linear_timestamp, gyro_timestamp, gravity_timestamp):
    timestamps = [linear_timestamp, gyro_timestamp, gravity_timestamp]
    comparison_values = [
        int(str(int(ts))[:len(str(int(ts))) - 4]) for ts in timestamps
    ]
    return comparison_values.index(min(comparison_values))


# 타임스탬프 조정 함수
def adjust_timestamps(input_file, output_file, interval):
    df = pd.read_csv(input_file, header=None)
    df.columns = ['activity_class', 'timestamp', 'x', 'y', 'z']
    df['activity_class'] = df['activity_class'].astype(int)

    timestamps = df['timestamp'].copy()
    for i in range(1, len(timestamps)):
        expected = timestamps[i - 1] + interval
        if timestamps[i] < expected:
            timestamps[i] = expected
        elif timestamps[i] > expected:
            timestamps[i] = expected
    df['timestamp'] = timestamps

    df.to_csv(output_file, index=False, header=False, float_format='%.9e')
    print(f"수정된 파일이 저장되었습니다: {output_file}")


# 보간 함수
def interpolation(df, columns):
    """
    NaN 값을 선형 보간으로 채우는 함수
    :param df: DataFrame
    :param columns: 보간할 열 리스트
    :return: 보간된 DataFrame
    """
    for column in columns:
        # 선형 보간 수행
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df[column].interpolate(method='linear', inplace=True)
    return df

def execute():
    for i in range(1, 7):
        input_folder = f"./testing/{i}"
        output_folder = f"./testing/{i}"
        result_folder = f"./testing/{i}"
        os.makedirs(result_folder, exist_ok=True)
        adjust_and_sync_files(input_folder, output_folder, result_folder)