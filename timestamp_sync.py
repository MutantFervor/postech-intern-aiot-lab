import csv
import pandas as pd
import os

def adjust_and_sync_files(input_folder, output_folder, result_folder, interval=10_000):
    # 파일 경로 설정
    linear_file = f"{input_folder}/linear.csv"
    gyro_file = f"{input_folder}/gyro.csv"
    gravity_file = f"{input_folder}/gravity.csv"

    # 파일 읽기
    linear = open(linear_file, 'r')
    gyro = open(gyro_file, 'r')
    gravity = open(gravity_file, 'r')

    # 동기화된 파일 생성
    linear_revised = f"{output_folder}/linear_revised.csv"
    gyro_revised = f"{output_folder}/gyro_revised.csv"
    gravity_revised = f"{output_folder}/gravity_revised.csv"

    # CSV readers
    linear_reader = ([value.strip() for value in line if value.strip() != ''] for line in csv.reader(linear))
    gyro_reader = ([value.strip() for value in line if value.strip() != ''] for line in csv.reader(gyro))
    gravity_reader = ([value.strip() for value in line if value.strip() != ''] for line in csv.reader(gravity))

    # CSV writers
    with open(linear_revised, 'w', newline='') as linear_out, \
            open(gyro_revised, 'w', newline='') as gyro_out, \
            open(gravity_revised, 'w', newline='') as gravity_out:

        linear_writer = csv.writer(linear_out)
        gyro_writer = csv.writer(gyro_out)
        gravity_writer = csv.writer(gravity_out)

        readers = [linear_reader, gyro_reader, gravity_reader]
        writers = [linear_writer, gyro_writer, gravity_writer]
        lines = [next(linear_reader, None), next(gyro_reader, None), next(gravity_reader, None)]

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

    # 타임스탬프 조정 (result 파일이 없으면 생성하는 로직 필요)
    adjust_timestamps(linear_revised, f"{result_folder}/result/linear.csv", interval)
    adjust_timestamps(gyro_revised, f"{result_folder}/result/gyro.csv", interval)
    adjust_timestamps(gravity_revised, f"{result_folder}/result/gravity.csv", interval)


# 데이터 유효성 검사
def isValidLine(line):
    if not line or len(line) != 5:
        return False
    try:
        value = float(line[1])
        if value.is_integer() and len(str(int(value))) >= 6:
            return True
    except ValueError:
        return False
    return False


# 동일 타임스탬프 확인 (수정 필요)
def isSameTimestamp(linear_timestamp, gyro_timestamp, gravity_timestamp):
    timestamps = [linear_timestamp, gyro_timestamp, gravity_timestamp]
    comparison_keys = [str(int(ts))[:6] if len(str(int(ts))) == 10 else str(int(ts))[:7] for ts in timestamps]
    return all(key == comparison_keys[0] for key in comparison_keys)


# 최소 타임스탬프 찾기 (수정 필요)
def findMinTimestamp(linear_timestamp, gyro_timestamp, gravity_timestamp):
    timestamps = [linear_timestamp, gyro_timestamp, gravity_timestamp]
    comparison_values = [int(str(int(ts))[:6]) if len(str(int(ts))) == 10 else int(str(int(ts))[:7]) for ts in
                         timestamps]
    return comparison_values.index(min(comparison_values))


# 타임스탬프 조정 함수
def adjust_timestamps(input_file, output_file, interval):
    df = pd.read_csv(input_file, header=None)
    df.columns = ['activity_class', 'timestamp', 'x', 'y', 'z']
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
    

# 이상치 처리 interpolation 함수 필요
# def interpolation():


# 실행
# for i in range(1, 7):
    # input_folder = f"./testing/{i}"
    # output_folder = f"./testing/{i}"
    # result_folder = f"./testing/{i}/result"
    # os.makedirs(result_folder, exist_ok=True)
    # adjust_and_sync_files(input_folder, output_folder, result_folder)