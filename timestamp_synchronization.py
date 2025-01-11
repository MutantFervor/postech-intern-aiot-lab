import csv

# main_file 기준으로 target_file을 정리한다.
def adjustFile(input_folder, output_folder):
    # open file
    linear_file = f"{input_folder}/linear.csv"
    gyro_file = f"{input_folder}/gyro.csv"
    gravity_file = f"{input_folder}/gravity.csv"

    linear = open(linear_file, 'r')
    gyro = open(gyro_file, 'r')
    gravity = open(gravity_file, 'r')

    # Create new files (open 'a' mode)
    linear_revised = open(f"{output_folder}/linear_revised.csv", 'a', newline='')
    gyro_revised = open(f"{output_folder}/gyro_revised.csv", 'a', newline='')
    gravity_revised = open(f"{output_folder}/gravity_revised.csv", 'a', newline='')

    # CSV readers
    linear_reader = csv.reader(linear)
    gyro_reader = csv.reader(gyro)
    gravity_reader = csv.reader(gravity)

    # CSV writers
    linear_writer = csv.writer(linear_revised)
    gyro_writer = csv.writer(gyro_revised)
    gravity_writer = csv.writer(gravity_revised)

    # read first line
    linear_line = next(linear_reader, None)
    gyro_line = next(gyro_reader, None)
    gravity_line = next(gravity_reader, None)

    readers = [linear_reader, gyro_reader, gravity_reader]
    writers = [linear_writer, gyro_writer, gravity_writer]
    lines = [linear_line, gyro_line, gravity_line]

    while all(lines):
        # Skip invalid or blank lines
        for i in range(3):
            while lines[i] is not None and (not isValidLine(lines[i]) or len(lines[i]) == 0):
                lines[i] = next(readers[i], None)
            if lines[i] is None:  # Stop if end of file is reached
                break

        # Check if all lines are valid
        if not all(lines):
            break

        # timestamp 정보 넘기기 (2번째에 위치)
        if isSameTimestamp(lines[0][1], lines[1][1], lines[2][1]):
            for i in range(3):  # Write each line to corresponding file
                writers[i].writerow(lines[i])

            # go to next line
            for i in range(3):
                lines[i] = next(readers[i], None)
        else:
            tempNum = findMinTimestamp(lines[0][1], lines[1][1], lines[2][1])
            lines[tempNum] = next(readers[tempNum], None)

# data가 유효한지 검사
def isValidLine(line):
    if not line or len(line) != 5:  # 최소 5개의 열이 있어야 유효
        return False
    # 숫자인지 확인 (소수점, 지수 표기법 등 포함)
    try:
        float(line[1])  # 두 번째 열(timestamp)이 숫자인지 확인
        return True
    except ValueError:
        return False

# 세 timeStamp가 같은지 확인 함수
def isSameTimestamp(linear_timestamp, gyro_timestamp, gravity_timestamp):
    # 문자열로 변환 후 앞 6자리 비교
    timestamps = [
        str(int(linear_timestamp))[:6],
        str(int(gyro_timestamp))[:6],
        str(int(gravity_timestamp))[:6]
    ]
    return all(t == timestamps[0] for t in timestamps)

# 최소 timestmap 찾기
def findMinTimestamp(linear_timestamp, gyro_timestamp, gravity_timestamp):
    # 문자열로 변환 후 앞 6자리 추출
    timestamps = [
        int(str(int(linear_timestamp))[:6]),
        int(str(int(gyro_timestamp))[:6]),
        int(str(int(gravity_timestamp))[:6])
    ]
    return timestamps.index(min(timestamps))

# 실행 (나중에 파일 경로 작성, 상대경로로 작성)
for i in range(1, 7):
    input_folder = f"./testing/{i}"
    output_folder = f"./testing/{i}"
    adjustFile(input_folder, output_folder)