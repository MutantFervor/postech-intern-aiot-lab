# linear/gyro/gravity의 timestamp를 동기화시킨다. (0.0*초)
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
    linear_reader = ([value.strip() for value in line if value.strip() != ''] for line in csv.reader(linear))
    gyro_reader = ([value.strip() for value in line if value.strip() != ''] for line in csv.reader(gyro))
    gravity_reader = ([value.strip() for value in line if value.strip() != ''] for line in csv.reader(gravity))

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

    # count = 0
    while all(lines):
        for i in range(3):
            while lines[i] is not None:  # 파일 끝이 아니면 반복
                # count = count + 1
                print(f"Line: {lines[i]}, Length: {len(lines[i])}")
                if isValidLine(lines[i]):  # 유효한 데이터면 루프 탈출
                    break
                lines[i] = next(readers[i], None)  # 다음 줄로 이동

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
    if not line or len(line) != 5:  # 정확히 5개의 열만 유효
        return False
    try:
        value = float(line[1])
        if value.is_integer() and len(str(int(value))) >= 6:  # 정수이고 6자리 이상인지 확인
            return True
    except ValueError:
        return False
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
# 파일은 기존에 timestamp 기준으로 오름차순으로 작성되어야 결과가 제대로 산출됨
for i in range(1, 3):
    input_folder = f"./testing/{i}"
    output_folder = f"./testing/{i}"
    adjustFile(input_folder, output_folder)