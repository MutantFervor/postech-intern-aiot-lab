import pandas as pd

def adjust_timestamps(input_file, output_file):
    # CSV 파일 읽기
    df = pd.read_csv(input_file, header=None)
    df.columns = ['activity_class', 'timestamp', 'x', 'y', 'z']  # 열 이름 설정

    # 타임스탬프 수정 작업
    timestamps = df['timestamp'].copy()
    for i in range(1, len(timestamps)):
        # 이전 타임스탬프에 10ms(10,000 마이크로초)를 더함
        expected = int(str(timestamps[i-1])[:6]) + 1  # 6번째 자리만 증가
        current = int(str(timestamps[i])[:6])        # 현재 타임스탬프의 앞 6자리
        if current > expected:  # 간격이 너무 크다면, 6번째 자리 보정
            timestamps[i] = int(f"{expected}{str(timestamps[i])[6:]}")
        else:
            timestamps[i] = int(str(timestamps[i]))

    # 수정된 타임스탬프를 데이터프레임에 반영
    df['timestamp'] = timestamps

    # 데이터를 그대로 저장 (소수점 표현 유지)
    df.to_csv(output_file, index=False, header=False, float_format='%.12g')  # 소수점 자리 유지
    print(f"수정된 파일이 저장되었습니다: {output_file}")

# 실행 예제
for i in range(3, 4):
    input_folder = f"./testing/{i}/gravity.csv"
    output_folder = f"./testing/{i}/result/gravity.csv"
    adjust_timestamps(input_folder, output_folder)

    input_folder = f"./testing/{i}/gyro.csv"
    output_folder = f"./testing/{i}/result/gyro.csv"
    adjust_timestamps(input_folder, output_folder)

    input_folder = f"./testing/{i}/linear.csv"
    output_folder = f"./testing/{i}/result/linear.csv"
    adjust_timestamps(input_folder, output_folder)