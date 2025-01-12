# linear/gyro/gravity의 timestamp를 동기화시킨다. (0.0*초)
import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 읽기 (첫 번째 행에 열 이름이 없는 경우 header=None 설정)
df = pd.read_csv('./testing/6/linear.csv', header=None)

# 열 이름 설정
df.columns = ['index', 'timestamp', 'x', 'y', 'z']

# 2번째 열을 datetime 형식으로 변환
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 타임스탬프를 인덱스로 설정
df = df.set_index('timestamp')

# x, y, z 값을 시각화
plt.figure(figsize=(15, 8))

# x 값 시각화
plt.plot(df.index, df['x'], label='x', linewidth=1)

# y 값 시각화
plt.plot(df.index, df['y'], label='y', linewidth=1)

# z 값 시각화
plt.plot(df.index, df['z'], label='z', linewidth=1)

# 그래프 스타일 설정
plt.title('x, y, z Values over Time', fontsize=16)
plt.xlabel('Timestamp', fontsize=14)
plt.ylabel('Values', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)

# 그래프 출력
plt.show()