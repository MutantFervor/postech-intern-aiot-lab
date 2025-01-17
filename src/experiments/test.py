import pandas as pd
import numpy as np

def interpolation(df, columns):
    for column in columns:
        # (1) NaN을 골라내 선형 보간
        df[column] = pd.to_numeric(df[column], errors='coerce')
        df[column].interpolate(method='linear', inplace=True)

        # (2) IQR을 적용하여 이상치 제거
        Q1 = df[column].quantile(0.25)  # 1사분위수 (25%)
        Q3 = df[column].quantile(0.75)  # 3사분위수 (75%)
        IQR = Q3 - Q1  # IQR 계산
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # (3) IQR 범위를 벗어난 값을 NaN으로 처리
        df[column] = df[column].apply(lambda x: x if lower_bound <= x <= upper_bound else np.nan)

        # (4) 0을 NaN으로 대체
        df[column] = df[column].replace(0, np.nan)

        # (5) IQR 제거 후 맨 앞/맨 뒤 NaN 처리
        df[column] = df[column].ffill()  # 앞쪽 NaN 처리
        df[column] = df[column].bfill()  # 뒤쪽 NaN 처리

        # (6) 생긴 NaN 값을 다시 선형 보간
        df[column].interpolate(method='linear', inplace=True)
    return df

# 데이터 예제
data = {
    'x': [np.nan, 2, 10, 20, 'kj', 5000, np.nan, np.nan],
    'y': [5, 15, 25, 300, 45, 55, np.nan, np.nan],
    'z': [0.1, 0.2, 0.3, np.nan, 0.5, 1000, 0.7, np.nan]
}
df = pd.DataFrame(data)

# 보간 및 이상치 제거 수행
interpolated_df = interpolation(df, ['x', 'y', 'z'])
print(interpolated_df)