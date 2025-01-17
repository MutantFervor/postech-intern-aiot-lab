import pandas as pd
from utils import timestamp_sync as ts
from utils import feature_value_conversion as fvc

# 해당 모델에 저장된 normalizaiton의 params를 이용하여 정규화
def normalization_to_test_data(input_folder):
    # 파일 경로 설정
    data_file = f"{input_folder}/feature_data.csv"
    save_path = f"{input_folder}/normalize_data.csv"

    # 데이터 로드
    data = pd.read_csv(data_file, header=None, skiprows=1)
    columns_to_normalize = list(range(data.shape[1] - 1))
    params = pd.read_csv(f"{input_folder}/../../train/normalize_params.csv") # params

    # normalization
    a_num = b_num = 0
    for column in columns_to_normalize:
        a_i = params.iloc[a_num, 1]
        b_i = params.iloc[b_num, 2]

        data[column] = a_i * data[column] + b_i
        a_num, b_num = a_num + 1, b_num + 1

    # 정규화된 데이터 저장
    data.to_csv(save_path, index=False, header=False)
    print(f"정규화된 데이터를 '{save_path}'로 저장했습니다.")
    return


# test data를 합해서 normalization
def make_combined_normalized_data(input_folder):
    fvc.combine_feature_files(input_folder)
    data_file = f"{input_folder}/combined_feature_data.csv"
    save_path = f"{input_folder}/normalize_data.csv"

    data = pd.read_csv(data_file, header=None)
    columns_to_normalize = list(range(data.shape[1] - 1))

    # params load
    params = pd.read_csv(f"{input_folder}/../train/normalize_params.csv")

    # normalization
    a_num = b_num = 0
    for column in columns_to_normalize:
        a_i = params.iloc[a_num, 1]
        b_i = params.iloc[b_num, 2]

        data[column] = a_i * data[column] + b_i
        a_num, b_num = a_num + 1, b_num + 1

    # 정규화된 데이터 저장
    data.to_csv(save_path, index=False, header=False)
    print(f"정규화된 데이터를 '{save_path}'로 저장했습니다.")
    return