import feature_value_conversion as fvc
import timestamp_sync as ts
import pandas as pd

# 해당 테스트 데이터의 activity class를 찾아야 함
def find_activity_class(data):
    value = data.iloc[0, 36]
    return value

# 해당 모델에 저장된 normalizaiton의 params를 이용하여 정규화
def normalization_to_test_data(input_folder):
    # 파일 경로 설정
    data_file = f"{input_folder}/result/feature.csv"
    save_path = f"{input_folder}/result/normalized_data.csv"

    # 데이터 로드
    data = pd.read_csv(data_file, header=None)  # 헤더 없이 데이터 로드
    columns_to_normalize = list(range(data.shape[1]))

    # params를 가져온다.
    value = find_activity_class(data)
    params = pd.read_csv(f"./testing/{value}/result/normalization_params.csv")

    # 정규화 파라미터 저장용 리스트
    normalization_params = []
    
    a_num = b_num = 0
    
    # 정규화 수행
    for column in columns_to_normalize:
        a_i = params.iloc[a_num, 1]
        b_i = params.iloc[b_num, 2]

        print(a_i)

        # 데이터 normalization
        data[column] = a_i * data[column] + b_i
        
        # 증가
        a_num = a_num + 1
        b_num = b_num + 1

    # 정규화된 데이터 저장
    data.to_csv(save_path, index=False, header=False)
    print(f"정규화된 데이터를 '{save_path}'로 저장했습니다.")

# 실행
folder = './test_data'
ts.adjust_and_sync_files("./test_data", "./test_data", "./test_data")
fvc.load_file(folder, folder)
fvc.cut_to_window(folder)
normalization_to_test_data("./test_data")