from utils import timestamp_sync as ts
from utils import feature_value_conversion as fvc
from models import test_data_maker as tdm
from models import train_model as tm
from models import test_data_evaluation as tde

model_path = "../../svm_trained_model.joblib"  # 모델 파일 경로
labels = [1, 2, 3, 4, 5, 6]  # 클래스 라벨

"""
# 1. 전체에 데이터가 분포해 있는 경우
for i in range(1, 7):
    folder = f"../../data/raw/test/{i}"
    ts.adjust_and_sync_files(folder, f"../../data/processed/test/{i}")
    fvc.merge_sensor_files(f"../../data/processed/test/{i}", f"../../data/processed/test/{i}")
    fvc.cut_to_window(f"../../data/processed/test/{i}", f"../../data/processed/test/{i}")
tdm.make_combined_normalized_data("../../data/processed/test")
test_file_path = "../../data/processed/test/normalize_data.csv"
tde.evaluate_saved_model(model_path, test_file_path, labels)# 검증 데이터 파일 경로
"""


# 2. 특정 파일에 있는 데이터를 검증
folder = f"../../data/raw/test/another_data"
ts.adjust_and_sync_files(folder, f"../../data/processed/test/another_data")
fvc.merge_sensor_files(f"../../data/processed/test/another_data", f"../../data/processed/test/another_data")
fvc.cut_to_window(f"../../data/processed/test/another_data", f"../../data/processed/test/another_data")
tdm.normalization_to_test_data("../../data/processed/test/another_data")
test_file_path = "../../data/processed/test/another_data/normalize_data.csv"
tde.evaluate_saved_model(model_path, test_file_path, labels)