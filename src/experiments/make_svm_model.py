from utils import timestamp_sync as ts
from utils import feature_value_conversion as fvc
from models import train_model as tm

# 1. adjust timestamps of files
for i in range(1, 7):
    ts.adjust_and_sync_files(f"../../data/raw/train/{i}", f"../../data/processed/train/{i}")

# 2. feature value conversion
for i in range(1, 7):
   fvc.merge_sensor_files(f"../../data/processed/train/{i}", f"../../data/processed/train/{i}")
   fvc.cut_to_window(f"../../data/processed/train/{i}", f"../../data/processed/train/{i}")
fvc.normalize_data("../../data/processed/train")

# 3. make trained model
file_path = '../../data/processed/train/normalize_data.csv'
model_save_path = '../../svm_trained_model.joblib'
tm.train_and_save_svm(file_path, model_save_path)