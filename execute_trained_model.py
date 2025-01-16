import timestamp_sync as ts
import feature_value_conversion as fvc
import svm_classifier as sc

ts.execute()
fvc.execute()

file_path = './testing/normalized_data.csv'
model_save_path = './svm_trained_model.joblib'
sc.train_and_save_svm(file_path, model_save_path)