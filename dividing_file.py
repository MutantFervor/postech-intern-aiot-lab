import os
import pandas as pd


def split_csv(file_path, output_train_path, output_test_path, split_ratio=0.2):
    # Load the CSV file
    data = pd.read_csv(file_path)

    # Determine the split index
    split_index = int(len(data) * split_ratio)

    # Split the data
    test_data = data[:split_index]  # First 20% for testing
    train_data = data[split_index:]  # Remaining 80% for training

    # Ensure directories for train and test files exist
    os.makedirs(os.path.dirname(output_train_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_test_path), exist_ok=True)

    # Save the datasets to separate CSV files
    test_data.to_csv(output_test_path, index=False)
    train_data.to_csv(output_train_path, index=False)

    print(f"Data successfully split for {file_path}!")
    print(f"Training data saved to: {output_train_path}")
    print(f"Testing data saved to: {output_test_path}")


def combine_csv_files(file_paths, output_file):
    # 각 CSV 파일을 읽어서 DataFrame으로 저장
    dataframes = [pd.read_csv(file) for file in file_paths]

    # 모든 DataFrame을 하나로 합치기
    combined_df = pd.concat(dataframes, ignore_index=True)

    # 합쳐진 데이터 저장
    combined_df.to_csv(output_file, index=False)

    print(f"Combined CSV saved to {output_file}")

# 예시: CSV 파일 경로 리스트
file_paths = [
    "C:/Users/USER/Desktop/test_data/1/",
    "./data/file2.csv",
    "./data/file3.csv",
    "./data/file4.csv",
    "./data/file5.csv"
]

# 결과 저장 경로
output_file = "./data/combined_file.csv"

# 함수 호출
combine_csv_files(file_paths, output_file)


# Main loop with folder creation logic
# for i in range(1, 7):
    # for file in ['linear.csv', 'gyro.csv', 'gravity.csv']:
        # input_csv = f"./origin_training/{i}/{file}"
        # train_csv = f"./origin_training/{i}/training_data/{file}"
        # test_csv = f"./origin_training/{i}/test_data/{file}"
        # split_csv(input_csv, train_csv, test_csv, split_ratio=0.2)