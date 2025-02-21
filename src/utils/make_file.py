import os

def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"folder created: {path}")
    return