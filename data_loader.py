import kagglehub
import os
import zipfile

def load_dataset():
    path = kagglehub.dataset_download("sshikamaru/car-object-detection")

    # Extract zip files
    for file in os.listdir(path):
        if file.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(path, file), 'r') as zip_ref:
                zip_ref.extractall(path)

    print("Dataset ready at:", path)
    return path
