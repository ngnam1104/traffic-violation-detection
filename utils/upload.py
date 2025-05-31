import os
from roboflow import Roboflow

rf = Roboflow(api_key="LSFCpy2aQiddjn2WPysY")
workspace = rf.workspace("traffic-violation-detection-tfosp")
project = workspace.project("traffic-violation-detection-1x8ri")

def upload_folder(folder_path):
    total_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
    upload_result = project.upload(folder_path, num_workers=4)
    print(f"[✔] Uploaded {total_files} files from '{folder_path}'")

upload_folder("datasets/train/images")
upload_folder("datasets/valid/images")
upload_folder("datasets/test/images")
