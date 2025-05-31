import os
import shutil
import random

# Cấu hình đường dẫn
base_path = "datasets/train"
image_dir = os.path.join(base_path, "images")
label_dir = os.path.join(base_path, "labels")
output_dir = "datasets/folds"  # thư mục chứa 5 folds

# Số lượng fold
k = 20

# Tạo danh sách tất cả các file ảnh
image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
random.shuffle(image_files)

# Chia file thành 5 fold
folds = [[] for _ in range(k)]
for idx, file in enumerate(image_files):
    folds[idx % k].append(file)

# Tạo thư mục và sao chép ảnh và nhãn tương ứng
for fold_idx, fold in enumerate(folds):
    fold_image_dir = os.path.join(output_dir, f"fold{fold_idx + 1}", "images")
    fold_label_dir = os.path.join(output_dir, f"fold{fold_idx + 1}", "labels")
    os.makedirs(fold_image_dir, exist_ok=True)
    os.makedirs(fold_label_dir, exist_ok=True)

    for image_file in fold:
        label_file = os.path.splitext(image_file)[0] + ".txt"

        # Sao chép ảnh
        shutil.copy(os.path.join(image_dir, image_file), os.path.join(fold_image_dir, image_file))
        
        # Sao chép nhãn nếu có
        label_path = os.path.join(label_dir, label_file)
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(fold_label_dir, label_file))
            
    print(f"Fold {fold_idx + 1} created with {len(fold)} images.")
