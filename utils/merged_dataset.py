import os
import shutil
import yaml

# Cấu hình
DATASET_PATH = "datasets"
MERGED_PATH = os.path.join(DATASET_PATH, "merged")
IGNORED_CLASSES = {}

# Lưu mapping global
global_class_names = []
class_name_to_index = {}

def load_yaml_names(data_yaml_path):
    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)
        return data['names']

def get_new_class_index(class_name):
    if class_name in IGNORED_CLASSES:
        return None
    if class_name not in class_name_to_index:
        class_name_to_index[class_name] = len(global_class_names)
        global_class_names.append(class_name)
    return class_name_to_index[class_name]

def process_annotation_file(ann_path, original_names, dest_ann_path):
    new_lines = []
    with open(ann_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            class_idx = int(parts[0])
            class_name = original_names[class_idx]
            new_class_idx = get_new_class_index(class_name)
            if new_class_idx is not None:
                parts[0] = str(new_class_idx)
                new_lines.append(" ".join(parts) + "\n")

    if new_lines:
        with open(dest_ann_path, "w") as f:
            f.writelines(new_lines)
        return True
    return False

# def process_annotation_file(ann_path, original_names, dest_ann_path):
#     new_lines = []
#     with open(ann_path, "r") as f:
#         lines = f.readlines()
#         for line in lines:
#             parts = line.strip().split()
#             if len(parts) < 5:
#                 continue  # Bỏ qua dòng lỗi

#             class_idx = int(parts[0])
#             class_name = original_names[class_idx]

#             # Nếu class bị loại bỏ thì bỏ qua
#             new_class_idx = get_new_class_index(class_name)
#             if new_class_idx is not None:
#                 parts[0] = class_name  # <-- Ghi tên class thay vì index
#                 new_lines.append(" ".join(parts) + "\n")

#     if new_lines:
#         with open(dest_ann_path, "w") as f:
#             f.writelines(new_lines)
#         return True
#     return False


def copy_dataset(dataset_dir):
    print(f"📂 Processing dataset: {dataset_dir}")
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    original_names = load_yaml_names(yaml_path)

    for split in ["train", "valid", "test"]:
        image_dir = os.path.join(dataset_dir, split, "images")
        label_dir = os.path.join(dataset_dir, split, "labels")

        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            continue

        merged_img_dir = os.path.join(MERGED_PATH, split, "images")
        merged_lbl_dir = os.path.join(MERGED_PATH, split, "labels")
        os.makedirs(merged_img_dir, exist_ok=True)
        os.makedirs(merged_lbl_dir, exist_ok=True)

        for file in os.listdir(label_dir):
            if not file.endswith(".txt"):
                continue
            src_ann_path = os.path.join(label_dir, file)
            dst_ann_path = os.path.join(merged_lbl_dir, file)

            # Chuyển nhãn và ghi file
            success = process_annotation_file(src_ann_path, original_names, dst_ann_path)

            if success:
                # Copy ảnh tương ứng nếu còn nhãn hợp lệ
                for ext in [".jpg", ".png", ".jpeg"]:
                    src_img_path = os.path.join(image_dir, file.replace(".txt", ext))
                    if os.path.exists(src_img_path):
                        dst_img_path = os.path.join(merged_img_dir, os.path.basename(src_img_path))
                        shutil.copy2(src_img_path, dst_img_path)
                        break

def save_merged_yaml():
    yaml_path = os.path.join(MERGED_PATH, "data.yaml")
    data = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(global_class_names),
        "names": global_class_names
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)
    print(f"\n✅ Merged YAML saved to: {yaml_path}")
    print(f"🔢 Total classes: {len(global_class_names)}")
    print(f"📛 Classes: {global_class_names}")

def main():
    # Danh sách thư mục dataset con (tên thư mục chứa data.yaml tương ứng)
    subdatasets = [
        "crosswalk", 
        "license plate", 
        "stop line", 
        "traffic-light", 
        "vehicle"
    ]

    for ds in subdatasets:
        copy_dataset(os.path.join(DATASET_PATH, ds))

    save_merged_yaml()
    print("\n🎉 Done merging datasets.")

if __name__ == "__main__":
    main()
