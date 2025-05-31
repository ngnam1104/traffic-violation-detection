import os
from collections import defaultdict

names = ["crosswalk", "License_Plate", "stop_line", "traffic_lights", "sign_B1", "sign_B2", "sign_B3", "Traffic Light -Red", "Traffic Light -Green", "Traffic Light -Off", "Traffic Light -Yellow", "bus", "pickup-van", "microbus", "car", "motorbike", "truck"]

def count_labels_in_yolo_format(label_dir):
    label_counts = defaultdict(int)
    for file_name in os.listdir(label_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(label_dir, file_name)
            with open(file_path, 'r') as f:
                lines = f.readlines()
                classes_in_file = set()
                for line in lines:
                    class_id = line.strip().split()[0]
                    classes_in_file.add(class_id)
                for class_id in classes_in_file:
                    label_counts[class_id] += 1
    return label_counts

if __name__ == "__main__":
    base_path = "datasets"
    subfolders = ["train/labels", "valid/labels", "test/labels"]
    total_counts = defaultdict(int)

    for subfolder in subfolders:
        label_dir = os.path.join(base_path, subfolder)
        if not os.path.exists(label_dir):
            continue
        print(f"Đếm trong thư mục: {label_dir}")
        counts = count_labels_in_yolo_format(label_dir)
        for class_id, count in counts.items():
            total_counts[class_id] += count
            print(f"  Class {names[int(class_id)]}: {count} ảnh")

    print("\n📊 Tổng cộng trong tập 'merged':")
    for class_id, count in sorted(total_counts.items(), key=lambda x: int(x[0])):
        print(f"Class {names[int(class_id)]}: {count} ảnh")
