import os

def check_images_labels(base_dir, subset):
    print(f"\n🔍 Checking subset: {subset}")
    image_dir = os.path.join(base_dir, subset, "images")
    label_dir = os.path.join(base_dir, subset, "labels")

    image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]

    images_without_labels = []
    empty_label_files = []
    labels_without_images = []

    for image_file in image_files:
        name = os.path.splitext(image_file)[0]
        label_file = f"{name}.txt"
        label_path = os.path.join(label_dir, label_file)

        if not os.path.exists(label_path):
            images_without_labels.append(image_file)
        else:
            with open(label_path, "r") as f:
                if f.read().strip() == "":
                    empty_label_files.append(label_file)

    for label_file in label_files:
        name = os.path.splitext(label_file)[0]
        found = any(os.path.exists(os.path.join(image_dir, f"{name}{ext}"))
                    for ext in [".jpg", ".png", ".jpeg"])
        if not found:
            labels_without_images.append(label_file)

    print("📌 Ảnh không có nhãn:")
    for f in images_without_labels:
        print(" -", f)

    print("📌 Nhãn rỗng (không có object):")
    for f in empty_label_files:
        print(" -", f)

    print("📌 Nhãn không có ảnh:")
    for f in labels_without_images:
        print(" -", f)


# Gọi hàm cho từng tập
base_dataset_path = "datasets"
for subset in ["train", "valid", "test"]:
    check_images_labels(base_dataset_path, subset)