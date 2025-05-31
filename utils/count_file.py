import os

def count_files(base_path):
    for subset in ['train', 'valid', 'test', "folds/fold1", "folds/fold2", "folds/fold3", "folds/fold4", "folds/fold5"]:
        img_dir = os.path.join(base_path, subset, 'images')
        lbl_dir = os.path.join(base_path, subset, 'labels')

        num_imgs = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(img_dir) else 0
        num_lbls = len([f for f in os.listdir(lbl_dir) if f.endswith('.txt')]) if os.path.exists(lbl_dir) else 0
        print(base_path)
        print(f"📁 {subset.upper()}:")
        print(f"   🖼️  Image files: {num_imgs}")
        print(f"   🏷️  Label files: {num_lbls}\n")


base_dataset_path = ["datasets"]  # hoặc "datasets/merged" nếu merged nằm trong datasets/
for path in base_dataset_path:
    count_files(path)
