import os

# Danh sách nhãn cần xóa (theo index cũ trong YAML)
REMOVE_LABELS = [3, 4, 5, 6]  # traffic_lights, sign_B1, sign_B2, sign_B3

def clean_labels(label_dir):
    for root, _, files in os.walk(label_dir):
        for file in files:
            if file.endswith('.txt'):
                path = os.path.join(root, file)
                with open(path, 'r') as f:
                    lines = f.readlines()
                new_lines = []
                for line in lines:
                    if line.strip() == '':
                        continue
                    class_idx = int(line.split()[0])
                    if class_idx not in REMOVE_LABELS:
                        # Giảm index nếu lớn hơn nhãn bị xóa
                        for idx in sorted(REMOVE_LABELS):
                            if class_idx > idx:
                                class_idx -= 1
                        parts = line.strip().split()
                        parts[0] = str(class_idx)
                        new_lines.append(' '.join(parts))
                # Nếu file không còn object nào, xóa file label và ảnh tương ứng
                if not new_lines:
                    os.remove(path)
                    img_path = path.replace('labels', 'images').rsplit('.', 1)[0] + '.jpg'
                    if os.path.exists(img_path):
                        os.remove(img_path)
                else:
                    with open(path, 'w') as f:
                        f.write('\n'.join(new_lines) + '\n')

for split in ['train', 'valid', 'test']:
    label_dir = f'datasets/{split}/labels'
    clean_labels(label_dir)