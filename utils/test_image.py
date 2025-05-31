import os
import cv2
import numpy as np
import math

def draw_yolov11_annotations(images_dir, labels_dir, class_names, indices):
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()

    for idx in indices:
        if idx < 0 or idx >= len(image_files):
            print(f"Index {idx} không hợp lệ.")
            continue

        img_file = image_files[idx]
        img_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, img_name + '.txt')

        if not os.path.exists(label_path):
            print(f"Không tìm thấy label: {img_name}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Không thể đọc ảnh: {img_path}")
            continue

        h, w = img.shape[:2]

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                label = class_names[class_id] if class_id < len(class_names) else str(class_id)
                color = (0, 255, 0)

                coords = list(map(float, parts[1:]))

                if len(coords) == 4:
                    # Case 1: bbox (x_center, y_center, width, height)
                    x, y, bw, bh = coords
                    x1 = int((x - bw / 2) * w)
                    y1 = int((y - bh / 2) * h)
                    x2 = int((x + bw / 2) * w)
                    y2 = int((y + bh / 2) * h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                elif len(coords) == 5:
                    # Case 2: rotated bbox (x, y, w, h, angle)
                    x, y, bw, bh, angle = coords
                    rect_center = (x * w, y * h)
                    rect_size = (bw * w, bh * h)
                    angle_deg = -angle * 180 / math.pi  # Convert radian to degrees if needed
                    rect = ((rect_center[0], rect_center[1]), rect_size, angle_deg)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
                    cv2.putText(img, label, (int(rect_center[0]), int(rect_center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                elif len(coords) >= 6 and len(coords) % 2 == 0:
                    # Case 3: polygon (x1 y1 x2 y2 ...)
                    points = np.array([(coords[i] * w, coords[i+1] * h) for i in range(0, len(coords), 2)], dtype=np.int32)
                    cv2.polylines(img, [points], isClosed=True, color=(255, 0, 0), thickness=2)
                    x1, y1 = points[0]
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                else:
                    print(f"Dòng không nhận dạng: {line}")

        cv2.imshow(f'Image {idx}: {img_file}', img)
        print(f"Hiển thị: {img_path}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# === Ví dụ gọi hàm ===
class_names = [
  "crosswalk", "License_Plate", "stop_line",
  "Traffic Light -Red-", "Traffic Light -Green-", "Traffic Light -Off-", "Traffic Light -Yellow-",
  "bus", "pickup-van", "microbus", "car", "motorbike", "truck"
]
indices = [0, 123, 200, 5, 100, 20, 400, 50, 600, 100, 200, 150, 210]  # Tùy chỉnh

draw_yolov11_annotations(
    images_dir='datasets/test/images',
    labels_dir='datasets/test/labels',
    class_names=class_names,
    indices=indices
)
