from ultralytics import YOLO
import cv2
import numpy as np
import time
import gradio as gr
import math


def load_yolo_model(model_path: str = 'yolov12.pt'):
    model = YOLO(model_path)
    return model

# def draw_box(image, box_data, label=None, color=(0, 255, 0), thickness=2):
#     """
#     Vẽ bounding box hoặc polygon lên ảnh, hỗ trợ 3 định dạng:
#     - box_data: list gồm:
#         + 4 giá trị => bbox thường: [x_center, y_center, w, h] (normalized)
#         + 5 giá trị => bbox xoay: [x_center, y_center, w, h, angle (radian)]
#         + >=6 giá trị (số chẵn) => polygon: [x1, y1, x2, y2, ..., xn, yn] (normalized)
#     """

#     h, w = image.shape[:2]

#     if len(box_data) == 4:
#         # Trường hợp 1: bbox thường
#         x, y, bw, bh = box_data
#         x1 = int((x - bw / 2) * w)
#         y1 = int((y - bh / 2) * h)
#         x2 = int((x + bw / 2) * w)
#         y2 = int((y + bh / 2) * h)
#         points = np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], dtype=np.int32)

#     elif len(box_data) == 5:
#         # Trường hợp 2: bbox xoay
#         x, y, bw, bh, angle = box_data
#         center = (x * w, y * h)
#         size = (bw * w, bh * h)
#         angle_deg = -angle * 180 / math.pi  # đổi radian -> độ
#         rect = (center, size, angle_deg)
#         points = cv2.boxPoints(rect).astype(np.int32)

#     elif len(box_data) >= 6 and len(box_data) % 2 == 0:
#         # Trường hợp 3: đa giác
#         points = np.array([
#             (box_data[i] * w, box_data[i+1] * h)
#             for i in range(0, len(box_data), 2)
#         ], dtype=np.int32)

#     else:
#         raise ValueError(f"Dữ liệu không hợp lệ: {box_data}")

#     # Vẽ đa giác
#     pts = points.reshape((-1, 1, 2))
#     image = cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)

#     # Vẽ nhãn
#     if label:
#         x, y = points[0]
#         (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#         cv2.rectangle(image, (x, y - text_h - 4), (x + text_w, y), (255, 255, 255), -1)
#         cv2.putText(image, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

#     return image


# Biến toàn cục lưu trạng thái xe đã vi phạm
def is_crossing_red_light(vehicle_box, stop_line_polygon, traffic_light_status):
    vehicle_poly = np.array(vehicle_box, dtype=np.int32)
    stop_line_poly = np.array(stop_line_polygon, dtype=np.int32)
    inter, _ = cv2.intersectConvexConvex(vehicle_poly, stop_line_poly)
    if traffic_light_status == "Traffic Light -Red-" and inter > 0:
        return True
    return False


# Dữ liệu lưu trạng thái đè vạch từng xe: {vehicle_id: start_time}
over_line_memory = {}

def is_over_line(vehicle_poly, line_poly):
    # vehicle_poly, line_poly: numpy array các điểm polygon
    inter, _ = cv2.intersectConvexConvex(vehicle_poly, line_poly)
    return inter > 0

def track_over_line(vehicle_id, vehicle_poly, line_poly, current_time, threshold_sec=3):
    """
    Theo dõi xe có đè vạch quá thời gian threshold_sec không
    """
    if is_over_line(vehicle_poly, line_poly):
        if vehicle_id not in over_line_memory:
            over_line_memory[vehicle_id] = current_time  # bắt đầu đè vạch
        else:
            # Đã đè vạch trước đó, kiểm tra thời gian
            if current_time - over_line_memory[vehicle_id] >= threshold_sec:
                return True  # Vi phạm đè vạch quá lâu
    else:
        # Nếu không còn đè nữa thì xóa khỏi bộ nhớ
        if vehicle_id in over_line_memory:
            del over_line_memory[vehicle_id]
    return False


def draw_box(image, box, label=None, color=(0, 255, 0)):
    """
    Vẽ đa giác polygon (box là list điểm [(x1,y1), (x2,y2), ...])
    """
    pts = np.array(box, np.int32).reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
    if label:
        x, y = pts[0][0]
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image

def process_image(image, model, stop_line_polygon, traffic_light_status, frame_time):
    """
    Nhận ảnh và mô hình, vẽ box và trạng thái vi phạm lên ảnh.
    Tách riêng 2 loại vi phạm: vượt đèn đỏ và đè vạch.
    """
    results = model.predict(image)[0]

    for i, (box, cls) in enumerate(zip(results.boxes.xyxy, results.boxes.cls)):
        label = model.names[int(cls)]
        x1, y1, x2, y2 = map(int, box[:4])
        vehicle_box = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        vehicle_poly = np.array(vehicle_box, dtype=np.int32)

        # Kiểm tra vi phạm đèn đỏ
        cross_red_light = is_crossing_red_light(vehicle_box, stop_line_polygon, traffic_light_status)

        # Kiểm tra vi phạm đè vạch (quá thời gian 3s)
        over_line_violation = track_over_line(i, vehicle_poly, np.array(stop_line_polygon, dtype=np.int32), frame_time)

        # Xác định trạng thái và màu sắc
        if cross_red_light:
            color = (0, 0, 255)  # đỏ
            label_text = f"{label} - Vượt đèn đỏ"
        elif over_line_violation:
            color = (0, 0, 255)  # đỏ
            label_text = f"{label} - Đè vạch"
        else:
            color = (0, 255, 0)  # xanh
            label_text = label

        image = draw_box(image, vehicle_box, label_text, color=color)

    # Vẽ vạch dừng (stop line)
    if stop_line_polygon:
        stop_pts = np.array(stop_line_polygon, np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [stop_pts], isClosed=True, color=(255, 255, 0), thickness=3)

    return image


def create_ui(model):
    gr.Interface(
        fn=lambda img: process_image(img, model),
        inputs=gr.Image(type="numpy", label="Chọn ảnh"),
        outputs=gr.Image(type="numpy", label="Kết quả"),
        title="Phát hiện vi phạm giao thông"
    ).launch()

def main():
    model_path = "runs/train/yolov12-custom/weights/best.pt"
    model = load_yolo_model(model_path)
    create_ui(model)

if __name__ == "__main__":
    main()
