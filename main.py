import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO
import easyocr
import json
from datetime import datetime
import os

# ======= 1. Load tất cả mô hình ========
model_paths = {
    "vehicle": "best_vehicle.pt",
    "license_plate": "best_license_plate.pt",
    "crosswalk": "best_crosswalk.pt",
    "stop_line": "best_stop_line.pt",
    "traffic_light": "best_traffic_light.pt"
}
models = {name: YOLO(path) for name, path in model_paths.items()}
reader = easyocr.Reader(['en'])  # Đọc tiếng Anh, có thể thêm 'vi' nếu cần

# ======= 2. Hàm detect từ tất cả mô hình ========
def detect_all_objects(image):
    results_all = {}
    for name, model in models.items():
        results_all[name] = model(image)[0]
    return results_all

# ======= 3. Vẽ bounding boxes ========
def draw_all_boxes(image, results_all):
    colors = {
        "vehicle": (0, 255, 0),
        "license_plate": (255, 255, 0),
        "crosswalk": (0, 0, 255),
        "stop_line": (255, 0, 0),
        "traffic_light": (255, 0, 255)
    }

    for obj_name, results in results_all.items():
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = results.names[cls_id]
            color = colors.get(obj_name, (255, 255, 255))

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# ======= 4. Kiểm tra va chạm giữa các box ========
def boxes_overlap(boxA, boxB):
    xA1, yA1, xA2, yA2 = map(int, boxA)
    xB1, yB1, xB2, yB2 = map(int, boxB)

    bottom_left = (xA1, yA2)
    bottom_right = (xA2, yA2)

    def point_in_box(point, box):
        x, y = point
        bx1, by1, bx2, by2 = map(int, box)
        return bx1 <= x <= bx2 and by1 <= y <= by2

    return point_in_box(bottom_left, boxB) or point_in_box(bottom_right, boxB)

# ======= 5. Nhận diện màu đèn giao thông ========
def detect_traffic_light_color(results_traffic_light):
    for box in results_traffic_light.boxes:
        cls_id = int(box.cls[0])
        label = results_traffic_light.names[cls_id]
        if "Red" in label:
            return "red"
        elif "Green" in label:
            return "green"
        elif "Yellow" in label:
            return "yellow"
    return "unknown"

# ======= 6. Kiểm tra vi phạm và ghi lại box vi phạm ========
def check_violations(results_all):
    violations_text = []
    violating_boxes = []

    vehicle_boxes = results_all["vehicle"].boxes
    stop_line_boxes = results_all["stop_line"].boxes
    crosswalk_boxes = results_all["crosswalk"].boxes
    traffic_light_results = results_all["traffic_light"]

    light_color = detect_traffic_light_color(traffic_light_results)

    for vbox in vehicle_boxes:
        vbox_xy = vbox.xyxy[0]
        over_stopline = any(boxes_overlap(vbox_xy, sbox.xyxy[0]) for sbox in stop_line_boxes)
        over_crosswalk = any(boxes_overlap(vbox_xy, cbox.xyxy[0]) for cbox in crosswalk_boxes)

        violated = False
        if light_color == "red":
            if over_stopline or over_crosswalk:
                violations_text.append("🚦 Red Light Violation: Vehicle crossed stop line or crosswalk during red light.")
                violated = True
        else:
            if over_crosswalk:
                violations_text.append("🚫 Crosswalk Violation: Vehicle stopped on or crossed the pedestrian crosswalk.")
                violated = True

        if violated:
            violating_boxes.append(vbox_xy)

    return "\n".join(set(violations_text)) if violations_text else "✅ No violations detected.", violating_boxes

# ======= 7. Cắt ảnh từ các box vi phạm ========
def crop_boxes_from_image(image, boxes_xyxy):
    cropped_images = []
    for box in boxes_xyxy:
        x1, y1, x2, y2 = map(int, box)
        cropped = image[y1:y2, x1:x2]
        if cropped.size > 0:
            # Chuyển đổi không gian màu sang LAB
            lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Áp dụng CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l)

            # Khử nhiễu nhẹ
            l_denoised = cv2.fastNlMeansDenoising(l_clahe, None, h=10)

            # Tăng cường độ nét
            gaussian_blurred = cv2.GaussianBlur(l_denoised, (5, 5), 1.5)
            sharpened = cv2.addWeighted(l_denoised, 1.5, gaussian_blurred, -0.5, 0)

            # Ghép lại kênh và chuyển về không gian màu BGR
            lab_enhanced = cv2.merge((sharpened, a, b))
            enhanced_image = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

            cropped_images.append((enhanced_image, box))
    return cropped_images


# ======= 8. Hàm xử lý chính ========
def process_image(image_input):
    results_all = detect_all_objects(image_input)
    image_with_boxes = draw_all_boxes(image_input.copy(), results_all)
    violations_text, violating_boxes = check_violations(results_all)
    cropped_data = crop_boxes_from_image(image_input, violating_boxes)

    license_texts = []
    cropped_license_images = []

    for idx, (cropped_img, box_xy) in enumerate(cropped_data):
        lp_results = models["license_plate"](cropped_img)[0]

        for box in lp_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            pad = 3  # nới rộng 3 pixel mỗi chiều
            x1_global = max(0, x1 + int(box_xy[0]) - pad)
            y1_global = max(0, y1 + int(box_xy[1]) - pad)
            x2_global = min(image_input.shape[1], x2 + int(box_xy[0]) + pad)
            y2_global = min(image_input.shape[0], y2 + int(box_xy[1]) + pad)

            lp_crop = image_input[y1_global:y2_global, x1_global:x2_global]
            if lp_crop.size == 0:
                continue

            # Cải tiến tiền xử lý ảnh biển số
            lp_gray = cv2.cvtColor(lp_crop, cv2.COLOR_BGR2GRAY)
            lp_equalized = cv2.equalizeHist(lp_gray)
            lp_denoised = cv2.fastNlMeansDenoising(lp_equalized, h=30)
            sharpen_kernel = np.array([[0, -1, 0],
                                       [-1, 5.75, -1],
                                       [0, -1, 0]])
            lp_sharpened = cv2.filter2D(lp_denoised, -1, sharpen_kernel)
            lp_final = cv2.cvtColor(lp_sharpened, cv2.COLOR_GRAY2BGR)

            cropped_license_images.append(lp_final)

            ocr_result = reader.readtext(lp_final)
            if ocr_result:
                ocr_result_sorted = sorted(ocr_result, key=lambda x: x[0][0][1])
                text = ' '.join([res[1] for res in ocr_result_sorted])
                license_texts.append(f"🚘 Vehicle {idx+1} Plate Number: {text}")
            else:
                license_texts.append(f"🚘 Vehicle {idx+1} Plate Number: Not recognized")

    license_summary = "\n".join(license_texts) if license_texts else "🔍 No license plates detected or recognized."
    
    if license_texts:
        for text in license_texts:
            if "Plate Number: Not recognized" in text:
                continue

            license_number = text.split("Plate Number: ")[-1].strip()
            violation_type = "VUOT_DEN_DO" if "Red Light" in violations_text else "DUNG_LEN_VACH"
            timestamp = datetime.now().isoformat(timespec='seconds')
            location = "Ngã tư Trần Duy Hưng"

            new_record = {
                "licensePlate": license_number,
                "violationType": violation_type,
                "timestamp": timestamp,
                "location": location
            }

            json_path = "violation_result.json"
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        if not isinstance(data, list):
                            data = []
                    except json.JSONDecodeError:
                        data = []
            else:
                data = []

            data.append(new_record)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            break  # chỉ ghi 1 bản ghi đầu tiên nếu có nhiều biển số

    return image_with_boxes, violations_text + "\n\n" + license_summary, [img for img, _ in cropped_data] + cropped_license_images

# ======= 9. Giao diện Gradio ========
demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Image(type="numpy", label="Image with Bounding Boxes"),
        gr.Textbox(label="Violation Summary"),
        gr.Gallery(label="Violated Vehicles (Cropped)")
    ],
    title="🚓 Traffic Violation Detection System",
    description="Detects vehicles, traffic lights, stop lines, and crosswalks to flag red light or crosswalk violations."
)

if __name__ == "__main__":
    demo.launch()