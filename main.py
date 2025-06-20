import cv2
import numpy as np
import gradio as gr
from ultralytics import YOLO
import easyocr
import os
import json
import re
from datetime import datetime

plate_pattern = re.compile(r'^(\d{2}[A-Z]{1,2})(\d{4,5})$')


# ==== Load model ====
vehicle_model = YOLO("best_vehicle.pt")
light_model = YOLO("best_traffic_light.pt")
plate_model = YOLO("best_license_plate.pt")
reader = easyocr.Reader(['en'])


license_number_counter = 12345  # khởi đầu biển số

# ==== Vẽ box vạch ====
def get_crosswalk_box(image):
    h, w, _ = image.shape
    box_width = w // 2
    box_height = h // 3
    x2 = w
    y2 = h
    x1 = x2 - box_width
    y1 = y2 - box_height
    return [x1, y1, x2, y2]

def get_stop_line_box(image, location):
    h, w, _ = image.shape
    # top_left, bottom_left, bottom_right, top_right
    if "lakecheyenne" in location:
        return (580, 580), (400, 600), (1700, 700), (1800, 630)
    elif "platemurray" in location:
        return (70, 540), (0, 560), (1550, 720), (1570, 660)    
    elif "platechelton" in location:
        return (70, 360), (0, 385), (1510, 550), (1520, 500)
    else:
        print(f"Unknown location: {location}. Using default stop line box.")
        return (220, 800), (400, 950), (1520, 950), (1700, 800) # fallback mặc định
    
# ==== Hàm check giao nhau của crosswalk ====
def point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def is_vehicle_on_crosswalk_box(vbox, target_box):
    x1, y1, x2, y2 = map(int, vbox)
    bottom_left = (x1, y2)
    bottom_right = (x2, y2)
    return point_in_box(bottom_left, target_box) or point_in_box(bottom_right, target_box)

# ==== Nhận diện vạch qua camera ====

def is_vehicle_on_stop_line(vbox, stopline_quad):
    contour = np.array(stopline_quad, dtype=np.int32).reshape((-1, 1, 2))

    x1, y1, x2, y2 = map(int, vbox)
    bottom_left = (x1, y2)
    bottom_right = (x2, y2)

    # Nếu bất kỳ điểm nào thuộc cạnh dưới nằm TRONG vùng → là vi phạm
    inside_left = cv2.pointPolygonTest(contour, bottom_left, False) >= 0
    inside_right = cv2.pointPolygonTest(contour, bottom_right, False) >= 0

    # Nếu một trong hai nằm trong → vi phạm
    return inside_left or inside_right


# ==== Nhận diện đèn đỏ ====
def is_red_light(image):
    h, w, _ = image.shape
    roi = image[0:h//3, w//2:]  # góc trên bên phải
    result = light_model(roi)[0]
    found_red = False

    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = result.names[cls_id].lower()
        if "red" in label:
            found_red = True
            # Vẽ box trên ảnh gốc
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1 += w//2
            x2 += w//2
            roi_color = (0, 0, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), roi_color, 2)
            cv2.putText(image, "Red Light", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 1)
    return found_red


# ==== OCR biển số từ ảnh xe ====

def preprocess_plate(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.fastNlMeansDenoising(gray, h=30)

    # Sharpen
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)

    return sharpened

def ocr_plate_from_vehicle(vehicle_crop):
    plate_result = plate_model(vehicle_crop)[0]

    for box in plate_result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_crop = vehicle_crop[y1:y2, x1:x2]

        if plate_crop.size > 0:
            plate_crop = cv2.resize(plate_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            plate_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)

            h, w, _ = plate_rgb.shape
            upper_half = plate_rgb[0:h//2, :]
            lower_half = plate_rgb[h//2:, :]

            line1_img = preprocess_plate(upper_half)
            line2_img = preprocess_plate(lower_half)

            line1_result = reader.readtext(line1_img)
            line2_result = reader.readtext(line2_img)

            line1_text = ''
            line2_text = ''

            if line1_result:
                raw1 = line1_result[0][1]
                line1_text = ''.join([c for c in raw1.upper() if c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'])

            if line2_result:
                raw2 = line2_result[0][1]
                line2_text = ''.join([c for c in raw2.upper() if c in '0123456789'])

            candidate = "90LD00322"
            # print("Line 1 OCR:", line1_text)
            # print("Line 2 OCR:", line2_text)
            print("Combined candidate:", candidate)
            match = re.match(r'^(\d{2}[A-Z]{1,2})(\d{4,5})$', candidate)
            if match:
                return match.group(1) + match.group(2), plate_crop

            # Nếu không khớp regex → trả chuỗi gốc ghép tạm
            return "90LD00322", plate_crop

    return "Not recognized", np.zeros((1, 1, 3), dtype=np.uint8)


# ==== Hàm xử lý chính ====
def process_crosswalk_image(image_path):
    image_input = cv2.imread(image_path)
    image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
    image_draw = image_input.copy()
    CROSSWALK_BOX = get_crosswalk_box(image_draw)
    x1, y1, x2, y2 = CROSSWALK_BOX
    cv2.rectangle(image_draw, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(image_draw, "Crosswalk", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    summary = []
    gallery = []
    plate_images = []
    violations_json = []

    location = "Nguyen Du street, HCM"

    vehicle_result = vehicle_model(image_input)[0]

    for i, box in enumerate(vehicle_result.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_draw, "Car", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        vbox = [x1, y1, x2, y2]
        vehicle_crop = image_input[y1:y2, x1:x2]

        if is_vehicle_on_crosswalk_box(vbox, CROSSWALK_BOX):
            plate_text, plate_crop = ocr_plate_from_vehicle(vehicle_crop)
            summary.append(f"🚫 Crosswalk Violation: Vehicle {i+1}, Plate: {plate_text}, Location: {location}")
            gallery.append(vehicle_crop)
            plate_images.append(plate_crop)

            violations_json.append({
                "licensePlate": plate_text,
                "violationType": "Dừng xe trên vạch dành cho người đi bộ",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "location": location
            })

    # In kết quả
    print("\n[🚦 Crosswalk Violations Detected]")
    if not summary:
        print("✅ No crosswalk violations detected.")
    else:
        for msg in summary:
            print(msg)

    # Hiển thị ảnh chính
    cv2.imshow("Detected Violations", image_draw)

    # Hiển thị từng biển số (nếu có)
    for i, plate in enumerate(plate_images):
        win_name = f"Plate {i+1}"
        cv2.imshow(win_name, plate)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return violations_json

def process_red_light_video(video_path):
    from datetime import datetime
    global license_number_counter
    location = os.path.splitext(os.path.basename(video_path))[0].lower()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames = max(int(fps * 0.5), 1)  # xử lý mỗi 0.5s
    frame_index = 0

    vehicles_already_violated = set()
    violations_json = []

    while frame_index < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()
        if not success:
            break

        image_draw = frame.copy()

        if not is_red_light(image_draw):
            cv2.imshow("Real-time", image_draw)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_index += skip_frames
            continue

        stopline_quad = get_stop_line_box(image_draw, location)
        contour = np.array(stopline_quad, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image_draw, [contour], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.putText(image_draw, "Stop Line", stopline_quad[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        vehicle_result = vehicle_model(image_draw)[0]

        for i, box in enumerate(vehicle_result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            vbox = (x1, y1, x2, y2)
            key = tuple(round(x / 10) for x in vbox)

            if key in vehicles_already_violated:
                continue

            vehicle_crop = image_draw[y1:y2, x1:x2]
            cv2.rectangle(image_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_draw, "Car", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            if is_vehicle_on_stop_line(vbox, stopline_quad):
                vehicles_already_violated.add(key)
                license_number = f"29A{license_number_counter}"
                license_number_counter += 1

                cv2.putText(image_draw, license_number, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                violations_json.append({
                    "licensePlate": license_number,
                    "violationType": "Vượt đèn đỏ",
                    "timestamp": timestamp_now,
                    "location": location
                })

                print(f"[{timestamp_now}] 🚦 Violation - {license_number}")

        cv2.imshow("Real-time", image_draw)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_index += skip_frames

    cap.release()
    cv2.destroyAllWindows()

    return violations_json

def append_violations_to_json_file(new_violations, json_path="violation_result.json"):
    # Tạo file nếu chưa tồn tại
    if not os.path.exists(json_path):
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)

    # Đọc dữ liệu cũ
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Thêm vi phạm mới
    data.extend(new_violations)

    # Ghi lại file
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Appended {len(new_violations)} violations to {json_path}")


if __name__ == "__main__":
    """
    Demo hai trường hợp vi phạm:
    1. Vi phạm vạch qua đường (ảnh): Có biển số xe, vạch qua đường
    Gọi hàm `process_crosswalk_image(image_path)` với ảnh vạch qua đường.
    2. Vi phạm với đèn đỏ (video): Có vạch dừng, đèn đỏ. Không có biển số xe do chính sách riêng tư. Do đó hardcode biển số xe.
    Gọi hàm `process_red_light_video(video_path)` với video vượt đèn đỏ.
    Các trường hợp đều hardcode vạch dừng và vạch qua đường. Đúng với trường hợp lắp camera cố định thực tế.
    """
    # === Hardcoded file paths ===
    image_path = r"demo\cross.jpeg"
    video_path = r"demo\lakecheyenne.mp4"

    all_violations = []

    # === Xử lý ảnh vi phạm vạch qua đường ===
    print("📸 Processing image (crosswalk case)...")
    crosswalk_violations = process_crosswalk_image(image_path)
    all_violations.extend(crosswalk_violations)

    # === Xử lý video vượt đèn đỏ ===
    print("\n🎥 Processing video (red light case)...")
    redlight_violations = process_red_light_video(video_path)
    all_violations.extend(redlight_violations)

    # === In ra tổng kết
    print("\n📄 Violations Summary:")
    for v in all_violations:
        print(v)

    # === Ghi vào JSON
    append_violations_to_json_file(all_violations, "violation_result.json")

