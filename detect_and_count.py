import cv2
from ultralytics import YOLO

# Load model YOLOv8 pre-trained
model = YOLO("yolov8n.pt")  # bản nhỏ, chạy nhanh

# Đường dẫn video cần detect
video_path = "856376-hd_1920_1080_30fps.mp4"  # đổi thành video của bạn
cap = cv2.VideoCapture(video_path)

# Lấy thông tin video
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Tùy chọn: Lưu video kết quả
out = cv2.VideoWriter(
    "result_count.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    results = model(frame)

    # Lọc chỉ người
    person_count = 0
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        if label == "person":
            person_count += 1

    # Vẽ kết quả nhận dạng
    annotated_frame = results[0].plot()

    # Thêm số người lên video
    cv2.putText(
        annotated_frame,
        f"Person count: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        2
    )

    # Hiển thị video
    cv2.imshow("YOLOv8 Person Count", annotated_frame)

    # Lưu video kết quả
    out.write(annotated_frame)

    # Nhấn Q để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
