import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile

# Thiết lập giao diện
st.set_page_config(page_title="YOLOv8 Object Detection", layout="wide")
st.title("🎯 YOLOv8 - Chọn loại đối tượng để nhận dạng trong Video")

# Upload video
uploaded_video = st.file_uploader("Tải video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

if uploaded_video:
    # Lưu video tạm
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    # Load YOLO model
    model = YOLO("yolov8n.pt")

    # Lấy danh sách nhãn từ model
    labels_list = list(model.names.values())

    # Chọn loại đối tượng
    target_label = st.selectbox("Chọn loại đối tượng muốn nhận dạng", labels_list, index=0)

    # Mở video
    cap = cv2.VideoCapture(tfile.name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # File video kết quả
    output_path = "result_streamlit.mp4"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        results = model(frame)

        # Đếm số lượng đối tượng
        target_count = 0
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label == target_label:
                target_count += 1

        # Annotate frame
        annotated_frame = results[0].plot()
        cv2.putText(
            annotated_frame,
            f"{target_label} count: {target_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2
        )

        out.write(annotated_frame)
        stframe.image(annotated_frame, channels="BGR", use_column_width=True)

    cap.release()
    out.release()

    st.success("✅ Hoàn thành nhận dạng!")
    with open(output_path, "rb") as f:
        st.download_button("📥 Tải video kết quả", f, file_name="result_detect.mp4")
