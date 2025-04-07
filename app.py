import streamlit as st
import cv2
from PIL import Image
import numpy as np

# Constants
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# Load pre-trained models
FACE_PROTO = "opencv_face_detector.pbtxt"
FACE_MODEL = "opencv_face_detector_uint8.pb"
AGE_PROTO = "age_deploy.prototxt"
AGE_MODEL = "age_net.caffemodel"
GENDER_PROTO = "gender_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"

face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

# Functions
def highlight_face(net, frame, conf_threshold=0.7):
    """Detect and highlight faces in the frame."""
    frame_copy = frame.copy()
    frame_height, frame_width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_copy, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            face_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 150)), 8)
    return frame_copy, face_boxes


def process_frame(frame, face_net, age_net, gender_net, model_mean_values, age_list, gender_list):
    """Process a frame for face, age, and gender detection."""
    result_img, face_boxes = highlight_face(face_net, frame)

    if not face_boxes:
        return result_img, "No face detected!"

    predictions = []
    for face_box in face_boxes:
        face = frame[max(0, face_box[1] - 20): min(face_box[3] + 20, frame.shape[0] - 1),
                     max(0, face_box[0] - 20): min(face_box[2] + 20, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), model_mean_values, swapRB=False)

        # Predict gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]

        # Predict age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]

        label = f"{gender}, {age}"
        predictions.append(label)
        cv2.putText(result_img, label, (face_box[0], face_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    return result_img, predictions


# Streamlit UI
st.title("Age and Gender Detection")
st.sidebar.header("Options")

# Sidebar options
use_webcam = st.sidebar.checkbox("Use Webcam")
confidence_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.5, max_value=1.0, value=0.7, step=0.1)
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    frame = np.array(image)
    result_img, predictions = process_frame(frame, face_net, age_net, gender_net, MODEL_MEAN_VALUES, AGE_LIST, GENDER_LIST)
    st.image(result_img, caption="Processed Image", use_column_width=True)
    if isinstance(predictions, list):
        st.success(f"Predictions: {', '.join(predictions)}")
    else:
        st.warning(predictions)

elif use_webcam:
    st.warning("Starting webcam...")
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Unable to access the webcam.")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame, predictions = process_frame(frame_rgb, face_net, age_net, gender_net, MODEL_MEAN_VALUES, AGE_LIST, GENDER_LIST)
        frame_placeholder.image(processed_frame, use_column_width=True)
    cap.release()
    cv2.destroyAllWindows()

else:
    st.info("Upload an image or enable the webcam to start detecting!")
