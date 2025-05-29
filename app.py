import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# Load YOLOv8s model
model = YOLO('yolov8s.pt')

# Load FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False)

# Load Gender Classifier
gender_net = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")
GENDER_LIST = ["Male", "Female"]

# Start webcam
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w, _ = frame.shape

    # YOLOv8 person detection
    results = model(frame)[0]
    person_count = 0

    for result in results.boxes:
        if int(result.cls[0]) == 0:  # Class 0 is 'person'
            person_count += 1
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            person_crop = frame[y1:y2, x1:x2]

            rgb_person = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            face_result = face_mesh.process(rgb_person)

            if face_result.multi_face_landmarks:
                for landmarks in face_result.multi_face_landmarks:
                    ph, pw, _ = person_crop.shape
                    xs = [int(pt.x * pw) for pt in landmarks.landmark]
                    ys = [int(pt.y * ph) for pt in landmarks.landmark]
                    x_min, x_max = max(min(xs)-10, 0), min(max(xs)+10, pw)
                    y_min, y_max = max(min(ys)-10, 0), min(max(ys)+10, ph)
                    face_crop = person_crop[y_min:y_max, x_min:x_max]

                    # Gender classification
                    if face_crop.size == 0:
                        continue
                    blob = cv2.dnn.blobFromImage(face_crop, 1, (227, 227), (78.426, 87.768, 114.895), swapRB=False)
                    gender_net.setInput(blob)
                    gender_preds = gender_net.forward()
                    gender = GENDER_LIST[gender_preds[0].argmax()]

                    # Draw bounding box & gender
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, gender, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    # Show people count
    cv2.putText(frame, f"People Count: {person_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

    # Show the frame
    cv2.imshow("YOLOv8 + FaceMesh Gender Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()