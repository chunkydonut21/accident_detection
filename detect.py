import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("accident_detection_model.keras")

# Define the video source (0 for webcam or path to a video file)
video_path = './data/brutal_car_crash.mp4'
cap = cv2.VideoCapture(video_path)

# Define image size for resizing to match model input
img_size = (250, 250)

# Process video frames in real-time
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and normalize the frame to match model input size
    img = cv2.resize(frame, img_size)
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict using the model
    prediction = model.predict(img)
    label = "Accident" if prediction[0][0] > 0.5 else "No Accident"
    confidence = prediction[0][0] if label == "Accident" else 1 - prediction[0][0]

    # Display prediction and confidence on the frame
    cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
    cv2.putText(frame, f"{label} ({confidence:.2f})", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 0, 255) if label == "Accident" else (0, 255, 0), 2)

    # Show the frame with predictions
    cv2.imshow("Accident Detection", frame)

    # Press 'q' to exit the video display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
