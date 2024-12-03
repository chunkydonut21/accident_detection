import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("accident_detection_model.keras")

# Define the video source (0 for webcam or path to a video file)
video_path = './data/raw_video.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get the video frame width, height, and FPS
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object to save the video
output_path = '/Users/shivammaheshwari/Downloads/output_labeled_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Codec for mp4 format
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

if not out.isOpened():
    print("Error: Could not open VideoWriter for output.")
    exit()


# Define image size for resizing to match model input
img_size = (250, 250)

# Process video frames in real-time
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize and preprocess the frame for prediction
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
    
    # Write the labeled frame to the output video file
    out.write(frame)

    # Show the frame with predictions (optional)
    cv2.imshow("Accident Detection", frame)

    # Press 'q' to exit the video display
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()  # Important to release the VideoWriter
cv2.destroyAllWindows()

print(f"Labeled video saved as {output_path}")
