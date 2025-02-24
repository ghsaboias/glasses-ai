import cv2
import numpy as np
from ultralytics import YOLO
from mss import mss
import time

# Load YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Set up screen capture
sct = mss()
monitor = {"top": 0, "left": 2780, "width": 320, "height": 320}  # Adjust as needed

# Frame skipping: process every nth frame for detection
frame_skip = 2  # Adjust this (1 = no skipping, 2 = every 2nd frame, etc.)
count = 0

# Variable to store the last annotated frame
last_annotated_frame = None

print("Starting livestream detection. Press 'q' to quit.")

prev_time = time.time()

while True:
    # Timing for capture
    capture_start = time.time()
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    capture_time = time.time() - capture_start

    # Run detection only every 'frame_skip' frames
    if count % frame_skip == 0:
        inference_start = time.time()
        results = model(frame, conf=0.4, device="mps")
        inference_time = time.time() - inference_start
        annotated_frame = results[0].plot()
        last_annotated_frame = annotated_frame  # Update the last annotated frame
    else:
        # Use the last annotated frame if available, otherwise raw frame
        annotated_frame = (
            last_annotated_frame if last_annotated_frame is not None else frame
        )

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Add FPS and timing info to the frame
    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        annotated_frame,
        f"Capture: {capture_time:.3f}s",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )
    cv2.putText(
        annotated_frame,
        f"Inference: {inference_time if 'inference_time' in locals() else 0:.3f}s",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )

    # Show the frame
    cv2.imshow("YOLOv8 Live Detection", annotated_frame)

    # Print timing logs
    print(
        f"Capture: {capture_time:.3f}s, Inference: {inference_time if 'inference_time' in locals() else 0:.3f}s, FPS: {fps:.1f}"
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    count += 1

cv2.destroyAllWindows()
