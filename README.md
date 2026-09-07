# DIPT-WORKSHOP-2
# Name: Yokesh H
# reg No:212224230312
# Object Detection Using Laptop Camera

## Aim

To access the **laptop camera**, capture an image, and detect objects using **YOLOv8**.

## Requirements

* Anaconda
* Jupyter Notebook
* Laptop Camera

## Steps

1. Open **Jupyter Notebook** using Anaconda.
2. Access your **laptop camera** using OpenCV.
3. Wait for **5 seconds** and capture an image.
4. Display the captured image.
5. Apply **YOLOv8 object detection**.
6. Display the detected image with **bounding boxes and labels**.

## Algorithm

```text
Laptop Camera
      ↓
Capture Image
      ↓
Display Image
      ↓
YOLOv8 Detection
      ↓
Display Detected Objects
```

## Student Task

* Capture your own image using the laptop camera.
* Detect the objects present in the image.
* Display the final output.
* Perform the experiment with **3 different scenes**.

## Submission

* Jupyter Notebook (`.ipynb`)
* Original captured images
* YOLOv8 output images
* Screenshot of the final result

## GitHub Reference

https://github.com/ultralytics/ultralytics

**Platform:** Anaconda + Jupyter Notebook only.

## Program
```
import cv2

# Open the web camera
cap = cv2.VideoCapture(0)

# Background subtractor for foreground-object detection
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=True
)

while True:

    # Read frame from webcam
    ret, frame = cap.read()

    if not ret:
        print("Cannot read frame from web camera")
        break

    # Create foreground mask
    mask = bg_subtractor.apply(frame)

    # Remove small noise
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_DILATE,
        kernel
    )

    # Find contours of detected objects
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Process each detected contour
    for contour in contours:

        # Calculate contour area
        area = cv2.contourArea(contour)

        # Ignore very small regions
        if area > 2500:

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Display detection text
            cv2.putText(
                frame,
                "Object Detected",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # Display camera frame
    cv2.imshow(
        "Workshop 2 - Object Detection",
        frame
    )

    # Press 'q' to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
```
## Output
<img width="794" height="621" alt="image" src="https://github.com/user-attachments/assets/7307cb87-7f60-4eaa-9fb8-5e4bb1981b3c" />
