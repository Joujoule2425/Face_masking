import os
import cv2
import mediapipe as mp

# Create output directory
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

# Define process_img function
def process_img(img, face_detection):
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    if out.detections is not None:
        for detection in out.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # Blur the detected face region
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (30, 30))

    return img

# Simulate argparse with a simple class
class Args:
    mode = 'webcam'  # Change to 'image' or 'video' if needed
    filePath = None  # Add file path here if mode != webcam

args = Args()

# Initialize face detection model
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    if args.mode == "image":
        # Load image
        img = cv2.imread(args.filePath)

        img = process_img(img, face_detection)

        # Save processed image
        cv2.imwrite(os.path.join(output_dir, 'output.png'), img)
        print("Image processed and saved to ./output/output.png")

        # Optionally show image
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif args.mode == "video":
        cap = cv2.VideoCapture(args.filePath)
        ret, frame = cap.read()

        output_video = cv2.VideoWriter(
            os.path.join(output_dir, 'output.mp4'),
            cv2.VideoWriter_fourcc(*'MP4V'),
            25,
            (frame.shape[1], frame.shape[0])
        )

        while ret:
            frame = process_img(frame, face_detection)
            output_video.write(frame)
            ret, frame = cap.read()

        cap.release()
        output_video.release()
        print("Video processed and saved to ./output/output.mp4")

    elif args.mode == "webcam":
        cap = cv2.VideoCapture(0)  # Try 1 or 2 if 0 doesn't work
        if not cap.isOpened():
            print("Webcam not accessible. Try a different index.")
        else:
            print("Press 'q' to quit.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = process_img(frame, face_detection)
            cv2.imshow('Webcam - Press q to quit', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()