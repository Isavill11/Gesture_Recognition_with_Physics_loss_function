"""
ASL Alphabet:
      https://www.kaggle.com/datasets/grassknoted/asl-alphabet

"""

import os
import csv
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── Configuration ────────────────────────────────────────────────────────────

FRAME_RATE  = 1          # seconds between saved samples
DATASET_SIZE = 100       # images/landmark-rows per class
DATA_DIR     = 'asl_alphabet_test'

HAND_CLASSES = [i.removesuffix('.jpg') for i in os.listdir('asl_alphabet_test')]


print(f"hand classes: {HAND_CLASSES}")

SAVE_MODE = 'landmarks'


mp_draw     = None  # Drawing utils not available in new API
mp_styles   = None


def landmark_row(hand_landmarks, label: str) -> list:
    row = []
    for lm in hand_landmarks.landmark:
        row.extend([lm.x, lm.y])   # z excluded (noisy on webcam)
    row.append(label)
    return row


def csv_header() -> list:
    header = []
    for i in range(21):
        header += [f'x{i}', f'y{i}']
    header.append('gesture_class')
    return header


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    for cls in HAND_CLASSES:
        if SAVE_MODE == 'images':
            os.makedirs(os.path.join(DATA_DIR, cls), exist_ok=True)


def count_existing(class_name: str) -> int:
    if SAVE_MODE == 'images':
        d = os.path.join(DATA_DIR, class_name)
        return len([f for f in os.listdir(d) if f.endswith('.jpg')])
    else:
        csv_path = os.path.join(DATA_DIR, 'landmarks.csv')
        if not os.path.exists(csv_path):
            return 0
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            return sum(1 for row in reader if row and row[-1] == class_name)



def main():
    ensure_dirs()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check that it is connected.")

    # Prepare CSV file (landmarks mode)
    csv_path = os.path.join(DATA_DIR, 'landmarks.csv')
    csv_file, csv_writer = None, None
    if SAVE_MODE == 'landmarks':
        write_header = not os.path.exists(csv_path)
        csv_file   = open(csv_path, 'a', newline='')
        csv_writer = csv.writer(csv_file)
        if write_header:
            csv_writer.writerow(csv_header())

    with vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='./hand_landmarker.task'),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
    ) as landmarker:

        for class_name in HAND_CLASSES:
            print(f"\n── Class: {class_name} ──")
            start_index = count_existing(class_name)
            print(f"   Existing samples: {start_index} / {DATASET_SIZE}")

            if start_index >= DATASET_SIZE:
                print("   Already complete, skipping.")
                continue

            print("   Position your hand and press [Q] to begin collecting.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f'Class: {class_name}  |  Press Q to start',
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 80), 2)
                cv2.imshow('Data Collection', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            counter   = start_index
            last_save = 0.0

            while counter < DATASET_SIZE:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_image)

                display = frame.copy()
                saved_this_frame = False

                if result.hand_landmarks:
                    for hand_lms in result.hand_landmarks:
                        # Drawing removed as drawing utils not available in new API
                        # mp_draw.draw_landmarks(
                        #     display, hand_lms,
                        #     mp.solutions.hands.HAND_CONNECTIONS,
                        #     mp_styles.get_default_hand_landmarks_style(),
                        #     mp_styles.get_default_hand_connections_style()
                        # )

                        now = time.time()
                        if now - last_save >= FRAME_RATE:
                            if SAVE_MODE == 'landmarks':
                                row = landmark_row(hand_lms, class_name)
                                csv_writer.writerow(row)
                                csv_file.flush()
                            else:  # images
                                h, w, _ = frame.shape
                                xs = [int(lm.x * w) for lm in hand_lms.landmark]
                                ys = [int(lm.y * h) for lm in hand_lms.landmark]
                                pad = 20
                                x1 = max(0, min(xs) - pad)
                                y1 = max(0, min(ys) - pad)
                                x2 = min(w, max(xs) + pad)
                                y2 = min(h, max(ys) + pad)
                                crop = frame[y1:y2, x1:x2]
                                img_path = os.path.join(DATA_DIR, class_name, f'{counter}.jpg')
                                cv2.imwrite(img_path, crop)

                            counter   += 1
                            last_save  = now
                            saved_this_frame = True
                            print(f'   Saved {counter}/{DATASET_SIZE}')

                status_color = (0, 200, 255) if saved_this_frame else (200, 200, 200)
                cv2.putText(display, f'{class_name}  {counter}/{DATASET_SIZE}',
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)
                cv2.putText(display, 'Press [F] to quit early',
                            (20, display.shape[0] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)
                cv2.imshow('Data Collection', display)

                key = cv2.waitKey(25) & 0xFF
                if key == ord('f'):
                    print("Early quit.")
                    cap.release()
                    if csv_file:
                        csv_file.close()
                    cv2.destroyAllWindows()
                    return

            print(f"   ✓ Completed {DATASET_SIZE} samples for '{class_name}'.")

    cap.release()
    if csv_file:
        csv_file.close()
    cv2.destroyAllWindows()
    print("\nAll classes collected. Data saved to:", DATA_DIR)


if __name__ == '__main__':
    main()
