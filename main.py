"""
main.py
Real-time hand gesture recognition using a standard webcam + MediaPipe Hands.

Replaces the original Kinect / YOLO pipeline.
Object detection has been removed; this file handles gesture recognition only.

Requirements:
    pip install opencv-python mediapipe joblib numpy
    pip install keras tensorflow   (or your existing TF/Keras install)
"""

import os
import time
import cv2
import numpy as np
import joblib
import mediapipe as mp
from keras._tf_keras.keras.models import load_model

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME      = 'gesture_model_v1'          # must match what you trained
MODELS_DIR      = 'Trained-Models'
GESTURE_HOLD_TIME   = 3.0   # seconds a gesture must be held to "trigger"
COOLDOWN_DURATION   = 5.0   # seconds before the same gesture can trigger again

# ── Load model + preprocessors ────────────────────────────────────────────────

model_path   = os.path.join(MODELS_DIR, f'{MODEL_NAME}.keras')
scaler_path  = os.path.join(MODELS_DIR, f'{MODEL_NAME}_scaler.pkl')
encoder_path = os.path.join(MODELS_DIR, f'{MODEL_NAME}_encoder.pkl')

gesture_model   = load_model(model_path)
scaler          = joblib.load(scaler_path)
label_encoder   = joblib.load(encoder_path)

print("Loaded model:", model_path)
print("Classes:", label_encoder.classes_)

# ── MediaPipe setup ───────────────────────────────────────────────────────────

mp_hands  = mp.solutions.hands
mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_landmarks(hand_landmarks) -> np.ndarray:
    """Return a (1, 42) array of flattened (x, y) landmark coords."""
    row = []
    for lm in hand_landmarks.landmark:
        row.extend([lm.x, lm.y])
    return np.array(row, dtype=np.float32).reshape(1, -1)


def predict_gesture(hand_landmarks) -> tuple[str, float]:
    """Return (class_name, confidence) for a detected hand."""
    features  = extract_landmarks(hand_landmarks)
    scaled    = scaler.transform(features)
    probs     = gesture_model.predict(scaled, verbose=0)[0]
    idx       = int(np.argmax(probs))
    return label_encoder.classes_[idx], float(probs[idx])


# ── State ─────────────────────────────────────────────────────────────────────

current_gesture  = None
gesture_timer    = None
last_trigger_time = 0.0

# ── Main loop ─────────────────────────────────────────────────────────────────

def on_gesture_triggered(gesture: str):
    """
    Called once when a gesture has been held for GESTURE_HOLD_TIME seconds.
    Replace / extend with your robotic arm commands or any other action.
    """
    print(f"[ACTION] Gesture '{gesture}' triggered!")
    # e.g. execute_matlab_commands()


def main():
    global current_gesture, gesture_timer, last_trigger_time

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed, retrying…")
                continue

            frame = cv2.flip(frame, 1)                         # mirror
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            detected_gesture = None
            confidence_val   = 0.0

            if result.multi_hand_landmarks:
                for hand_lms in result.multi_hand_landmarks:
                    # Draw skeleton overlay
                    mp_draw.draw_landmarks(
                        frame, hand_lms,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

                    gesture, conf = predict_gesture(hand_lms)
                    if conf >= 0.65:
                        detected_gesture = gesture
                        confidence_val   = conf

                        # Draw label near wrist (landmark 0)
                        h, w, _ = frame.shape
                        wrist = hand_lms.landmark[0]
                        lx, ly = int(wrist.x * w), int(wrist.y * h)
                        cv2.putText(frame, f'{gesture} {conf:.0%}',
                                    (lx - 10, ly - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 80), 2)

            # ── Gesture hold logic ────────────────────────────────────────────
            now = time.time()

            if detected_gesture:
                if detected_gesture == current_gesture:
                    held = now - gesture_timer if gesture_timer else 0
                    # Progress bar
                    bar_w = int((min(held, GESTURE_HOLD_TIME) / GESTURE_HOLD_TIME) * 200)
                    cv2.rectangle(frame, (20, frame.shape[0] - 30),
                                  (20 + bar_w, frame.shape[0] - 10), (0, 200, 255), -1)
                    cv2.rectangle(frame, (20, frame.shape[0] - 30),
                                  (220, frame.shape[0] - 10), (100, 100, 100), 1)

                    if held >= GESTURE_HOLD_TIME:
                        if now - last_trigger_time >= COOLDOWN_DURATION:
                            on_gesture_triggered(detected_gesture)
                            last_trigger_time = now
                        else:
                            remaining = COOLDOWN_DURATION - (now - last_trigger_time)
                            cv2.putText(frame, f'Cooldown {remaining:.1f}s',
                                        (20, frame.shape[0] - 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
                        gesture_timer   = None
                        current_gesture = None
                else:
                    current_gesture = detected_gesture
                    gesture_timer   = now
            else:
                current_gesture = None
                gesture_timer   = None

            # ── HUD ──────────────────────────────────────────────────────────
            status = current_gesture if current_gesture else "No hand detected"
            cv2.putText(frame, status, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, 'Press [Q] to quit', (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

            cv2.imshow('Gesture Recognition', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
