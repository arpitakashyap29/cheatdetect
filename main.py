import cv2
import time
from gaze_detect import GazeDetector
from utility import AlertLogger
import numpy as np

# Define EAR calculation
def calculate_ear(eye):
    # eye: List of 6 (x, y) points of the eye
    A = np.linalg.norm(eye[1] - eye[5])  # Vertical distance between two sets of eye landmarks
    B = np.linalg.norm(eye[2] - eye[4])  # Vertical distance between two sets of eye landmarks
    C = np.linalg.norm(eye[0] - eye[3])  # Horizontal distance between the eye landmarks
    ear = (A + B) / (2.0 * C)  # Eye aspect ratio
    return ear

# Initialize modules
gaze = GazeDetector()
logger = AlertLogger("alert_log.txt")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("❌ Could not open webcam.")

print("✅ Webcam stream started. Press 'q' to quit.")

# Timing and control flags
face_not_visible_since = None
look_away_since = None
gaze_history = []

# Constants
LOOK_AWAY_THRESHOLD = 3     # seconds
FACE_NOT_VISIBLE_THRESHOLD = 3
GAZE_HISTORY_WINDOW = 10    # seconds for rapid glance detection
RAPID_GLANCE_COUNT = 6      # number of glances to trigger alert
BLINK_THRESHOLD = 0.25      # EAR threshold for blink detection
BLINK_CONSEC_FRAMES = 2     # frames for which EAR must be below threshold to register blink

blink_counter = 0
blink_total = 0
attention_score = 100  # Initial attention score (0-100)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame for gaze and other detections
    gaze_result = gaze.analyze_frame(frame)

    # 1. Check face visibility
    if not gaze_result.get("face_visible", False):
        if face_not_visible_since is None:
            face_not_visible_since = time.time()
        elif time.time() - face_not_visible_since > FACE_NOT_VISIBLE_THRESHOLD:
            logger.log(f" Face not visible for {FACE_NOT_VISIBLE_THRESHOLD} seconds")
            attention_score -= 5
            face_not_visible_since = time.time()  # reset
    else:
        face_not_visible_since = None

    # 2. Check gaze direction
    direction = gaze_result.get("gaze_direction", "unknown")
    if direction != "center":
        if look_away_since is None:
            look_away_since = time.time()
        elif time.time() - look_away_since > LOOK_AWAY_THRESHOLD:
            logger.log(f" User looking away from screen for {LOOK_AWAY_THRESHOLD} seconds")
            attention_score -= 5
            look_away_since = time.time()  # reset
    else:
        look_away_since = None

    # 3. Detect rapid side glances
    gaze_history.append((time.time(), direction))
    # Clean up old history
    gaze_history = [(t, d) for t, d in gaze_history if time.time() - t <= GAZE_HISTORY_WINDOW]
    side_glances = [d for _, d in gaze_history if d in ["left", "right"]]
    if len(side_glances) >= RAPID_GLANCE_COUNT:
        logger.log(f" Rapid left-right gaze detected {len(side_glances)} times in {GAZE_HISTORY_WINDOW} seconds")
        attention_score -= 3
        gaze_history = []  # reset to prevent duplicate logs

    # 4. Blink detection - Check for EAR to detect blink
    face_landmarks = gaze_result.get("face_landmarks", [])
    if face_landmarks:
        left_eye = [face_landmarks[i] for i in gaze.LEFT_EYE_IDX]
        right_eye = [face_landmarks[i] for i in gaze.RIGHT_EYE_IDX]

        # Calculate EAR for both eyes
        left_ear = calculate_ear(np.array(left_eye))
        right_ear = calculate_ear(np.array(right_eye))

        # Check if either eye is below the blink threshold
        if left_ear < BLINK_THRESHOLD or right_ear < BLINK_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= BLINK_CONSEC_FRAMES:
                blink_total += 1
                logger.log("Blink detected")
            blink_counter = 0

    # Update attention score display
    cv2.putText(frame, f"Gaze: {direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Blinks: {blink_total}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(frame, f"Attention: {attention_score}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.imshow("Gaze Tracking", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Webcam stream stopped.")
