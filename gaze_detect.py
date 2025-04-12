import cv2
import mediapipe as mp
import numpy as np

class GazeDetector:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Needed for iris tracking
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.LEFT_EYE_IDX = [33, 133]     # [left corner, right corner]
        self.RIGHT_EYE_IDX = [362, 263]   # [left corner, right corner]
        self.LEFT_IRIS = [468]            # Center of left iris
        self.RIGHT_IRIS = [473]           # Center of right iris

    def _normalized_to_pixel_coords(self, norm_landmark, image_width, image_height):
        return int(norm_landmark.x * image_width), int(norm_landmark.y * image_height)

    def analyze_frame(self, frame):
        results = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        h, w = frame.shape[:2]

        output = {
            "face_visible": False,
            "gaze_direction": "unknown",
            "blink_detected": False
        }

        if results.multi_face_landmarks:
            output["face_visible"] = True
            face_landmarks = results.multi_face_landmarks[0].landmark

            # Get landmarks in pixel coords
            left_eye = [self._normalized_to_pixel_coords(face_landmarks[i], w, h) for i in self.LEFT_EYE_IDX]
            right_eye = [self._normalized_to_pixel_coords(face_landmarks[i], w, h) for i in self.RIGHT_EYE_IDX]
            left_iris = self._normalized_to_pixel_coords(face_landmarks[self.LEFT_IRIS[0]], w, h)
            right_iris = self._normalized_to_pixel_coords(face_landmarks[self.RIGHT_IRIS[0]], w, h)

            # Debugging: Display landmark positions
            print(f"Left Eye: {left_eye}")
            print(f"Right Eye: {right_eye}")
            print(f"Left Iris: {left_iris}")
            print(f"Right Iris: {right_iris}")

            # Gaze direction logic (per eye)
            left_gaze = self._get_eye_gaze_direction(left_eye, left_iris)
            right_gaze = self._get_eye_gaze_direction(right_eye, right_iris)

            # Combine both eyes
            if left_gaze == right_gaze:
                output["gaze_direction"] = left_gaze
            else:
                output["gaze_direction"] = "moving"

        return output

    def _get_eye_gaze_direction(self, eye_coords, iris_center):
        left_corner, right_corner = eye_coords
        eye_width = right_corner[0] - left_corner[0]
        iris_x = iris_center[0]

        if eye_width == 0:
            return "unknown"

        ratio = (iris_x - left_corner[0]) / eye_width

        if ratio < 0.35:
            return "left"
        elif ratio > 0.65:
            return "right"
        else:
            return "center"

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    gaze = GazeDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = gaze.analyze_frame(frame)
        print(f"Face Visible: {result['face_visible']}")
        print(f"Gaze Direction: {result['gaze_direction']}")

        cv2.putText(frame, f"Gaze: {result['gaze_direction']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Eye Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
