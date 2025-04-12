# Real-Time Cheating Detection Using Eye Gaze

This project is a webcam-based cheating detection system built in Python. It uses face visibility, eye movement, and blink patterns to monitor student attention during online exams.
Firstly,for setting up the environment ,I personally used conda
conda create -n cheatdetect python=3.9
conda activate cheatdetect
Then I installed the required dependencies by using pip install opencv-python mediapipe numpy
The main dependencies are:
OpenCV
Mediapipe
NumPy
How to Run
Ensure your webcam is connected and working.

Run the main script:

bash
Copy code
python main.py
A webcam window will open showing your gaze direction, blink count, and attention score.

Press q to quit the stream.

Logs of suspicious behavior are saved to alert_log.txt.
Detection Strategy
This system combines multiple visual cues to monitor potential cheating behavior:

1. Face Visibility Detection
If the face is not visible for more than 3 seconds, it logs an alert and reduces the attention score.

2. Gaze Direction Detection
If the user looks away (left/right) for more than 3 seconds, it logs a distraction alert.

Rapid side glances (6+ within 10 seconds) are flagged as suspicious.

3. Blink Detection
Uses Eye Aspect Ratio (EAR) to detect blinks.

Sudden blinking patterns can be useful in future enhancements (e.g., detecting drowsiness or signaling).

4. Attention Scoring System
Starts at 100 and decreases based on alerts.

Helpful in summarizing the test-taker's overall attentiveness.

