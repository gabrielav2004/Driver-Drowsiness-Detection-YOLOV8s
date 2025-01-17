import time
import wave
import pyaudio
from ultralytics import YOLO

# Function to play sound using pyaudio
def play_sound(file_path):
    # Open the .wav file
    with wave.open(file_path, 'rb') as wf:
        # Set up the audio stream
        p = pyaudio.PyAudio()

        # Open stream using the wave file's format
        stream = p.open(format=pyaudio.paInt16,
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        # Read data from the wave file and play it
        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

# Load the YOLO model
model = YOLO('YOLO_DDBest.pt')

# Variables to track detection time
detected_time = 0
alarm_played = False
class_to_detect = "Buon_Ngu"
alarm_duration = 5  # Time in seconds

# Start video stream
results = model(source=0, stream=True, conf=0.4)  # Stream results

for r in results:  # Iterate over the streamed results
    boxes = r.boxes  # Get bounding box outputs
    target_detected = False  # Reset detection flag for each frame

    if boxes is not None:
        for box in boxes:
            class_id = int(box.cls)  # Class ID from the detection
            class_name = model.names[class_id]  # Map class ID to class name
            if class_name == class_to_detect:
                target_detected = True
                break  # No need to check further if already detected

    # Handle timing for detection
    if target_detected:
        if detected_time == 0:  # Start timer
            detected_time = time.time()
        elif time.time() - detected_time > alarm_duration and not alarm_played:
            # Play alarm if detection duration exceeds threshold
            print("Alarm Triggered: 'Buon Ngu' detected for more than 5 seconds!")
            play_sound("red-alert-fx.wav")  # Path to the audio file
            alarm_played = True
    else:
        detected_time = 0  # Reset timer if target class not detected
        alarm_played = False  # Reset alarm status

    # Show the frame (already handled by YOLO)
