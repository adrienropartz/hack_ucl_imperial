import cv2
import mediapipe as mp

# Initialize MediaPipe Holistic.
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# Initialize MediaPipe drawing utility.
mp_drawing = mp.solutions.drawing_utils

# Open a webcam feed.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR frame to RGB.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and detect landmarks.
    results = holistic.process(image)

    # Draw landmarks with distinct colors and styles
    # Pose landmarks in green
    mp_drawing.draw_landmarks(
        frame, 
        results.pose_landmarks, 
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2)
    )
    
    # Face landmarks in red
    mp_drawing.draw_landmarks(
        frame, 
        results.face_landmarks, 
        mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(0, 0, 200), thickness=1)
    )
    
    # Left hand landmarks in blue
    mp_drawing.draw_landmarks(
        frame, 
        results.left_hand_landmarks, 
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(200, 0, 0), thickness=2)
    )
    
    # Right hand landmarks in yellow
    mp_drawing.draw_landmarks(
        frame, 
        results.right_hand_landmarks, 
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0, 200, 200), thickness=2)
    )

    # Display the frame.
    cv2.imshow('MediaPipe Holistic', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()