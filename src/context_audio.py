import cv2
import speech_recognition as sr
import threading
import queue
from datetime import datetime, timedelta
from openai import OpenAI
import base64
import io
from PIL import Image
import time
import os
from dotenv import load_dotenv
import pygame
import tempfile
import sounddevice as sd
import numpy as np
from main import recognize_signs

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

text_queue = queue.Queue()
vision_queue = queue.Queue()
last_text = ""
last_vision_text = ""
text_timestamp = datetime.now()
vision_timestamp = datetime.now()
TEXT_DISPLAY_DURATION = timedelta(seconds=10)
voice_triggered = False
sign_text = ""
process_sign_language = False

# Initialize pygame mixer for audio playback
pygame.mixer.init()

def encode_image_to_base64(frame):
    # Convert CV2 frame to PIL Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def analyze_image(frame):
    try:
        base64_image = encode_image_to_base64(frame)
        
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "As a friendly companion, describe what's immediately around us in 1-2 short sentences. Focus on the most important things: any nearby obstacles, people, or immediate safety concerns a visually impaired person should know about. Don't say more than 3 sentences."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low"
                            }
                        }
                    ],
                }
            ],
            max_tokens=100  # Reduced token limit to ensure shorter responses
        )
        
        vision_queue.put(response.choices[0].message.content)
    except Exception as e:
        print(f"Error in image analysis: {e}")


def audio_processing():
    recognizer = sr.Recognizer()
    
    mic_index = 0
    
    while True:
        try:
            with sr.Microphone(device_index=mic_index) as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)
                try:
                    text = recognizer.recognize_google(audio)
                    # Check for trigger phrases
                    trigger_phrases = ["describe surround", "describe surrounding", "describe surroundings", "what's around", "whats around", "describe the room"]

                    if "sign language" in text.lower():
                        text_queue.put("Recognizing sign language...")
                        global sign_text, process_sign_language
                        process_sign_language = True  # New flag to trigger sign language processing
                    elif any(phrase in text.lower() for phrase in trigger_phrases):
                        text_queue.put("Analyzing surroundings...")
                        global voice_triggered
                        voice_triggered = True
                    else:
                        text_queue.put(f"Speech: {text}")
                except sr.UnknownValueError:
                    pass
                except sr.RequestError:
                    text_queue.put("Speech recognition service unavailable")
        except Exception as e:
            print(f"Error in audio processing: {e}")

# Start audio processing thread
audio_thread = threading.Thread(target=audio_processing, daemon=True)
audio_thread.start()

# Initialize video capture
cap = cv2.VideoCapture(2)  # Try index 2 first

"""if not cap.isOpened():
    print("Failed to open camera 2, trying camera 1...")
    cap = cv2.VideoCapture(1)  # Try index 1 as fallback"""

if not cap.isOpened():
    print("Failed to open OBS camera, falling back to default camera...")
    cap = cv2.VideoCapture(0)  # Fallback to default camera

print(f"Successfully opened camera with index: {cap.get(cv2.CAP_PROP_POS_FRAMES)}")

# Add this function to handle text-to-speech conversion
def text_to_speech(text):
    try:
        # Stop any currently playing audio
        pygame.mixer.music.stop()
        
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text,
            response_format="mp3",
        )
        
        # Save the audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name
        
        # Play the audio using pygame
        pygame.mixer.music.load(temp_file_path)
        pygame.mixer.music.play()
        
        # Wait for the audio to finish
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
            
        # Clean up the temporary file
        os.unlink(temp_file_path)
            
    except Exception as e:
        print(f"Error in text-to-speech conversion: {e}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Update transcribed text if available
    try:
        while not text_queue.empty():
            last_text = text_queue.get_nowait()
            text_timestamp = datetime.now()
    except queue.Empty:
        pass
    if process_sign_language:
        print("yute2")
        sign_text = recognize_signs(0)
        print(sign_text)
        process_sign_language = False  # Reset the flag

    # Only analyze image when voice triggered
    if voice_triggered:
        threading.Thread(target=analyze_image, args=(frame.copy(),), daemon=True).start()
        voice_triggered = False  # Reset the trigger

    # Update vision analysis text if available
    try:
        while not vision_queue.empty():
            last_vision_text = vision_queue.get_nowait()
            vision_timestamp = datetime.now()
            threading.Thread(target=text_to_speech, args=(last_vision_text,), daemon=True).start()
    except queue.Empty:
        pass

    # Draw existing landmarks
    # ... existing landmark drawing code ...

    # Add text overlays
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Create overlay for better text visibility
    overlay = frame.copy()
    
    # Speech transcription overlay (bottom)
    if datetime.now() - text_timestamp < TEXT_DISPLAY_DURATION:
        cv2.rectangle(overlay, (10, frame_height-60), 
                     (frame_width-10, frame_height-10), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, last_text, 
                    (20, frame_height-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    (255, 255, 255), 2)

    # Vision analysis overlay (top)
    if datetime.now() - vision_timestamp < TEXT_DISPLAY_DURATION:
        # Split text into multiple lines if too long
        words = last_vision_text.split()
        lines = []
        current_line = "Vision: "
        for word in words:
            if len(current_line + word) < 50:  # Adjust number based on your needs
                current_line += word + " "
            else:
                lines.append(current_line)
                current_line = word + " "
        lines.append(current_line)

        # Draw background for vision text
        cv2.rectangle(overlay, (10, 10), 
                     (frame_width-10, 20 + 30*len(lines)), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Draw each line of text
        for i, line in enumerate(lines):
            cv2.putText(frame, line, 
                        (20, 40 + 30*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                        (255, 255, 255), 2)

    cv2.imshow('MediaPipe Holistic with Speech and Vision', frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()