import os
import streamlit as st
import cv2
import numpy as np
import math
from collections import deque
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
from gtts import gTTS
import tempfile
import playsound

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Streamlit UI setup
st.set_page_config(page_title="Sign Language Recognition", layout="wide")
st.title("🖐️ Real-time Sign Language to Sentence Conversion")

# Sidebar controls
st.sidebar.header("Control Panel")
run = st.sidebar.checkbox("Start Webcam", value=False)
st.sidebar.markdown("### Instructions:")
st.sidebar.markdown("1. Position your hand in front of the webcam.")
st.sidebar.markdown("2. Make sure your hand is clearly visible.")
st.sidebar.markdown("3. The app will recognize letters and form words.")

# Placeholder for webcam output
FRAME_WINDOW = st.image([])

# Initialize session state parameters
if "recognized_letters" not in st.session_state:
    st.session_state.recognized_letters = []  # Holds recognized letters
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "previous_letter" not in st.session_state:
    st.session_state.previous_letter = None  # Track the previously predicted letter

# Load the classifier and labels
detector = HandDetector(maxHands=1)
model_path = r"C:\Users\Lenovo\OneDrive\Desktop\project2024\myenv\Model\keras_model.h5"
labels_path = r"C:\Users\Lenovo\OneDrive\Desktop\project2024\myenv\Model\labels.txt"
classifier = Classifier(model_path, labels_path)

# Set parameters for image preprocessing
offset = 20
imgSize = 300
labels = ["Apple", "Ball", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]  # A to Z

# Sample dictionary of valid words
valid_words = {"HELLO", "WORLD", "SIGN", "LANGUAGE", "STREAMLIT", "DEEP", "LEARNING"}

# Function to convert text to speech
def speak_word(word):
    tts = gTTS(text=word, lang='en')
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        tts.save(f"{tmp.name}.mp3")
        playsound.playsound(f"{tmp.name}.mp3")

# Main loop for capturing and processing frames
if run:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.write("Unable to access the camera.")
    else:
        # Placeholder for the recognized letters output
        recognized_letters_placeholder = st.empty()  # Create a placeholder for recognized letters

        while run:
            success, img = cap.read()
            if not success:
                st.write("Unable to capture the frame.")
                break
            
            imgOutput = img.copy()
            hands, img = detector.findHands(img)

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

                # Adjust the crop coordinates to stay within the image bounds
                y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
                x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
                imgCrop = img[y1:y2, x1:x2]

                # Check if imgCrop is valid (not empty) before proceeding
                if imgCrop.size != 0:
                    # Resize image for prediction based on aspect ratio
                    aspectRatio = h / w
                    if aspectRatio > 1:
                        k = imgSize / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize
                        prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    else:
                        k = imgSize / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize
                        prediction, index = classifier.getPrediction(imgWhite, draw=False)

                    # Get the predicted letter
                    predicted_letter = labels[index]

                    if st.session_state.frame_count % 20 == 0:  # Capture interval
                        # Only add if the new letter is different from the last letter added
                        if predicted_letter != st.session_state.previous_letter:
                            st.session_state.recognized_letters.append(predicted_letter)
                            st.session_state.previous_letter = predicted_letter  # Update previous_letter

                    # Overlay predicted label on the output frame
                    cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, predicted_letter, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                    cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

            FRAME_WINDOW.image(imgOutput, channels="BGR")
            st.session_state.frame_count += 1

            # Form a word from recognized letters if it matches any valid word
            formed_word = ''.join(st.session_state.recognized_letters)
            if formed_word in valid_words:
                st.success(f"Recognized Word: {formed_word}")
                # Clear recognized letters after forming a valid word
                st.session_state.recognized_letters.clear()

                # Button to speak the recognized word
                if st.button("Speak Recognized Word"):
                    speak_word(formed_word)

            # Update the placeholder with the current recognized letters dynamically
            recognized_letters_placeholder.write("### Recognized Letters: " + " ".join(st.session_state.recognized_letters))

    cap.release()
    cv2.destroyAllWindows()
else:
    # Reset session state when webcam is off
    st.session_state.recognized_letters.clear()
    st.session_state.frame_count = 0
    st.session_state.previous_letter = None  # Reset previous letter
