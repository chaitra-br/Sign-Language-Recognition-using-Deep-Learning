import os
import streamlit as st
import cv2
import numpy as np
import math
from collections import deque
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Streamlit UI setup
st.title("Real-time Sign Language to Sentence Conversion")

# Webcam control and stream window setup
run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])

# Initialize session state parameters
if "sentence" not in st.session_state:
    st.session_state.sentence = deque(maxlen=10)

if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0

predicted_word = ""
capture_interval = 20
max_sentence_length = 50

# Load the classifier and labels
detector = HandDetector(maxHands=1)
model_path = r"C:\Users\Lenovo\OneDrive\Desktop\project2024\myenv\Model\keras_model.h5"
labels_path = r"C:\Users\Lenovo\OneDrive\Desktop\project2024\myenv\Model\labels.txt"
classifier = Classifier(model_path, labels_path)

# Set parameters for image preprocessing
offset = 20
imgSize = 300
labels = ["Apple", "Boy", "Camel", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Main loop for capturing and processing frames
if run:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.write("Unable to access the camera.")
    else:
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
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

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
                
                predicted_word = labels[index]

                if st.session_state.frame_count % capture_interval == 0:
                    if len(st.session_state.sentence) == 0 or st.session_state.sentence[-1] != predicted_word:
                        st.session_state.sentence.append(predicted_word)

                    if len(st.session_state.sentence) > max_sentence_length:
                        st.session_state.sentence.popleft()

                # Overlay predicted label on the output frame
                cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, predicted_word, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

                st.image(imgCrop, caption="Cropped Image", channels="BGR")
                st.image(imgWhite, caption="Image for Prediction", channels="BGR")

            FRAME_WINDOW.image(imgOutput, channels="BGR")
            st.write(f"Detected Gesture: {predicted_word}")
            st.session_state.frame_count += 1

            if st.session_state.sentence:
                st.write("Detected Gestures: ", " ".join(st.session_state.sentence))
            else:
                st.write("Detected Gestures: None")

    cap.release()
    cv2.destroyAllWindows()
else:
    # Reset session state when webcam is off
    st.session_state.sentence.clear()
    st.session_state.frame_count = 0
