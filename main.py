#THIS IS THE MAIN FILE TO RUN IN PRESENTATION.
import os
import json
import streamlit as st
import cv2
import numpy as np
import math
from itertools import permutations
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

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
st.sidebar.markdown("3. Use the 'Form Sentence' and 'Clear' buttons as needed.")

# Placeholder for webcam output
FRAME_WINDOW = st.image([])

# Initialize session state parameters
st.session_state.setdefault("recognized_words", [])
st.session_state.setdefault("frame_count", 0)
st.session_state.setdefault("previous_word", None)
st.session_state.setdefault("sentence", "")

# Load the classifier and labels
try:
    detector = HandDetector(maxHands=1)
    model_path = r"C:\Users\Lenovo\OneDrive\Desktop\project2024\myenv\Model2\keras_model.h5"
    labels_path = r"C:\Users\Lenovo\OneDrive\Desktop\project2024\myenv\Model2\labels.txt"
    classifier = Classifier(model_path, labels_path)
except Exception as e:
    st.error(f"Error loading the model or labels: {e}")

# Load sentence structures from the JSON file
def load_sentence_data():
    try:
        with open('sentences.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Sentence JSON file not found.")
        return {}

sentences_data = load_sentence_data()

# Set parameters for image preprocessing
offset = 20
imgSize = 300
labels = [
    "Brother", "Call me", "Dislike", "Drink", "Fine", "Food", "Go", "GoodLuck", "Hello", "I am", 
    "I Hate You", "I Love You", "Look", "Mom", "Okay", "Perfect", "Please", "Small", 
    "Sorry", "Stop", "Tired", "Uneasy", "Wait", "Washroom", "Water", "What", 
    "Why", "Yes", "You"
]

# Function to generate sentences from recognized words based on the JSON data
def generate_sentences_from_json(words, sentence_data):
    sentences = []
    for word in words:
        for other_word in words:
            if word != other_word and word in sentence_data and other_word in sentence_data[word]:
                sentences.extend(sentence_data[word][other_word])
    return list(set(sentences))  # Remove duplicates

# Function to generate the most likely sentence (the sentence with the highest number of matches)
def get_most_likely_sentence(words, sentence_data):
    sentences = generate_sentences_from_json(words, sentence_data)
    if sentences:
        # Rank sentences by length (more words = higher probability of occurrence)
        most_likely_sentence = max(sentences, key=lambda x: len(x.split()))
        return most_likely_sentence
    return "No sentence formed."

# Main loop for capturing and processing frames
if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access the camera.")
    else:
        sentence_placeholder = st.empty()
        final_sentence_placeholder = st.empty()
        recognized_words_placeholder = st.empty()
        form_sentence_button = st.button("Form Sentence")
        clear_sentence_button = st.button("Clear Sentence")

        while run:
            success, img = cap.read()
            if not success:
                st.error("Unable to capture the frame.")
                break

            imgOutput = img.copy()
            hands, img = detector.findHands(img)

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

                y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
                x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
                imgCrop = img[y1:y2, x1:x2]

                if imgCrop.size != 0:
                    aspectRatio = h / w
                    if aspectRatio > 1:
                        k = imgSize / h
                        wCal = math.ceil(k * w)
                        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                        wGap = math.ceil((imgSize - wCal) / 2)
                        imgWhite[:, wGap:wCal + wGap] = imgResize
                    else:
                        k = imgSize / w
                        hCal = math.ceil(k * h)
                        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                        hGap = math.ceil((imgSize - hCal) / 2)
                        imgWhite[hGap:hCal + hGap, :] = imgResize

                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    predicted_word = labels[index]

                    if st.session_state.frame_count % 20 == 0:
                        if predicted_word != st.session_state.previous_word:
                            st.session_state.recognized_words.append(predicted_word)
                            st.session_state.previous_word = predicted_word

                    cv2.rectangle(
                        imgOutput, (x - offset, y - offset - 50), 
                        (x - offset + 90, y - offset), (255, 0, 255), cv2.FILLED
                    )
                    cv2.putText(
                        imgOutput, predicted_word, (x, y - 26), 
                        cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2
                    )
                    cv2.rectangle(
                        imgOutput, (x - offset, y - offset), 
                        (x + w + offset, y + h + offset), (255, 0, 255), 4
                    )

            FRAME_WINDOW.image(imgOutput, channels="BGR")
            recognized_words_placeholder.write("### Current Words: " + ", ".join(st.session_state.recognized_words))
            st.session_state.frame_count += 1

            if form_sentence_button:
                # Get the most likely sentence based on the recognized words
                most_likely_sentence = get_most_likely_sentence(st.session_state.recognized_words, sentences_data)
                st.session_state.sentence = most_likely_sentence
                final_sentence_placeholder.write("### Most Likely Sentence: \n" + st.session_state.sentence)

            if clear_sentence_button:
                st.session_state.recognized_words.clear()
                st.session_state.sentence = ""
                st.session_state.previous_word = None
                recognized_words_placeholder.write("### Current Words: ")
                final_sentence_placeholder.write("### Most Likely Sentence: ")

        cap.release()
        cv2.destroyAllWindows()
else:
    st.session_state.recognized_words.clear()
    st.session_state.frame_count = 0
    st.session_state.previous_word = None
    st.session_state.sentence = ""
