# Sign-Language-Recognition-using-Deep-Learning

Real-time hand sign to sentence conversion using computer vision and deep learning (Streamlit + OpenCV + TensorFlow)

---

## 📌 Project Overview

This project is designed to recognize Indian Sign Language gestures in real-time using a webcam. It leverages computer vision and a trained deep learning model to detect hand gestures, classify them, and convert them into meaningful English sentences interactively through a web app built using **Streamlit**.

---

## 🚀 Features

* 🖐️ Real-time hand gesture detection using webcam
* 🔤 Recognition of multiple sign language words
* ✍️ Sentence formation using selected gestures
* 🧠 Trained deep learning model with `keras_model.h5`
* 🎯 User-friendly interface with live feedback
* 🗂️ Sentence templates loaded from `sentences.json`

---

## 🧠 Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* Streamlit
* cvzone (Hand detection and classification)
* NumPy, Math
* JSON

---

## 🗂️ Folder Structure

```
Sign Language Recognition/
├── app.py                 # Streamlit app (alternate version)
├── main.py                # Main Streamlit application
├── back.py                # Backend functions for cleaning
├── datacollection.py      # Script for dataset collection
├── myenv/                 # Local virtual environment (not uploaded)
├── sentences.json         # Predefined sentence structures
```

---

## 🛠️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/sign-language-recognition.git
cd sign-language-recognition
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv env
source env/bin/activate  # For Windows: env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, you can use:

```bash
pip install streamlit opencv-python cvzone tensorflow numpy
```

### 4. Run the App

```bash
streamlit run main.py
```

### ✅ Make Sure:

* Your webcam is enabled
* `keras_model.h5` and `labels.txt` are correctly placed in the model directory

---

## 📄 License

This project is under the **MIT License** – feel free to use and modify.

---

## 🔗 Connect with Me

**📧 Email:** [chaitrabr03@gmail.com](mailto:chaitrabr03@gmail.com)

**🔗 LinkedIn:** [B R Chaitra](https://www.linkedin.com/in/br-chaitra/)

