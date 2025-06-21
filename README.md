# Sign-Language-Recognition-using-Deep-Learning
 Real-time hand sign to sentence conversion using computer vision and deep learning (Streamlit + OpenCV + TensorFlow)
📌 Project Overview
This project is designed to recognize Indian Sign Language gestures in real-time using a webcam. It leverages computer vision and a trained deep learning model to detect hand gestures, classify them, and convert them into meaningful English sentences interactively through a web app built using Streamlit.

🚀 Features
🖐️ Real-time hand gesture detection using webcam

🔤 Recognition of multiple sign language words

✍️ Sentence formation using selected gestures

🧠 Trained deep learning model with keras_model.h5

🎯 User-friendly interface with live feedback

🗂️ Sentence templates loaded from sentences.json

🧠 Technologies Used
Python

TensorFlow / Keras

OpenCV

Streamlit

cvzone (Hand detection and classification)

NumPy, Math

JSON

🗂️ Folder Structure
nginx
Copy
Edit
Sign Language Recognition/
├── app.py                     # Streamlit app (possibly deprecated version)
├── main.py                   # Main application to run with Streamlit
├── back.py                   # Backend functions (e.g., for cleaning)
├── datacollection.py         # Data collector for building dataset
├── myenv/                    # Local virtual environment (do not upload)
├── sentences.json            # Predefined sentence structures
└── Finalyear_Project_Report_Group5.pdf  # Project report
🛠️ How to Run
Clone the Repository

bash
Copy
Edit
git clone https://github.com/yourusername/sign-language-recognition.git
cd sign-language-recognition
Create Virtual Environment (Recommended)

bash
Copy
Edit
python -m venv env
source env/bin/activate   # On Windows: env\Scripts\activate
Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
If requirements.txt is missing, you can use:

bash
Copy
Edit
pip install streamlit opencv-python cvzone tensorflow numpy
Run the App

bash
Copy
Edit
streamlit run main.py
Make Sure:

Your webcam is enabled

keras_model.h5 and labels.txt are correctly placed in the model directory

📄 License
This project is under the MIT License – feel free to use and modify.

🔗 Connect with Me:
📧 Email: chaitrabr03@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/b-r-chaitra/

