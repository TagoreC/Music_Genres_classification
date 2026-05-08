# Music Genres Classification using CNN

## Overview
This project is a deep learning-based web application that classifies music audio files into one of 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock. It uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras and a user-friendly interface powered by Streamlit.

## Dataset
The model was trained on the famous GTZAN dataset (often referred to as the MNIST of sounds). It consists of 1,000 audio tracks, each 30 seconds long, divided into 10 genres (100 tracks per genre). The audio files were converted into Mel Spectrograms to train the CNN effectively.

## Features
- **Upload Audio**: Users can upload `.mp3` audio files directly to the web app.
- **Audio Playback**: The app includes an audio player to listen to the uploaded file.
- **Genre Prediction**: Using the pre-trained CNN model (`my_model.keras`), the application predicts the genre of the uploaded audio.
- **Interactive UI**: An intuitive and visually appealing interface built with Streamlit.

## Project Structure
- `dlapp.py`: The main Streamlit web application script.
- `my_model.keras`: The pre-trained Convolutional Neural Network model.
- `Train_Music_Genre_Classifier.ipynb`: A Jupyter Notebook containing the code used for data preprocessing, model building, and training.

## Requirements
- Python 3.x
- Streamlit
- TensorFlow
- NumPy
- librosa

## How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Music_Genres_Classification.git
   cd Music_Genres_Classification
   ```
2. Install the required dependencies:
   ```bash
   pip install streamlit tensorflow numpy librosa
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run dlapp.py
   ```
