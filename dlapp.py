import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import os


@st.cache_resource()
def load_model():
  model = tf.keras.models.load_model("C:/Users/vensunreddy/Downloads/my_model.keras")
  return model


def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    data = []
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    chunk_duration = 4  
    overlap_duration = 2  
                
    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate
                
    if len(audio_data) <= 0:
        return np.empty((0, *target_shape, 1), dtype=np.float32)

    if len(audio_data) <= chunk_samples:
        num_chunks = 1
    else:
        num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
                
    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
                    
        chunk = audio_data[start:end]
        if len(chunk) < chunk_samples:
            pad_width = int(chunk_samples - len(chunk))
            chunk = np.pad(chunk, (0, pad_width), mode="constant")
                    
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
                    

        resized = tf.image.resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape).numpy()
        data.append(resized)
    
    return np.array(data)



#Tensorflow Model Prediction
def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred,axis=1)
    unique_elements, counts = np.unique(predicted_categories, return_counts=True)
    #print(unique_elements, counts)
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]
    return max_elements[0]



#sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About Project","Prediction"])

## Main Page
if(app_mode=="Home"):
    st.markdown(
    """
    <style>
    .stApp {
        background-color: #181646;  /* Blue background */
        color: white;
    }
    h2, h3 {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

    st.markdown(''' ## Welcome to the,\n
    ## Music Genre Classification System! 🎶🎧''')
    image_path = "C:/Users/vensunreddy/Downloads/imagesfordlapp.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
*Our goal is to help in identifying music genres from audio tracks efficiently. Upload an audio file, and our system will analyze it to detect its genre. Discover the power of AI in music analysis!*

### How It Works
1. *Upload Audio:* Go to the *Genre Classification* page and upload an audio file.
2. *Analysis:* Our system will process the audio using advanced algorithms to classify it into one of the predefined genres.
3. *Results:* View the predicted genre along with related information.

### Why Choose Us?
- *Accuracy:* Our system leverages state-of-the-art deep learning models for accurate genre prediction.
- *User-Friendly:* Simple and intuitive interface for a smooth user experience.
- *Fast and Efficient:* Get results quickly, enabling faster music categorization and exploration.

### Get Started
Click on the *Genre Classification* page in the sidebar to upload an audio file and explore the magic of our Music Genre Classification System!

### About Us
Learn more about the project, our team, and our mission on the *About* page.
""")



#About Project
elif(app_mode=="About Project"):
    st.markdown("""
                ### About Project
                Music. Experts have been trying for a long time to understand sound and what differenciates one song from another. How to visualize sound. What makes a tone different from another.

                This data hopefully can give the opportunity to do just that.

                ### About Dataset
                #### Content
                1. *genres original* - A collection of 10 genres with 100 audio files each, all having a length of 30 seconds (the famous GTZAN dataset, the MNIST of sounds)
                2. *List of Genres* - blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
                3. *images original* - A visual representation for each audio file. One way to classify data is through neural networks. Because NNs (like CNN, what we will be using today) usually take in some sort of image representation, the audio files were converted to Mel Spectrograms to make this possible.
                4. *2 CSV files* - Containing features of the audio files. One file has for each song (30 seconds long) a mean and variance computed over multiple features that can be extracted from an audio file. The other file has the same structure, but the songs were split before into 3 seconds audio files (this way increasing 10 times the amount of data we fuel into our classification models). With data, more is always better.
                """)

    

#Prediction Page
elif(app_mode=="Prediction"):
    st.header("Model Prediction")
    test_mp3 = st.file_uploader("Upload an audio file", type=["mp3"])
    if test_mp3 is not None:
            os.makedirs('Test_Music', exist_ok=True)
            filepath = 'Test_Music/'+test_mp3.name
            with open(filepath, 'wb') as f:
                f.write(test_mp3.getbuffer())
            

    #Show Button
    if(st.button("Play Audio")):
        if test_mp3 is not None:
            st.audio(test_mp3)
        else:
            st.warning("Please upload an audio file first.")
    
    #Predict Button
    if(st.button("Predict")):
      if test_mp3 is None:
        st.error("Please upload an audio file before predicting.")
      else:
        with st.spinner("Please Wait.."):       
          X_test = load_and_preprocess_data(filepath)
          if X_test.size == 0:
            st.error("Could not process the uploaded audio. Please try another file.")
          else:
            result_index = model_prediction(X_test)
            st.balloons()
            label = ['blues', 'classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
            st.markdown(":blue[Model Prediction:] It's a  :red[{}] music**".format(label[result_index]))