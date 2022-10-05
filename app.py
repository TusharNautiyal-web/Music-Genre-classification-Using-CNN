import numpy as np 
import librosa
import math
from tensorflow import keras
import streamlit as st
import time
import pyautogui

def get_mfcc(audio_signal, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    #This function will be extracting mfcc from our audio signal. 
    new_data = {
        "mfcc": []
    }

    SAMPLE_RATE = 22050
    signal,sample_rate = librosa.load(audio_signal,sr = SAMPLE_RATE)
    TRACK_DURATION = int(librosa.get_duration(signal)) # measured in seconds
    SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    for d in range(num_segments):
        # calculate start and finish sample for current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment
        # extract mfcc
        mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T
        # store only mfcc feature with expected number of vectors
        if len(mfcc) == num_mfcc_vectors_per_segment:
            new_data["mfcc"].append(mfcc.tolist())

    return new_data["mfcc"]

def prediction(mfcc):
    # This function will provide us with prediction labels from our CNN model.
    cnn_model = keras.models.load_model('music-gen-classify-v1/')
    mfcc = np.array(mfcc)
    mfcc = mfcc[...,np.newaxis]
    prediction = cnn_model.predict(mfcc)
    return max(np.argmax(prediction,axis = 1))

def get_genre(prediction):
    # This function will provide us with genre.
    pred = ''
    if prediction == 0:
        pred = 'Blues'
        
    elif prediction == 1:
        pred = 'Classical'
    
    elif prediction == 2:
        pred = 'Country'
    
    elif prediction == 3:
        pred = 'Disco'
    
    elif prediction == 4:
        pred = 'Hip Hop'
        
    elif prediction == 5:
        pred = 'Jazz'    
    
    elif prediction == 6:
        pred = 'Metal'
    
    elif prediction == 7:
        pred = 'Pop'
    
    elif prediction == 8:
        pred = 'Reggae'
    
    elif prediction == 9:
        pred = 'Rock'
    
    return pred


def main():
    # Few Instructions 
    # The music sample should not exceed more then 30 sec.
    # 0-> Blues 1-> classical 2-> country 3-> disco 4-> hiphop 5-> jazz 6-> metal 7-> pop 8-> reggae 9-> rock
    # Right now only 10 genres are supported as we used GTZAN Dataset for music Genre Classification.
    st.set_page_config(layout='wide',page_title='Genre Classification',page_icon='🎵')
    st.title('Music Genre Classifcation With CNN')
    st.markdown('We use **GTZAN** Dataset which is a very popular dataset for Audio Classification. The Uploaded sample of audio file should be of less then **30sec** and **.WAV** format for best results try to provide sections that have the most **elemental** or **instrumental ensemble** and should be of 30sec. If you want to test the model select ***Untrained Samples***. The model right now support only 10 genre which are blues, jazz, rock, metal, country, reagge, hiphop, pop, disco. A project by Tushar Nautiyal')        
    selected_item = st.selectbox('Select Either Uploaded Samples or Upload your own',['Untrained Samples','Upload'])
    # after this selection we will upload file or use a untrained Samples.
    if selected_item is not None:
        if selected_item == 'Upload':
            files = st.file_uploader('Select .WAV File with maximum 30sec Time', type='wav', accept_multiple_files=False)
            
            if files is not None:
                audio,sr = librosa.load(files,sr = 22050)
                duration = int(librosa.get_duration(audio))
                if 'file_uploaded' not in st.session_state:
                    st.session_state['file_uploaded'] = True
                
                if duration>30:
                    st.session_state['file_uploaded'] = False
                    st.write('Reupload File as it exceeds the time limit')
                    bar = st.progress(0)
                    i = 0
                    st.write('Reloading')
                    for percent_complete in range(100):
                        time.sleep(0.01)
                        bar.progress(i+1)
                        i = i+1       
                    pyautogui.hotkey('ctrl', 'F5')

                elif st.session_state['file_uploaded'] == True:
                    st.audio(files, format="audio/wav", start_time=0)              
        
        
        elif selected_item == 'Untrained Samples':
            selected_file = st.selectbox("Select A Sample", ['Blues','Jazz','Country','Classical','Hiphop','Metal','Pop','Reggae','Rock'])
            files = f'Data/upload/user/{selected_file}.wav'
            st.audio(files, format="audio/wav", start_time=0)
        submitted = st.button("Submit")
        
    if submitted:
        with st.spinner('Model is Trying to predict your genre! Wait for it'):
            signal = files
            mfcc_for_track = get_mfcc(signal)
    
            # After getting mfcc lets use our model to predict
            predict = prediction(mfcc_for_track)
            genre = get_genre(int(predict))
        st.success('Yes its Done and here is the answer!')
        st.markdown(f'The Genre for your music is 🎵  : **{genre}** Music')

if __name__ == '__main__':
    main()