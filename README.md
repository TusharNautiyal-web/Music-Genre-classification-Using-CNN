# Music-Genre-classification-Using-CNN

<a href="https://www.linkedin.com/in/tusharnautiyal/"> <img src = "https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/></a> 
<img src = "https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue"/> <img src = "https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white"/> 

<a href= 'https://huggingface.co/spaces/TusharNautiyal/Music-Genre-Classification'>**Check Deployment**</a>

# Description
Being an audio engineer myself i used my domain knowledge to analyse audio signal features and preprocess audio data. We use **GTZAN** Dataset which is a very popular dataset for Audio Classification and trained a CNN model I also tried ANN but it was not giving good accuracy so i went ahead with CNN for this project. This project is deployed on Hugging Face Spaces. For checking out how processing mfcc values whicha re used to understand the audio signal is done. I did took some refrencing from <a href = "https://www.researchgate.net/publication/324218667_Music_Genre_Classification_using_Machine_Learning_Techniques">Research Paper</a> this research paper.

# Libraries and Frameworks Used
Librosa
Tensorflow 
Numpy
Matplotlib
Streamlit

# How to check deployment
You can see the deployed project here <a href = 'https://huggingface.co/spaces/TusharNautiyal/Music-Genre-Classification'>**Check Deployment**</a> or if you want to use this code in your ***local machine***, install requirements using requirements.txt after that use this command when u are inside your directory of repository.

```
streamlit run app.py
```

# Directories 

This directory contains untrained samples that are used for testing model.

```
|_ Data
  |_ Upload
    |_ user
```

This is saved model directory.

```
|_ music-gen-classify-v1
  |_ asset
  |_ variables
    |_variables.data-00000-of-00001
    |_variables.index
  |_ keras_metadata.pb
  |_ saved_model.pb
```
# Demonstration Video


https://user-images.githubusercontent.com/74553737/194239602-12ca490f-3675-47fd-9d04-0a65da2a93fe.mp4


# How it works

***Understanding Some Common Terms***
 
   1.Sample Rate: For our this project we used default sample rate of 22050Khz in movies usally the music is exported in 44.1 kHz or 48 kHz Audio sample rate but our data was in 22050Khz. The sampling rate refers to the number of samples of audio recorded every second. It is measured in samples per second or Hertz (abbreviated as Hz or kHz, with one kHz being 1000 Hz). An audio sample is just a number representing the measured acoustic wave value at a specific point in time.
   
   2. Hop Length: This is also knows as buffer size or buffer frames in terms of recording audio signals this buffer size provides higher quality or lower quality where 2048 samples buffer is the highest and lowest can be 16samples the more samples the more the latency. In terms of audio signal or hop legnth the same thing becomes The hop size (number of samples between each successive FFT window) of Fast Fourier transforms performed is equal to the size of the Fast Fourier transform divided by the overlap factor (e.g. if the frame size is 512 and the overlap is set to 2 then the hop size is 256 samples
   
   3.FFT: The "Fast Fourier Transform" (FFT) is an important measurement method in the science of audio and acoustics measurement. It converts a signal into individual spectral components and thereby provides frequency information about the signal.
   
   4. MFCC vs FFT: FFT takes a signal and determines the 'frequency content' of the signal. MFFCs are perceptually motivated features to match how humans perceive pitch. To go one further, the perceptual motivation is to overcome the fact that the FFT has a linear resolution
   
   5.Cepstrum Vs Spectrum: When a signal is analyzed in time domain they are called spectrum or you can say that signal is in spectral domain. but when the signal is analyzed in frequency domain and amplitude of such signal is taken to analyzed the signal then they are said to be in cepstral domain
   
   6. Cepsteral:  

 ***Understanding MFCC***
 
Mel-frequency cepstral coefficients (MFCCs) are coefficients that collectively make up an Mel frequency cepstrum (See not Spectrum its Cepstrum). They are derived from a type of cepstral representation of the audio clip (a nonlinear "spectrum-of-a-spectrum"). The difference between the cepstrum and the mel-frequency cepstrum is that in the MFC, the frequency bands are equally spaced on the mel scale, which approximates the human auditory system's response more closely than the linearly-spaced frequency bands used in the normal spectrum. This frequency warping can allow for better representation of sound, for example, in audio compression that might potentially reduce the transmission bandwidth and the storage requirements of audio signals.


# Important Points to keep in mind for checking

The Uploaded sample of audio file should be of less then **30sec** and **.WAV** format for best results try to provide sections that have the most **elemental** or **instrumental ensemble** and should be of 30sec. If you want to test the model select ***Untrained Samples***. The model right now support only 10 genre which are blues, jazz, rock, metal, country, reagge, hiphop, pop, disco.

Thank you for checking out my repository do like this if you loved it. In future we will be creating more features like chroma features STFT etc to make our model more accurate and better with enough data if you want to try it out by yourself you can clone the repo and you can start working on it.
