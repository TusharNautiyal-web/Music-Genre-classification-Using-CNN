# Music-Genre-classification-Using-CNN

<a href= '#'>**Check Deployment**</a>(Coming Soon)

# Description
Being an audio engineer myself i used my domain knowledge to analyse audio signal features and preprocess audio data. We use **GTZAN** Dataset which is a very popular dataset for Audio Classification and trained a CNN model I also tried ANN but it was not giving good accuracy so i went ahead with CNN for this project. This project is deployed on Hugging Face Spaces. For checking out how processing mfcc values whicha re used to understand the audio signal is done. I did took some refrencing from < a href = "https://www.researchgate.net/publication/324218667_Music_Genre_Classification_using_Machine_Learning_Techniques">Research Paper</a> this research paper.

# Libraries and Frameworks Used
Librosa
Tensorflow 
Numpy
Matplotlib
Streamlit

# How to check deployment
You can see the deployed project here <a href = '#'>**Check Deployment**</a> or if you want to use this code in your ***local machine***, install requirements using requirements.txt after that use this command when u are inside your directory of repository.

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



# Important Points to keep in mind for checking
The Uploaded sample of audio file should be of less then **30sec** and **.WAV** format for best results try to provide sections that have the most **elemental** or **instrumental ensemble** and should be of 30sec. If you want to test the model select ***Untrained Samples***. The model right now support only 10 genre which are blues, jazz, rock, metal, country, reagge, hiphop, pop, disco.

Thank you for checking out my repository do like this if you loved it.
