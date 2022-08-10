import tensorflow as tf
import keras
import numpy as np
import librosa

MODEL_PATH="models/model1.h5"
SAMPLES_TO_CONSIDER=22050

class _keyword_spotting_service:
    model=None
    mappings=[
        "eight",
        "nine",
        "three",
        "one",
        "zero",
        "seven",
        "two",
        "six",
        "five",
        "four"
    ]
    instance=None

    def predict(self,file_path):
        #extract MFCCs
        MFCC=self.preprocess(file_path) 
        #convert 2D arrays into 4D arrays (#segments,coefficients)->(#samples,#segments,coefficients,# channels)
        MFCCs=MFCC[np.newaxis,...,np.newaxis]
        #make prediction
        
        predictions=self.model.predict(MFCCs) # array of probabilities of all labels. Example:[[0.14,0.22,0.12....]]
        predicted_index=np.argmax(predictions)
        predicted_keyword= self.mappings[predicted_index]
        
        print(predictions)
        return predicted_keyword
    def preprocess(self,file_path,n_mfcc=13):
        #load audio file
        signal,sr=librosa.load(file_path)
        # ensure consistency of all audio files
        if(len(signal)>=SAMPLES_TO_CONSIDER):
            signal=signal[:SAMPLES_TO_CONSIDER]
        #extract MFCCs
        MFCC=librosa.feature.mfcc(signal,sr=sr,n_mfcc=n_mfcc)
        return MFCC.T
        
def keyword_spotting_service():
    # ensuring we have only one instance of KSS 
    if _keyword_spotting_service.instance is None:
        _keyword_spotting_service.instance=_keyword_spotting_service()
        _keyword_spotting_service.model=tf.keras.models.load_model(MODEL_PATH)
    return _keyword_spotting_service.instance

if __name__ == "__main__":
    kss=keyword_spotting_service()
    k1=kss.predict("test/5.wav")
    

    print(f"predicted keyword is {k1}")

